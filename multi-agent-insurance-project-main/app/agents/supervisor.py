"""Supervisor agent — routes to specialists via JSON parsing (no tools)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.database import log_agent_event
from app.services.llm import LLMClient
from app.prompts.supervisor import SUPERVISOR_PROMPT

logger = logging.getLogger(__name__)

_VALID_AGENTS = {
    "policy_agent",
    "billing_agent",
    "claims_agent",
    "general_help_agent",
    "human_escalation_agent",
    "end",
    "need_clarification",
}

# Regex fallback for extracting agent name
_AGENT_NAME_RE = re.compile(
    r"(policy_agent|billing_agent|claims_agent|general_help_agent|"
    r"human_escalation_agent|need_clarification|end)",
    re.IGNORECASE,
)

# JSON block extraction
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_supervisor_output(raw: str) -> dict[str, str]:
    """Parse supervisor output with layered fallback: JSON -> regex -> default."""
    # Layer 1: Try full JSON parse
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and "next_agent" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: Find JSON block in text
    match = _JSON_RE.search(raw)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "next_agent" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Layer 3: Keyword scan for agent name
    agent_match = _AGENT_NAME_RE.search(raw)
    if agent_match:
        agent = agent_match.group(1).lower()
        return {
            "next_agent": agent,
            "task": "Routed via keyword fallback",
            "justification": "JSON parse failed; matched agent name in text",
        }

    # Layer 4: Default to general_help
    logger.warning("Supervisor parse failed, defaulting to general_help_agent")
    return {
        "next_agent": "general_help_agent",
        "task": "Assist the user with their query",
        "justification": "Could not parse supervisor output",
    }


async def run_supervisor(
    llm: LLMClient,
    session: AsyncSession,
    conversation_id: str,
    user_input: str,
    conversation_history: str,
    customer_id: str = "",
    policy_number: str = "",
    claim_id: str = "",
) -> dict[str, str]:
    """Call the supervisor LLM and return routing decision."""
    prompt = SUPERVISOR_PROMPT.format(
        conversation_history=conversation_history,
        user_input=user_input,
        customer_id=customer_id or "unknown",
        policy_number=policy_number or "unknown",
        claim_id=claim_id or "unknown",
    )

    raw = await llm.complete(prompt, temperature=0.1, max_tokens=150)
    decision = _parse_supervisor_output(raw)

    # Normalise agent name
    next_agent = decision.get("next_agent", "general_help_agent").lower().strip()
    if next_agent not in _VALID_AGENTS:
        next_agent = "general_help_agent"
    decision["next_agent"] = next_agent

    await log_agent_event(
        session,
        conversation_id=conversation_id,
        agent_name="supervisor",
        event_type="routing_decision",
        payload=decision,
    )

    logger.info(
        "Supervisor -> %s (task=%s)", decision["next_agent"], decision.get("task", "")
    )
    return decision
