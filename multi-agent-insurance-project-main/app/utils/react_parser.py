"""
Layered ReAct output parser for phi-3-mini-4k-instruct.

phi-3-mini doesn't support native function calling, so we parse its free-text
output through four progressively looser layers:
  1. JSON extraction (```json block or top-level JSON)
  2. Regex (Action: / Action Input: lines)
  3. Keyword (known tool name anywhere in text)
  4. Fallback (entire response treated as Final Answer)

Each layer sets `parse_method` so we can track reliability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ReactOutput:
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[dict[str, Any]] = None
    final_answer: Optional[str] = None
    raw: str = ""
    parse_method: str = "unknown"


# ── Layer helpers ────────────────────────────────────────────────

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_ACTION_RE = re.compile(
    r"Action\s*:\s*(.+?)(?:\n|$).*?Action\s*Input\s*:\s*(.+)",
    re.DOTALL | re.IGNORECASE,
)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"Final\s*Answer\s*:\s*(.+)", re.DOTALL | re.IGNORECASE
)


def _extract_thought(text: str) -> str:
    m = _THOUGHT_RE.search(text)
    return m.group(1).strip() if m else ""


def _try_parse_json(text: str) -> Optional[dict]:
    """Attempt to parse a string as JSON, return None on failure."""
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ── Layer 1: JSON extraction ────────────────────────────────────

def _layer_json(text: str) -> Optional[ReactOutput]:
    # Try fenced block first
    m = _JSON_BLOCK_RE.search(text)
    candidates = []
    if m:
        candidates.append(m.group(1))
    # Try the whole text as JSON
    candidates.append(text)

    for candidate in candidates:
        obj = _try_parse_json(candidate)
        if obj is None:
            continue
        if "action" in obj and "action_input" in obj:
            action_input = obj["action_input"]
            if isinstance(action_input, str):
                action_input = _try_parse_json(action_input) or {"input": action_input}
            return ReactOutput(
                thought=obj.get("thought", _extract_thought(text)),
                action=obj["action"],
                action_input=action_input,
                raw=text,
                parse_method="json",
            )
        if "final_answer" in obj:
            return ReactOutput(
                thought=obj.get("thought", ""),
                final_answer=obj["final_answer"],
                raw=text,
                parse_method="json",
            )
    return None


# ── Layer 2: Regex ───────────────────────────────────────────────

def _layer_regex(text: str) -> Optional[ReactOutput]:
    # Check for Final Answer first
    fa_match = _FINAL_ANSWER_RE.search(text)
    action_match = _ACTION_RE.search(text)

    if action_match:
        action = action_match.group(1).strip()
        raw_input = action_match.group(2).strip()
        action_input = _try_parse_json(raw_input) or {"input": raw_input}
        return ReactOutput(
            thought=_extract_thought(text),
            action=action,
            action_input=action_input,
            raw=text,
            parse_method="regex",
        )

    if fa_match:
        return ReactOutput(
            thought=_extract_thought(text),
            final_answer=fa_match.group(1).strip(),
            raw=text,
            parse_method="regex",
        )

    return None


# ── Layer 3: Keyword (known tool names) ─────────────────────────

def _layer_keyword(
    text: str, known_tools: list[str] | None = None
) -> Optional[ReactOutput]:
    if not known_tools:
        return None

    text_lower = text.lower()
    for tool in known_tools:
        if tool.lower() in text_lower:
            # Try to find any JSON blob after the tool name
            idx = text_lower.index(tool.lower())
            remainder = text[idx + len(tool) :]
            action_input = _try_parse_json(remainder.strip()) or {}
            # Also scan full text for any JSON object
            if not action_input:
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    action_input = _try_parse_json(json_match.group()) or {}
            return ReactOutput(
                thought=_extract_thought(text),
                action=tool,
                action_input=action_input,
                raw=text,
                parse_method="keyword",
            )
    return None


# ── Layer 4: Fallback ───────────────────────────────────────────

def _layer_fallback(text: str) -> ReactOutput:
    return ReactOutput(
        thought="",
        final_answer=text.strip() if text.strip() else "I could not determine a response.",
        raw=text,
        parse_method="fallback",
    )


# ── Public API ──────────────────────────────────────────────────

def parse_react_output(
    text: str, known_tools: list[str] | None = None
) -> ReactOutput:
    """Parse LLM output through 4 layers, returning the first successful parse."""
    if not text or not text.strip():
        return _layer_fallback(text or "")

    for layer_fn in [
        lambda t: _layer_json(t),
        lambda t: _layer_regex(t),
        lambda t: _layer_keyword(t, known_tools),
    ]:
        result = layer_fn(text)
        if result is not None:
            return result

    return _layer_fallback(text)
