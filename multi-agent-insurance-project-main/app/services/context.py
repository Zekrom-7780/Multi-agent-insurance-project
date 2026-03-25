"""Context management for phi-3-mini's 4k token window.

Provides sliding-window history, entity extraction, and optional compression.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.utils.token_counter import count_tokens, truncate_to_tokens

if TYPE_CHECKING:
    from app.services.llm import LLMClient

logger = logging.getLogger(__name__)

# Budget: ~4096 total, reserve 800 for prompt template + response
MAX_CONTEXT_TOKENS = 3200
COMPRESSION_THRESHOLD = 2400

_ENTITY_PATTERNS = {
    "policy_number": re.compile(r"POL\d{6}"),
    "customer_id": re.compile(r"CUST\d{5}"),
    "claim_id": re.compile(r"CLM\d{6}"),
}


class ContextManager:
    """Manages conversation context within phi-3-mini's token budget."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def extract_entities(self, text: str) -> dict[str, str]:
        """Extract policy/customer/claim IDs from text via regex."""
        found: dict[str, str] = {}
        for key, pattern in _ENTITY_PATTERNS.items():
            m = pattern.search(text)
            if m:
                found[key] = m.group()
        return found

    def build_conversation_context(
        self,
        messages: list[dict[str, str]],
        user_input: str,
        context_summary: str | None = None,
        entities: dict[str, str] | None = None,
        task: str | None = None,
    ) -> str:
        """Build a context string that fits within the token budget.

        Priority (highest first):
          1. Current user_input + extracted entities + current task
          2. Newest messages (sliding window, newest first)
          3. context_summary (prepended if room)
        """
        parts: list[str] = []

        # Always include current input
        current = f"User: {user_input}"
        if entities:
            ent_str = ", ".join(f"{k}={v}" for k, v in entities.items())
            current += f"\n[Entities: {ent_str}]"
        if task:
            current += f"\n[Task: {task}]"
        parts.append(current)

        budget = MAX_CONTEXT_TOKENS - count_tokens(current)

        # Add messages newest-first until budget exhausted
        for msg in reversed(messages):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            line = f"{role.capitalize()}: {content}"
            line_tokens = count_tokens(line)
            if line_tokens > budget:
                break
            parts.insert(0, line)
            budget -= line_tokens

        # Prepend summary if there's room
        if context_summary and budget > 50:
            summary_line = f"[Summary of earlier conversation: {context_summary}]"
            summary_tokens = count_tokens(summary_line)
            if summary_tokens <= budget:
                parts.insert(0, summary_line)

        return "\n".join(parts)

    async def maybe_compress(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None = None,
    ) -> tuple[str | None, list[dict[str, str]]]:
        """If messages exceed COMPRESSION_THRESHOLD, summarise older ones.

        Returns (new_summary, remaining_messages).
        """
        total = sum(count_tokens(m.get("content", "")) for m in messages)
        if total < COMPRESSION_THRESHOLD or self._llm is None:
            return current_summary, messages

        # Keep the last 4 messages, summarise the rest
        keep = messages[-4:]
        to_compress = messages[:-4]
        if not to_compress:
            return current_summary, messages

        text_block = "\n".join(
            f"{m.get('role','')}: {m.get('content','')}" for m in to_compress
        )
        prompt = (
            "Summarise the following insurance support conversation in 2-3 sentences. "
            "Preserve policy numbers, customer IDs, claim IDs, and key decisions.\n\n"
            f"{truncate_to_tokens(text_block, 1500)}"
        )
        summary = await self._llm.complete(prompt, max_tokens=200)
        logger.info("Context compressed: %d msgs -> summary + %d msgs", len(to_compress), len(keep))
        return summary, keep
