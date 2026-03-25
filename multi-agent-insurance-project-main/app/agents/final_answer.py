"""Final answer agent — rewrites specialist response into a clean user-facing answer."""

from __future__ import annotations

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.prompts.final_answer import FINAL_ANSWER_PROMPT
from app.services.database import log_agent_event
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)

# Pattern that signals the model is starting to repeat itself
_REPETITION_RE = re.compile(r"\n---|\n\n\n")


class FinalAnswerAgent:
    name = "final_answer_agent"

    def __init__(
        self,
        llm: LLMClient,
        session: AsyncSession,
        conversation_id: str = "",
    ) -> None:
        self.llm = llm
        self.session = session
        self.conversation_id = conversation_id

    async def run(self, user_query: str, specialist_response: str) -> str:
        await log_agent_event(
            self.session,
            self.conversation_id,
            self.name,
            "rewrite_started",
        )

        prompt = FINAL_ANSWER_PROMPT.format(
            user_query=user_query,
            specialist_response=specialist_response,
        )

        raw = await self.llm.complete(
            prompt, max_tokens=150, frequency_penalty=1.2,
        )
        return self._trim_repetition(raw)

    @staticmethod
    def _trim_repetition(text: str) -> str:
        """Cut the response at the first repetition marker (e.g. '---')."""
        m = _REPETITION_RE.search(text)
        if m:
            text = text[: m.start()]
        return text.strip()
