"""Human escalation agent — no tools, empathetic handoff message."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.prompts.escalation import ESCALATION_PROMPT
from app.services.database import log_agent_event
from app.services.llm import LLMClient

logger = logging.getLogger(__name__)


class EscalationAgent:
    name = "escalation_agent"

    def __init__(
        self,
        llm: LLMClient,
        session: AsyncSession,
        conversation_id: str = "",
    ) -> None:
        self.llm = llm
        self.session = session
        self.conversation_id = conversation_id

    async def run(self, task: str, conversation_history: str) -> str:
        await log_agent_event(
            self.session,
            self.conversation_id,
            self.name,
            "escalation_started",
        )

        prompt = ESCALATION_PROMPT.format(
            task=task,
            conversation_history=conversation_history,
        )

        return await self.llm.complete(prompt, max_tokens=200)
