"""General help agent — RAG-backed, no tools."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.prompts.general_help import GENERAL_HELP_PROMPT
from app.services.database import log_agent_event
from app.services.llm import LLMClient
from app.services.rag import RAGService

logger = logging.getLogger(__name__)


class GeneralHelpAgent:
    name = "general_help_agent"

    def __init__(
        self,
        llm: LLMClient,
        rag: RAGService,
        session: AsyncSession,
        conversation_id: str = "",
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.session = session
        self.conversation_id = conversation_id

    async def run(
        self,
        task: str,
        user_input: str,
        conversation_history: str,
    ) -> str:
        # Retrieve FAQs
        faqs = self.rag.retrieve(user_input, n_results=3)
        faq_context = self.rag.format_for_prompt(faqs)

        await log_agent_event(
            self.session,
            self.conversation_id,
            self.name,
            "faq_retrieved",
            {"n_results": len(faqs)},
        )

        prompt = GENERAL_HELP_PROMPT.format(
            task=task,
            conversation_history=conversation_history,
            faq_context=faq_context,
        )

        return await self.llm.complete(prompt)
