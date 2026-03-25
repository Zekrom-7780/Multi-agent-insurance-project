"""Billing specialist agent."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.prompts.billing import build_billing_prompt
from app.services.llm import LLMClient
from app.tools.registry import ToolRegistry


class BillingAgent(BaseAgent):
    name = "billing_agent"

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        session: AsyncSession,
        conversation_id: str = "",
    ) -> None:
        billing_registry = ToolRegistry()
        for tool_name in ("get_billing_info", "get_payment_history"):
            spec = registry.get(tool_name)
            if spec:
                billing_registry.register(spec)
        super().__init__(llm, billing_registry, session, conversation_id)

    async def run(self, task: str, conversation_history: str) -> str:
        prompt = build_billing_prompt(
            task=task,
            conversation_history=conversation_history,
            tool_descriptions=self.registry.format_for_prompt(),
        )
        return await self.run_react_loop(prompt)
