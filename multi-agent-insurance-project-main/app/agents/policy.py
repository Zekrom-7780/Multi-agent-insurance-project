"""Policy specialist agent."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.prompts.policy import build_policy_prompt
from app.services.llm import LLMClient
from app.tools.registry import ToolRegistry


class PolicyAgent(BaseAgent):
    name = "policy_agent"

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        session: AsyncSession,
        conversation_id: str = "",
    ) -> None:
        # Only expose policy tools
        policy_registry = ToolRegistry()
        for tool_name in ("get_policy_details", "get_auto_policy_details"):
            spec = registry.get(tool_name)
            if spec:
                policy_registry.register(spec)
        super().__init__(llm, policy_registry, session, conversation_id)

    async def run(
        self,
        task: str,
        conversation_history: str,
        policy_number: str = "",
        customer_id: str = "",
    ) -> str:
        prompt = build_policy_prompt(
            task=task,
            policy_number=policy_number or "Not provided",
            customer_id=customer_id or "Not provided",
            conversation_history=conversation_history,
            tool_descriptions=self.registry.format_for_prompt(),
        )
        return await self.run_react_loop(prompt)
