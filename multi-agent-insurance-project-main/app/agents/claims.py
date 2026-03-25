"""Claims specialist agent."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.prompts.claims import build_claims_prompt
from app.services.llm import LLMClient
from app.tools.registry import ToolRegistry


class ClaimsAgent(BaseAgent):
    name = "claims_agent"

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        session: AsyncSession,
        conversation_id: str = "",
    ) -> None:
        claims_registry = ToolRegistry()
        spec = registry.get("get_claim_status")
        if spec:
            claims_registry.register(spec)
        super().__init__(llm, claims_registry, session, conversation_id)

    async def run(
        self,
        task: str,
        conversation_history: str,
        policy_number: str = "",
        claim_id: str = "",
    ) -> str:
        prompt = build_claims_prompt(
            task=task,
            policy_number=policy_number or "Not provided",
            claim_id=claim_id or "Not provided",
            conversation_history=conversation_history,
            tool_descriptions=self.registry.format_for_prompt(),
        )
        return await self.run_react_loop(prompt)
