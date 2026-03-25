"""In-memory session manager for conversation state."""

from __future__ import annotations

import uuid
from typing import Any

from app.models.state import ConversationState


class SessionManager:
    """Stores ConversationState per conversation_id in memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(
        self, conversation_id: str | None = None
    ) -> tuple[str, ConversationState]:
        """Return existing session or create a new one."""
        if conversation_id and conversation_id in self._sessions:
            return conversation_id, self._sessions[conversation_id]

        cid = conversation_id or str(uuid.uuid4())
        state: ConversationState = {
            "messages": [],
            "user_input": "",
            "conversation_id": cid,
            "customer_id": None,
            "policy_number": None,
            "claim_id": None,
            "next_agent": None,
            "task": None,
            "justification": None,
            "n_iteration": 0,
            "end_conversation": False,
            "requires_human_escalation": False,
            "final_answer": None,
            "agent_trace": [],
            "context_summary": None,
            "token_count": 0,
        }
        self._sessions[cid] = state
        return cid, state

    def update(self, conversation_id: str, state: ConversationState) -> None:
        self._sessions[conversation_id] = state

    def delete(self, conversation_id: str) -> None:
        self._sessions.pop(conversation_id, None)

    def get(self, conversation_id: str) -> ConversationState | None:
        return self._sessions.get(conversation_id)
