from typing import Any, Optional

from typing_extensions import TypedDict

from app.models.schemas import AgentStep


class ConversationState(TypedDict, total=False):
    # Core
    messages: list[dict[str, str]]
    user_input: str
    conversation_id: str

    # Extracted context
    customer_id: Optional[str]
    policy_number: Optional[str]
    claim_id: Optional[str]

    # Routing
    next_agent: Optional[str]
    task: Optional[str]
    justification: Optional[str]

    # Control
    n_iteration: int
    end_conversation: bool
    requires_human_escalation: bool

    # Output
    final_answer: Optional[str]
    agent_trace: list[AgentStep]

    # Context management (phi-3-mini 4k window)
    context_summary: Optional[str]
    token_count: int
