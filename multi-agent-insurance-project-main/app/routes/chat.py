"""POST /chat endpoint — runs the multi-agent graph and returns the response."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse
from app.services.context import ContextManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
) -> ChatResponse:
    """Accept a user message, run the agent graph, return the answer."""
    # Retrieve services from app.state (set in lifespan)
    session_mgr = request.app.state.session_mgr
    graph = request.app.state.graph
    context_mgr: ContextManager = request.app.state.context_mgr

    # Get or create conversation
    cid, state = session_mgr.get_or_create(body.conversation_id)

    # Inject user input and extracted entities
    state["user_input"] = body.message
    state["messages"] = state.get("messages", []) + [
        {"role": "user", "content": body.message}
    ]

    # Merge any provided identifiers (don't overwrite with None)
    if body.customer_id:
        state["customer_id"] = body.customer_id
    if body.policy_number:
        state["policy_number"] = body.policy_number
    if body.claim_id:
        state["claim_id"] = body.claim_id

    # Extract entities from user message
    entities = context_mgr.extract_entities(body.message)
    for key, value in entities.items():
        if value and not state.get(key):
            state[key] = value

    # Reset per-turn control fields
    state["end_conversation"] = False
    state["requires_human_escalation"] = False
    state["final_answer"] = None
    state["next_agent"] = None
    state["n_iteration"] = 0
    state["agent_trace"] = []

    # Run the graph (sessions are created inside graph nodes)
    compiled = graph.compile()
    final_state = await compiled.ainvoke(state)

    # Persist state
    session_mgr.update(cid, final_state)

    answer = final_state.get("final_answer") or ""
    if not answer:
        # If supervisor returned need_clarification, task contains the question
        if final_state.get("next_agent") == "need_clarification":
            answer = final_state.get("task", "Could you provide more details?")

    return ChatResponse(
        conversation_id=cid,
        message=answer,
        agent_trace=final_state.get("agent_trace", []),
        requires_human=final_state.get("requires_human_escalation", False),
    )
