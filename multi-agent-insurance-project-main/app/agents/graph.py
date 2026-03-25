"""LangGraph state machine wiring — supervisor-routed multi-agent graph."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.billing import BillingAgent
from app.agents.claims import ClaimsAgent
from app.agents.escalation import EscalationAgent
from app.agents.final_answer import FinalAnswerAgent
from app.agents.general_help import GeneralHelpAgent
from app.agents.policy import PolicyAgent
from app.agents.supervisor import run_supervisor
from app.models.schemas import AgentStep
from app.models.state import ConversationState
from app.services.context import ContextManager
from app.services.llm import LLMClient
from app.services.rag import RAGService
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

_SPECIALIST_KEYWORDS: list[tuple[list[str], str]] = [
    (["bill", "premium", "payment", "pay", "due", "balance", "invoice"], "billing_agent"),
    (["claim", "file", "accident", "damage", "incident"], "claims_agent"),
    (["policy", "coverage", "deductible", "endorsement", "insur"], "policy_agent"),
]


def _infer_specialist(user_input: str) -> str:
    """Keyword-based fallback when the supervisor misroutes."""
    text = user_input.lower()
    for keywords, agent in _SPECIALIST_KEYWORDS:
        if any(kw in text for kw in keywords):
            return agent
    return "general_help_agent"


def build_graph(
    llm: LLMClient,
    registry: ToolRegistry,
    rag: RAGService,
    context_mgr: ContextManager,
) -> StateGraph:
    """Build and compile the LangGraph state machine."""

    # Import here to access the module-level factory after init_db()
    from app.services import database as _db

    async def _get_session():
        """Create a fresh async session from the module-level factory."""
        return _db.async_session_factory()

    graph = StateGraph(ConversationState)

    # ── Node functions ───────────────────────────────────────────

    async def supervisor_node(state: ConversationState) -> ConversationState:
        n_iter = state.get("n_iteration", 0) + 1
        state["n_iteration"] = n_iter

        if n_iter > MAX_ITERATIONS:
            logger.warning("Max iterations reached — escalating")
            state["next_agent"] = "human_escalation_agent"
            state["task"] = "Escalating due to max iterations"
            return state

        # Compress context if needed
        summary, messages = await context_mgr.maybe_compress(
            state.get("messages", []),
            state.get("context_summary"),
        )
        state["context_summary"] = summary
        state["messages"] = messages

        history = context_mgr.build_conversation_context(
            messages=state.get("messages", []),
            user_input=state.get("user_input", ""),
            context_summary=state.get("context_summary"),
            entities={
                k: state.get(k, "")
                for k in ("customer_id", "policy_number", "claim_id")
                if state.get(k)
            },
            task=state.get("task"),
        )

        async with await _get_session() as session:
            decision = await run_supervisor(
                llm=llm,
                session=session,
                conversation_id=state.get("conversation_id", ""),
                user_input=state.get("user_input", ""),
                conversation_history=history,
                customer_id=state.get("customer_id", ""),
                policy_number=state.get("policy_number", ""),
                claim_id=state.get("claim_id", ""),
            )

        # Guard: don't allow 'end' if no specialist has responded yet
        has_specialist_response = any(
            m.get("role") == "assistant" for m in state.get("messages", [])
        )
        if decision["next_agent"] == "end" and not has_specialist_response:
            decision["next_agent"] = _infer_specialist(
                state.get("user_input", "")
            )
            decision["task"] = decision.get("task") or state.get("user_input", "")
            logger.warning(
                "Supervisor tried to 'end' before any specialist responded, "
                "rerouting to %s",
                decision["next_agent"],
            )

        state["next_agent"] = decision["next_agent"]
        state["task"] = decision.get("task", "")
        state["justification"] = decision.get("justification", "")
        state["agent_trace"] = state.get("agent_trace", []) + [
            AgentStep(
                agent="supervisor",
                action="route",
                detail=f"-> {decision['next_agent']}: {decision.get('task','')}",
            )
        ]

        return state

    async def _run_specialist(
        state: ConversationState, agent_cls: type, **extra_kwargs: Any
    ) -> ConversationState:
        async with await _get_session() as session:
            # Determine if the agent class accepts a registry parameter
            init_params = agent_cls.__init__.__code__.co_varnames
            if "registry" in init_params:
                agent = agent_cls(
                    llm=llm,
                    registry=registry,
                    session=session,
                    conversation_id=state.get("conversation_id", ""),
                )
            else:
                agent = agent_cls(
                    llm=llm,
                    session=session,
                    conversation_id=state.get("conversation_id", ""),
                )

            run_kwargs = {
                "task": state.get("task", ""),
                "conversation_history": context_mgr.build_conversation_context(
                    messages=state.get("messages", []),
                    user_input=state.get("user_input", ""),
                    context_summary=state.get("context_summary"),
                ),
                **extra_kwargs,
            }
            result = await agent.run(**run_kwargs)

        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": result}
        ]
        state["agent_trace"] = state.get("agent_trace", []) + [
            AgentStep(agent=agent.name, action="respond", detail=result[:200])
        ]
        return state

    async def policy_node(state: ConversationState) -> ConversationState:
        return await _run_specialist(
            state,
            PolicyAgent,
            policy_number=state.get("policy_number", ""),
            customer_id=state.get("customer_id", ""),
        )

    async def billing_node(state: ConversationState) -> ConversationState:
        return await _run_specialist(state, BillingAgent)

    async def claims_node(state: ConversationState) -> ConversationState:
        return await _run_specialist(
            state,
            ClaimsAgent,
            policy_number=state.get("policy_number", ""),
            claim_id=state.get("claim_id", ""),
        )

    async def general_help_node(state: ConversationState) -> ConversationState:
        async with await _get_session() as session:
            agent = GeneralHelpAgent(
                llm=llm,
                rag=rag,
                session=session,
                conversation_id=state.get("conversation_id", ""),
            )
            result = await agent.run(
                task=state.get("task", ""),
                user_input=state.get("user_input", ""),
                conversation_history=context_mgr.build_conversation_context(
                    messages=state.get("messages", []),
                    user_input=state.get("user_input", ""),
                    context_summary=state.get("context_summary"),
                ),
            )
        state["messages"] = state.get("messages", []) + [
            {"role": "assistant", "content": result}
        ]
        state["agent_trace"] = state.get("agent_trace", []) + [
            AgentStep(agent="general_help_agent", action="respond", detail=result[:200])
        ]
        return state

    async def escalation_node(state: ConversationState) -> ConversationState:
        async with await _get_session() as session:
            agent = EscalationAgent(
                llm=llm, session=session, conversation_id=state.get("conversation_id", "")
            )
            result = await agent.run(
                task=state.get("task", ""),
                conversation_history=context_mgr.build_conversation_context(
                    messages=state.get("messages", []),
                    user_input=state.get("user_input", ""),
                ),
            )
        state["final_answer"] = result
        state["requires_human_escalation"] = True
        state["end_conversation"] = True
        state["agent_trace"] = state.get("agent_trace", []) + [
            AgentStep(agent="escalation_agent", action="escalate", detail=result[:200])
        ]
        return state

    async def final_answer_node(state: ConversationState) -> ConversationState:
        # Get the last assistant message as specialist response
        specialist = ""
        for msg in reversed(state.get("messages", [])):
            if msg.get("role") == "assistant":
                specialist = msg.get("content", "")
                break

        async with await _get_session() as session:
            agent = FinalAnswerAgent(
                llm=llm, session=session, conversation_id=state.get("conversation_id", "")
            )
            result = await agent.run(
                user_query=state.get("user_input", ""),
                specialist_response=specialist,
            )
        state["final_answer"] = result
        state["end_conversation"] = True
        state["agent_trace"] = state.get("agent_trace", []) + [
            AgentStep(agent="final_answer_agent", action="rewrite", detail=result[:200])
        ]
        return state

    # ── Register nodes ───────────────────────────────────────────

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("policy_agent", policy_node)
    graph.add_node("billing_agent", billing_node)
    graph.add_node("claims_agent", claims_node)
    graph.add_node("general_help_agent", general_help_node)
    graph.add_node("human_escalation_agent", escalation_node)
    graph.add_node("final_answer", final_answer_node)

    graph.set_entry_point("supervisor")

    # ── Routing ──────────────────────────────────────────────────

    def route_supervisor(state: ConversationState) -> str:
        next_agent = state.get("next_agent", "general_help_agent")
        if next_agent == "end":
            return "final_answer"
        if next_agent == "need_clarification":
            return END  # return question to user; next HTTP request resumes
        if next_agent in (
            "policy_agent",
            "billing_agent",
            "claims_agent",
            "general_help_agent",
            "human_escalation_agent",
        ):
            return next_agent
        return "general_help_agent"

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "policy_agent": "policy_agent",
            "billing_agent": "billing_agent",
            "claims_agent": "claims_agent",
            "general_help_agent": "general_help_agent",
            "human_escalation_agent": "human_escalation_agent",
            "final_answer": "final_answer",
            END: END,
        },
    )

    # Specialists loop back to supervisor
    for node in ("policy_agent", "billing_agent", "claims_agent", "general_help_agent"):
        graph.add_edge(node, "supervisor")

    # Terminal nodes
    graph.add_edge("final_answer", END)
    graph.add_edge("human_escalation_agent", END)

    return graph
