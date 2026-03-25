"""Tests for the /chat endpoint with a mocked LLM."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_chat_endpoint_returns_response():
    """Test that /chat returns a valid ChatResponse shape."""
    from app.main import app
    from app.services.session import SessionManager
    from app.services.context import ContextManager

    mock_final_state = {
        "final_answer": "Your premium is $250.",
        "agent_trace": [],
        "requires_human_escalation": False,
        "next_agent": "end",
        "conversation_id": "test-conv-1",
        "messages": [],
    }

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(return_value=mock_final_state)

    mock_graph = MagicMock()
    mock_graph.compile.return_value = mock_compiled

    app.state.session_mgr = SessionManager()
    app.state.graph = mock_graph
    app.state.context_mgr = ContextManager()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/chat",
            json={"message": "What is my premium?", "policy_number": "POL000001"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "conversation_id" in data
    assert data["message"] == "Your premium is $250."
    assert data["requires_human"] is False


@pytest.mark.asyncio
async def test_chat_multi_turn_same_conversation():
    """Test that multi-turn uses the same conversation_id."""
    from app.main import app
    from app.services.session import SessionManager
    from app.services.context import ContextManager

    mock_compiled = MagicMock()
    mock_compiled.ainvoke = AsyncMock(
        return_value={
            "final_answer": "Response",
            "agent_trace": [],
            "requires_human_escalation": False,
            "next_agent": "end",
            "conversation_id": "",
            "messages": [],
        }
    )
    mock_graph = MagicMock()
    mock_graph.compile.return_value = mock_compiled

    app.state.session_mgr = SessionManager()
    app.state.graph = mock_graph
    app.state.context_mgr = ContextManager()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp1 = await client.post("/chat", json={"message": "Hello"})
        cid = resp1.json()["conversation_id"]

        resp2 = await client.post(
            "/chat",
            json={"message": "Follow up", "conversation_id": cid},
        )

    assert resp2.json()["conversation_id"] == cid
