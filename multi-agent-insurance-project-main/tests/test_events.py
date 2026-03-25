"""Tests for the SSE events endpoint."""

import json

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import AgentEvent
from app.services.database import log_agent_event


@pytest.mark.asyncio
async def test_log_events_sequential(db_session: AsyncSession):
    """Test that events are logged with sequential IDs."""
    await log_agent_event(db_session, "conv-sse-1", "supervisor", "routing_decision", {"next_agent": "billing_agent"})
    await log_agent_event(db_session, "conv-sse-1", "billing_agent", "tool_executed", {"tool": "get_billing_info"})
    await log_agent_event(db_session, "conv-sse-1", "final_answer", "end", {})

    result = await db_session.execute(
        text("SELECT * FROM agent_events WHERE conversation_id = 'conv-sse-1' ORDER BY id")
    )
    rows = result.mappings().all()
    assert len(rows) == 3
    assert rows[0]["agent_name"] == "supervisor"
    assert rows[1]["agent_name"] == "billing_agent"
    assert rows[2]["event_type"] == "end"
