"""Tests for database models, CRUD, and agent event logging."""

import json

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import AgentEvent, Customer, Policy
from app.services.database import log_agent_event


@pytest.mark.asyncio
async def test_create_customer(db_session: AsyncSession):
    await db_session.execute(
        text(
            "INSERT INTO customers VALUES "
            "('CUST99999','Test','User','t@t.com','555-111-1111','2000-01-01','TX')"
        )
    )
    await db_session.commit()
    result = await db_session.execute(
        select(Customer).where(Customer.customer_id == "CUST99999")
    )
    c = result.scalar_one()
    assert c.first_name == "Test"
    assert c.state == "TX"


@pytest.mark.asyncio
async def test_policy_foreign_key(seeded_session: AsyncSession):
    result = await seeded_session.execute(
        select(Policy).where(Policy.policy_number == "POL000001")
    )
    p = result.scalar_one()
    assert p.customer_id == "CUST00001"
    assert p.premium_amount == 250.00


@pytest.mark.asyncio
async def test_log_agent_event(db_session: AsyncSession):
    await log_agent_event(
        db_session,
        conversation_id="conv-123",
        agent_name="supervisor",
        event_type="routing_decision",
        payload={"next_agent": "billing_agent"},
    )
    result = await db_session.execute(
        select(AgentEvent).where(AgentEvent.conversation_id == "conv-123")
    )
    event = result.scalar_one()
    assert event.agent_name == "supervisor"
    assert json.loads(event.payload)["next_agent"] == "billing_agent"
