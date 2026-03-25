"""Shared test fixtures: in-memory DB with seed data, MockLLMClient, temp ChromaDB."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.db import Base
from app.services.llm import LLMClient
from app.utils.react_parser import ReactOutput


# ── In-memory async DB ───────────────────────────────────────────

@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def seeded_session(db_session: AsyncSession):
    """Insert minimal seed data for tool tests."""
    await db_session.execute(
        text(
            "INSERT INTO customers VALUES "
            "('CUST00001','John','Smith','john@test.com','555-000-0000','1990-01-01','CA')"
        )
    )
    await db_session.execute(
        text(
            "INSERT INTO policies VALUES "
            "('POL000001','CUST00001','auto','2023-06-01',250.00,'monthly','active')"
        )
    )
    await db_session.execute(
        text(
            "INSERT INTO auto_policy_details VALUES "
            "('POL000001','VIN12345678901234567','Toyota','Camry',2021,"
            "100000,500,500,1,1)"
        )
    )
    await db_session.execute(
        text(
            "INSERT INTO billing VALUES "
            "('BILL000001','POL000001','2024-01-01','2024-01-15',250.00,'pending')"
        )
    )
    await db_session.execute(
        text(
            "INSERT INTO payments VALUES "
            "('PAY000001','BILL000001','2024-01-10',250.00,'credit_card','TXN100001','completed')"
        )
    )
    await db_session.execute(
        text(
            "INSERT INTO claims VALUES "
            "('CLM000001','POL000001','2024-02-01','collision',5000.00,'under_review')"
        )
    )
    await db_session.commit()
    yield db_session


# ── Mock LLM ────────────────────────────────────────────────────

class MockLLMClient:
    """LLM client that returns pre-configured responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self._call_count = 0

    async def complete(self, prompt: str, **kwargs) -> str:
        if self._responses:
            resp = self._responses[self._call_count % len(self._responses)]
        else:
            resp = '{"next_agent": "end", "task": "done", "justification": "test"}'
        self._call_count += 1
        return resp

    async def react_step(self, prompt: str, known_tools=None, **kwargs) -> ReactOutput:
        raw = await self.complete(prompt, **kwargs)
        from app.utils.react_parser import parse_react_output
        return parse_react_output(raw, known_tools=known_tools)
