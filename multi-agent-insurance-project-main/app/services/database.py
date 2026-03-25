import json
import logging
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.models.db import AgentEvent, Base

logger = logging.getLogger(__name__)

engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db(url: str | None = None) -> None:
    """Create the async engine, session factory, and all tables."""
    global engine, async_session_factory
    db_url = url or settings.DATABASE_URL
    engine = create_async_engine(db_url, echo=False)
    async_session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialised: %s", db_url)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async session."""
    assert async_session_factory is not None, "Database not initialised"
    async with async_session_factory() as session:
        yield session


async def log_agent_event(
    session: AsyncSession,
    conversation_id: str,
    agent_name: str,
    event_type: str,
    payload: dict | None = None,
) -> None:
    """Insert an agent event row for observability."""
    event = AgentEvent(
        conversation_id=conversation_id,
        agent_name=agent_name,
        event_type=event_type,
        payload=json.dumps(payload or {}),
        created_at=datetime.utcnow(),
    )
    session.add(event)
    await session.commit()
