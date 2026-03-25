"""SSE endpoint for streaming agent events in real time."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.models.db import AgentEvent
from app.services.database import get_db_session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/chat/{conversation_id}/events")
async def stream_events(
    conversation_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Stream agent events for a conversation as SSE."""

    async def event_generator():
        last_id = 0
        while True:
            result = await session.execute(
                text(
                    """
                    SELECT id, agent_name, event_type, payload, created_at
                    FROM agent_events
                    WHERE conversation_id = :cid AND id > :last_id
                    ORDER BY id ASC
                    """
                ),
                {"cid": conversation_id, "last_id": last_id},
            )
            rows = result.mappings().all()

            for row in rows:
                last_id = row["id"]
                data = {
                    "agent_name": row["agent_name"],
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload"]) if row["payload"] else {},
                    "timestamp": str(row["created_at"]),
                }
                yield {"event": row["event_type"], "data": json.dumps(data)}

                # Terminate on end event
                if row["event_type"] == "end":
                    return

            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())
