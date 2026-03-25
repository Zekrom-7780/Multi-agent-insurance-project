"""Claims-related tool functions."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_claim_status(
    session: AsyncSession,
    claim_id: str | None = None,
    policy_number: str | None = None,
) -> list[dict] | dict:
    """Get claim status by claim_id or the latest 3 claims for a policy."""
    if claim_id:
        result = await session.execute(
            text(
                """
                SELECT c.*, p.policy_type
                FROM claims c
                JOIN policies p ON c.policy_number = p.policy_number
                WHERE c.claim_id = :cid
                """
            ),
            {"cid": claim_id},
        )
    elif policy_number:
        result = await session.execute(
            text(
                """
                SELECT c.*, p.policy_type
                FROM claims c
                JOIN policies p ON c.policy_number = p.policy_number
                WHERE c.policy_number = :pn
                ORDER BY c.claim_date DESC LIMIT 3
                """
            ),
            {"pn": policy_number},
        )
    else:
        return {"error": "Provide claim_id or policy_number"}

    rows = result.mappings().all()
    if rows:
        return [dict(r) for r in rows]
    return {"error": "Claim not found"}
