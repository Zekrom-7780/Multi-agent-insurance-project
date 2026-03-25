"""Policy-related tool functions."""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_policy_details(
    session: AsyncSession, policy_number: str
) -> dict:
    """Fetch policy details joined with customer name."""
    result = await session.execute(
        text(
            """
            SELECT p.policy_number, p.customer_id, p.policy_type,
                   p.start_date, p.premium_amount, p.billing_frequency,
                   p.status, c.first_name, c.last_name
            FROM policies p
            JOIN customers c ON p.customer_id = c.customer_id
            WHERE p.policy_number = :pn
            """
        ),
        {"pn": policy_number},
    )
    row = result.mappings().first()
    if row:
        return dict(row)
    return {"error": "Policy not found"}


async def get_auto_policy_details(
    session: AsyncSession, policy_number: str
) -> dict:
    """Fetch auto-specific policy details."""
    result = await session.execute(
        text(
            """
            SELECT apd.*, p.policy_type, p.premium_amount
            FROM auto_policy_details apd
            JOIN policies p ON apd.policy_number = p.policy_number
            WHERE apd.policy_number = :pn
            """
        ),
        {"pn": policy_number},
    )
    row = result.mappings().first()
    if row:
        return dict(row)
    return {"error": "Auto policy details not found"}
