"""Billing-related tool functions."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_billing_info(
    session: AsyncSession,
    policy_number: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """Get the most recent pending bill for a policy or customer."""
    if policy_number:
        result = await session.execute(
            text(
                """
                SELECT b.*, p.premium_amount, p.billing_frequency
                FROM billing b
                JOIN policies p ON b.policy_number = p.policy_number
                WHERE b.policy_number = :val AND b.status = 'pending'
                ORDER BY b.due_date DESC LIMIT 1
                """
            ),
            {"val": policy_number},
        )
    elif customer_id:
        result = await session.execute(
            text(
                """
                SELECT b.*, p.premium_amount, p.billing_frequency
                FROM billing b
                JOIN policies p ON b.policy_number = p.policy_number
                WHERE p.customer_id = :val AND b.status = 'pending'
                ORDER BY b.due_date DESC LIMIT 1
                """
            ),
            {"val": customer_id},
        )
    else:
        return {"error": "Provide policy_number or customer_id"}

    row = result.mappings().first()
    if row:
        return dict(row)
    return {"error": "Billing information not found"}


async def get_payment_history(
    session: AsyncSession, policy_number: str
) -> list[dict]:
    """Return the last 10 payments for a policy."""
    result = await session.execute(
        text(
            """
            SELECT p.payment_date, p.amount, p.status, p.payment_method
            FROM payments p
            JOIN billing b ON p.bill_id = b.bill_id
            WHERE b.policy_number = :pn
            ORDER BY p.payment_date DESC LIMIT 10
            """
        ),
        {"pn": policy_number},
    )
    rows = result.mappings().all()
    return [dict(r) for r in rows] if rows else []
