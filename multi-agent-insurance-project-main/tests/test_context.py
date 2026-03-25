"""Tests for context manager: entity extraction, token budget, compression."""

import pytest

from app.services.context import ContextManager


def test_extract_entities_policy():
    cm = ContextManager()
    entities = cm.extract_entities("My policy number is POL000123")
    assert entities["policy_number"] == "POL000123"


def test_extract_entities_customer():
    cm = ContextManager()
    entities = cm.extract_entities("Customer CUST00456 called in")
    assert entities["customer_id"] == "CUST00456"


def test_extract_entities_claim():
    cm = ContextManager()
    entities = cm.extract_entities("I filed CLM000789 last week")
    assert entities["claim_id"] == "CLM000789"


def test_extract_entities_multiple():
    cm = ContextManager()
    entities = cm.extract_entities("POL000001 belongs to CUST00001, claim CLM000001")
    assert len(entities) == 3


def test_extract_entities_none():
    cm = ContextManager()
    entities = cm.extract_entities("What does life insurance cover?")
    assert len(entities) == 0


def test_build_context_fits_budget():
    cm = ContextManager()
    # Create a large message list
    messages = [
        {"role": "user", "content": f"Message {i} " + "x" * 100}
        for i in range(50)
    ]
    context = cm.build_conversation_context(
        messages=messages,
        user_input="Latest question",
    )
    from app.utils.token_counter import count_tokens
    assert count_tokens(context) <= 3200


def test_build_context_includes_entities():
    cm = ContextManager()
    context = cm.build_conversation_context(
        messages=[],
        user_input="Tell me about my policy",
        entities={"policy_number": "POL000001"},
    )
    assert "POL000001" in context


def test_build_context_includes_summary():
    cm = ContextManager()
    context = cm.build_conversation_context(
        messages=[],
        user_input="Next question",
        context_summary="Previously discussed billing for POL000001",
    )
    assert "Previously discussed" in context


@pytest.mark.asyncio
async def test_maybe_compress_below_threshold():
    cm = ContextManager(llm=None)
    messages = [{"role": "user", "content": "Hello"}]
    summary, remaining = await cm.maybe_compress(messages)
    assert summary is None
    assert remaining == messages
