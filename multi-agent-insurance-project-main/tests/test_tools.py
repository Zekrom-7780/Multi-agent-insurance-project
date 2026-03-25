"""Tests for tool functions with in-memory seeded DB."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import BaseAgent
from app.tools.policy_tools import get_policy_details, get_auto_policy_details
from app.tools.billing_tools import get_billing_info, get_payment_history
from app.tools.claims_tools import get_claim_status
from app.tools.registry import ToolSpec


@pytest.mark.asyncio
async def test_get_policy_details(seeded_session: AsyncSession):
    result = await get_policy_details(seeded_session, "POL000001")
    assert result["policy_number"] == "POL000001"
    assert result["first_name"] == "John"
    assert result["premium_amount"] == 250.00


@pytest.mark.asyncio
async def test_get_policy_not_found(seeded_session: AsyncSession):
    result = await get_policy_details(seeded_session, "POL999999")
    assert "error" in result


@pytest.mark.asyncio
async def test_get_auto_policy_details(seeded_session: AsyncSession):
    result = await get_auto_policy_details(seeded_session, "POL000001")
    assert result["vehicle_make"] == "Toyota"
    assert result["vehicle_model"] == "Camry"


@pytest.mark.asyncio
async def test_get_billing_info(seeded_session: AsyncSession):
    result = await get_billing_info(seeded_session, policy_number="POL000001")
    assert result["bill_id"] == "BILL000001"
    assert result["amount_due"] == 250.00


@pytest.mark.asyncio
async def test_get_payment_history(seeded_session: AsyncSession):
    result = await get_payment_history(seeded_session, "POL000001")
    assert len(result) == 1
    assert result[0]["amount"] == 250.00


@pytest.mark.asyncio
async def test_get_claim_status_by_id(seeded_session: AsyncSession):
    result = await get_claim_status(seeded_session, claim_id="CLM000001")
    assert isinstance(result, list)
    assert result[0]["claim_id"] == "CLM000001"
    assert result[0]["status"] == "under_review"


@pytest.mark.asyncio
async def test_get_claim_status_by_policy(seeded_session: AsyncSession):
    result = await get_claim_status(seeded_session, policy_number="POL000001")
    assert isinstance(result, list)
    assert len(result) >= 1


# ── _remap_args tests ───────────────────────────────────────────


def _make_spec(name: str, properties: dict, required: list) -> ToolSpec:
    """Helper to build a ToolSpec with a given parameter schema."""
    async def _noop(session):
        pass
    return ToolSpec(
        name=name,
        description="test",
        parameters={"type": "object", "properties": properties, "required": required},
        fn=_noop,
    )


class TestRemapArgs:
    """Verify _remap_args correctly maps generic 'input' to real param names."""

    def test_no_input_key_unchanged(self):
        spec = _make_spec("t", {"policy_number": {"type": "string"}}, ["policy_number"])
        args = {"policy_number": "POL000001"}
        assert BaseAgent._remap_args(spec, args) == {"policy_number": "POL000001"}

    def test_tool_expects_input_unchanged(self):
        spec = _make_spec("t", {"input": {"type": "string"}}, ["input"])
        args = {"input": "some value"}
        assert BaseAgent._remap_args(spec, args) == {"input": "some value"}

    def test_single_required_param(self):
        spec = _make_spec("get_policy_details",
                          {"policy_number": {"type": "string"}},
                          ["policy_number"])
        result = BaseAgent._remap_args(spec, {"input": "POL000001"})
        assert result == {"policy_number": "POL000001"}

    def test_policy_number_pattern_match(self):
        spec = _make_spec("get_billing_info",
                          {"policy_number": {"type": "string"},
                           "customer_id": {"type": "string"}},
                          [])
        result = BaseAgent._remap_args(spec, {"input": "POL000004"})
        assert result == {"policy_number": "POL000004"}

    def test_customer_id_pattern_match(self):
        spec = _make_spec("get_billing_info",
                          {"policy_number": {"type": "string"},
                           "customer_id": {"type": "string"}},
                          [])
        result = BaseAgent._remap_args(spec, {"input": "CUST00001"})
        assert result == {"customer_id": "CUST00001"}

    def test_claim_id_pattern_match(self):
        spec = _make_spec("get_claim_status",
                          {"claim_id": {"type": "string"},
                           "policy_number": {"type": "string"}},
                          [])
        result = BaseAgent._remap_args(spec, {"input": "CLM000001"})
        assert result == {"claim_id": "CLM000001"}

    def test_single_property_fallback(self):
        spec = _make_spec("t", {"query": {"type": "string"}}, [])
        result = BaseAgent._remap_args(spec, {"input": "hello"})
        assert result == {"query": "hello"}

    def test_entity_embedded_in_text(self):
        """LLM outputs 'policy POL000004' instead of just 'POL000004'."""
        spec = _make_spec("get_billing_info",
                          {"policy_number": {"type": "string"},
                           "customer_id": {"type": "string"}},
                          [])
        result = BaseAgent._remap_args(spec, {"input": "policy POL000004"})
        assert result == {"policy_number": "POL000004"}

    def test_entity_with_surrounding_text(self):
        """LLM outputs 'get billing for CUST00001 please'."""
        spec = _make_spec("get_billing_info",
                          {"policy_number": {"type": "string"},
                           "customer_id": {"type": "string"}},
                          [])
        result = BaseAgent._remap_args(spec, {"input": "get billing for CUST00001 please"})
        assert result == {"customer_id": "CUST00001"}

    def test_last_resort_uses_first_property(self):
        """When no pattern matches, falls back to first schema property."""
        spec = _make_spec("get_billing_info",
                          {"policy_number": {"type": "string"},
                           "customer_id": {"type": "string"}},
                          [])
        result = BaseAgent._remap_args(spec, {"input": "unknown_value"})
        assert result == {"policy_number": "unknown_value"}

    def test_no_properties_returns_unchanged(self):
        spec = _make_spec("t", {}, [])
        result = BaseAgent._remap_args(spec, {"input": "value"})
        assert result == {"input": "value"}
