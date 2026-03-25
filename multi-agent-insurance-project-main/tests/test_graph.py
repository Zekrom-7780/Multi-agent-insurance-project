"""Tests for the LangGraph wiring with mocked agents."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agents.supervisor import _parse_supervisor_output


class TestSupervisorParser:
    def test_valid_json(self):
        raw = '{"next_agent": "billing_agent", "task": "Get billing info", "justification": "User asked about billing"}'
        result = _parse_supervisor_output(raw)
        assert result["next_agent"] == "billing_agent"
        assert result["task"] == "Get billing info"

    def test_json_in_text(self):
        raw = 'Here is my decision: {"next_agent": "policy_agent", "task": "Look up policy", "justification": "test"}'
        result = _parse_supervisor_output(raw)
        assert result["next_agent"] == "policy_agent"

    def test_keyword_fallback(self):
        raw = "I think we should route to billing_agent because the user asked about payment."
        result = _parse_supervisor_output(raw)
        assert result["next_agent"] == "billing_agent"

    def test_default_fallback(self):
        raw = "I'm not sure what to do here."
        result = _parse_supervisor_output(raw)
        assert result["next_agent"] == "general_help_agent"

    def test_escalation_keyword(self):
        raw = "The user wants human_escalation_agent help."
        result = _parse_supervisor_output(raw)
        assert result["next_agent"] == "human_escalation_agent"

    def test_end_routing(self):
        raw = '{"next_agent": "end", "task": "Done", "justification": "Answered"}'
        result = _parse_supervisor_output(raw)
        assert result["next_agent"] == "end"

    def test_need_clarification(self):
        raw = '{"next_agent": "need_clarification", "task": "Ask for policy number", "justification": "Missing"}'
        result = _parse_supervisor_output(raw)
        assert result["next_agent"] == "need_clarification"
