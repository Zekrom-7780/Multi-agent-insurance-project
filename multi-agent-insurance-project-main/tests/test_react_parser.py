"""Extensive tests for the layered ReAct parser."""

import json

import pytest

from app.utils.react_parser import parse_react_output


class TestJSONLayer:
    def test_well_formed_json_action(self):
        text = json.dumps({
            "thought": "Need to look up the policy",
            "action": "get_policy_details",
            "action_input": {"policy_number": "POL000001"},
        })
        r = parse_react_output(text)
        assert r.parse_method == "json"
        assert r.action == "get_policy_details"
        assert r.action_input == {"policy_number": "POL000001"}
        assert r.thought == "Need to look up the policy"

    def test_well_formed_json_final_answer(self):
        text = json.dumps({
            "thought": "I have the answer",
            "final_answer": "Your premium is $250.",
        })
        r = parse_react_output(text)
        assert r.parse_method == "json"
        assert r.final_answer == "Your premium is $250."

    def test_markdown_fenced_json(self):
        text = '```json\n{"action": "get_billing_info", "action_input": {"policy_number": "POL000001"}}\n```'
        r = parse_react_output(text)
        assert r.parse_method == "json"
        assert r.action == "get_billing_info"

    def test_action_input_as_string(self):
        text = json.dumps({
            "action": "get_claim_status",
            "action_input": '{"claim_id": "CLM000001"}',
        })
        r = parse_react_output(text)
        assert r.parse_method == "json"
        assert r.action_input == {"claim_id": "CLM000001"}


class TestRegexLayer:
    def test_react_format(self):
        text = (
            "Thought: I should look up the policy\n"
            "Action: get_policy_details\n"
            'Action Input: {"policy_number": "POL000001"}'
        )
        r = parse_react_output(text)
        assert r.parse_method == "regex"
        assert r.action == "get_policy_details"
        assert r.action_input == {"policy_number": "POL000001"}
        assert "look up" in r.thought

    def test_final_answer_format(self):
        text = (
            "Thought: I have all the info\n"
            "Final Answer: Your policy is active with a $250 premium."
        )
        r = parse_react_output(text)
        assert r.parse_method == "regex"
        assert "250" in r.final_answer

    def test_action_input_not_json(self):
        text = "Action: get_policy_details\nAction Input: POL000001"
        r = parse_react_output(text)
        assert r.parse_method == "regex"
        assert r.action_input == {"input": "POL000001"}


class TestKeywordLayer:
    def test_tool_name_in_text(self):
        text = "I need to call get_billing_info for policy POL000001"
        r = parse_react_output(text, known_tools=["get_billing_info"])
        assert r.parse_method == "keyword"
        assert r.action == "get_billing_info"

    def test_tool_name_with_json_after(self):
        text = 'Let me use get_claim_status {"claim_id": "CLM000001"}'
        r = parse_react_output(text, known_tools=["get_claim_status"])
        assert r.parse_method == "keyword"
        assert r.action_input.get("claim_id") == "CLM000001"


class TestFallbackLayer:
    def test_plain_text(self):
        text = "Your premium is $250 per month."
        r = parse_react_output(text)
        assert r.parse_method == "fallback"
        assert r.final_answer == text

    def test_empty_input(self):
        r = parse_react_output("")
        assert r.parse_method == "fallback"
        assert r.final_answer == "I could not determine a response."

    def test_none_input(self):
        r = parse_react_output(None)
        assert r.parse_method == "fallback"

    def test_garbled_output(self):
        text = "!@#$%^&*() random garbage without any structure"
        r = parse_react_output(text)
        assert r.parse_method == "fallback"
        assert r.final_answer is not None
