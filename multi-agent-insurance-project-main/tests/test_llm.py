"""Tests for LLM client with mocked AsyncOpenAI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.llm import LLMClient


@pytest.mark.asyncio
async def test_react_step_json_action():
    raw_response = '{"action": "get_policy_details", "action_input": {"policy_number": "POL000001"}}'

    mock_choice = MagicMock()
    mock_choice.message.content = raw_response
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch("app.services.llm.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=mock_resp)

        client = LLMClient(base_url="http://fake", api_key="fake")
        result = await client.react_step("test prompt")

    assert result.parse_method == "json"
    assert result.action == "get_policy_details"


@pytest.mark.asyncio
async def test_react_step_final_answer():
    raw_response = "Thought: done\nFinal Answer: Your premium is $250."

    mock_choice = MagicMock()
    mock_choice.message.content = raw_response
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch("app.services.llm.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=mock_resp)

        client = LLMClient(base_url="http://fake", api_key="fake")
        result = await client.react_step("test prompt")

    assert result.final_answer is not None
    assert "250" in result.final_answer


@pytest.mark.asyncio
async def test_complete_returns_raw_text():
    raw = "Hello, world!"

    mock_choice = MagicMock()
    mock_choice.message.content = raw
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]

    with patch("app.services.llm.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(return_value=mock_resp)

        client = LLMClient(base_url="http://fake", api_key="fake")
        result = await client.complete("prompt")

    assert result == "Hello, world!"
