"""Tests for exception hierarchy and HTTP error handler."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.exceptions import (
    InsuranceAgentError,
    LLMError,
    LLMParseError,
    SessionNotFoundError,
    ToolExecutionError,
)


def test_base_exception_defaults():
    exc = InsuranceAgentError()
    assert exc.status_code == 500
    assert exc.message == "Internal server error"


def test_llm_error():
    exc = LLMError("LLM down")
    assert exc.status_code == 502
    assert exc.message == "LLM down"


def test_session_not_found():
    exc = SessionNotFoundError()
    assert exc.status_code == 404


def test_tool_execution_error():
    exc = ToolExecutionError("DB connection failed")
    assert exc.status_code == 500
    assert exc.message == "DB connection failed"


def test_llm_parse_error():
    exc = LLMParseError()
    assert exc.status_code == 502
