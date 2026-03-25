"""Exception hierarchy for the insurance agent API."""


class InsuranceAgentError(Exception):
    """Base exception for the application."""

    status_code: int = 500

    def __init__(self, message: str = "Internal server error"):
        self.message = message
        super().__init__(message)


class LLMError(InsuranceAgentError):
    """LLM service unavailable or returned an error."""

    status_code = 502

    def __init__(self, message: str = "LLM service error"):
        super().__init__(message)


class LLMParseError(InsuranceAgentError):
    """Could not parse LLM output into the expected structure."""

    status_code = 502

    def __init__(self, message: str = "Failed to parse LLM response"):
        super().__init__(message)


class ToolExecutionError(InsuranceAgentError):
    """A tool function raised an exception."""

    status_code = 500

    def __init__(self, message: str = "Tool execution failed"):
        super().__init__(message)


class SessionNotFoundError(InsuranceAgentError):
    """Conversation session not found."""

    status_code = 404

    def __init__(self, message: str = "Session not found"):
        super().__init__(message)


class MaxIterationsError(InsuranceAgentError):
    """Agent loop exceeded the maximum allowed iterations."""

    status_code = 500

    def __init__(self, message: str = "Maximum iterations exceeded"):
        super().__init__(message)


class TokenBudgetExceeded(InsuranceAgentError):
    """Context exceeded the token budget for the model."""

    status_code = 400

    def __init__(self, message: str = "Token budget exceeded"):
        super().__init__(message)
