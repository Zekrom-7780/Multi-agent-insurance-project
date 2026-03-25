"""Billing specialist agent prompt."""

from app.prompts.react_instructions import REACT_FORMAT

BILLING_PROMPT = """You are a Billing Specialist for an insurance company.

TASK: {task}

AVAILABLE TOOLS:
{tool_descriptions}

CONVERSATION:
{conversation_history}

{react_format}

Answer only what is asked. Do not provide extra information.
Begin.""".strip()


def build_billing_prompt(**kwargs: str) -> str:
    return BILLING_PROMPT.format(react_format=REACT_FORMAT, **kwargs)
