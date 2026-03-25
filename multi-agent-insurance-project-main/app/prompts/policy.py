"""Policy specialist agent prompt."""

from app.prompts.react_instructions import REACT_FORMAT

POLICY_PROMPT = """You are a Policy Specialist for an insurance company.

TASK: {task}
Policy Number: {policy_number}
Customer ID: {customer_id}

AVAILABLE TOOLS:
{tool_descriptions}

CONVERSATION:
{conversation_history}

{react_format}

Begin.""".strip()


def build_policy_prompt(**kwargs: str) -> str:
    return POLICY_PROMPT.format(react_format=REACT_FORMAT, **kwargs)
