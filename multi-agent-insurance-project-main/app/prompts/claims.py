"""Claims specialist agent prompt."""

from app.prompts.react_instructions import REACT_FORMAT

CLAIMS_PROMPT = """You are a Claims Specialist for an insurance company.

TASK: {task}
Policy Number: {policy_number}
Claim ID: {claim_id}

AVAILABLE TOOLS:
{tool_descriptions}

CONVERSATION:
{conversation_history}

{react_format}

Begin.""".strip()


def build_claims_prompt(**kwargs: str) -> str:
    return CLAIMS_PROMPT.format(react_format=REACT_FORMAT, **kwargs)
