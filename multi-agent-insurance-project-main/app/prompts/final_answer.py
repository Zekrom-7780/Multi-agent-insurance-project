"""Final answer agent prompt — rewrites specialist response for the user."""

FINAL_ANSWER_PROMPT = """The user asked: "{user_query}"

The specialist provided:
{specialist_response}

Rewrite the specialist's answer as a short, friendly reply to the user.
Rules:
- Maximum 2-3 sentences
- Answer the question directly, then offer further help
- No technical details, no internal instructions, no tool calls
- Do NOT repeat yourself or produce multiple versions

Reply:""".strip()
