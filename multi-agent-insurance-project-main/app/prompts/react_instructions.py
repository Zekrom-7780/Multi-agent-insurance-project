"""Shared ReAct format instructions injected into every tool-using agent prompt."""

REACT_FORMAT = """
RESPONSE FORMAT (follow exactly):

To use a tool, respond:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <JSON arguments>

When you have the final answer, respond:
Thought: <your reasoning>
Final Answer: <your response to the user>

RULES:
- Use EXACTLY one Action per response
- Action Input MUST be valid JSON
- After receiving an Observation, continue with Thought/Action or give Final Answer
- Do NOT invent data — only use information from tool Observations
""".strip()
