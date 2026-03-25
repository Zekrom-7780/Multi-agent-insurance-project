"""Base agent with a ReAct loop that replaces OpenAI's native function calling."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import LLMError, LLMParseError, ToolExecutionError
from app.services.database import log_agent_event
from app.services.llm import LLMClient
from app.tools.registry import ToolRegistry
from app.utils.react_parser import ReactOutput, parse_react_output

logger = logging.getLogger(__name__)


class BaseAgent:
    """ReAct-loop agent that iteratively calls tools via text parsing."""

    name: str = "base"

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        session: AsyncSession,
        conversation_id: str = "",
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.session = session
        self.conversation_id = conversation_id

    async def run_react_loop(
        self,
        prompt: str,
        max_steps: int = 3,
    ) -> str:
        """Execute the ReAct loop: Thought -> Action -> Observation -> ... -> Final Answer."""
        known_tools = self.registry.names()
        working_prompt = prompt
        last_observation: str | None = None
        prev_call: tuple[str | None, str] | None = None  # (action, args_key)

        for step in range(1, max_steps + 1):
            await self._log_event("react_step_start", {"step": step})

            try:
                result = await self.llm.react_step(
                    working_prompt, known_tools=known_tools
                )
            except Exception as exc:
                logger.error("LLM call failed at step %d: %s", step, exc)
                # Retry once with a format reminder
                if step == 1:
                    working_prompt += (
                        "\n\nIMPORTANT: Respond with Thought/Action/Action Input "
                        "or Thought/Final Answer format."
                    )
                    try:
                        result = await self.llm.react_step(
                            working_prompt, known_tools=known_tools
                        )
                    except Exception:
                        raise LLMError(str(exc)) from exc
                else:
                    raise LLMError(str(exc)) from exc

            await self._log_event(
                "react_step_parsed",
                {
                    "step": step,
                    "parse_method": result.parse_method,
                    "action": result.action,
                    "has_final_answer": result.final_answer is not None,
                },
            )

            # If we got a final answer, we're done
            if result.final_answer is not None:
                await self._log_event("react_final_answer", {"answer_length": len(result.final_answer)})
                return result.final_answer

            # Execute the tool
            if result.action:
                # Detect duplicate tool call — same action + same args as last step
                call_key = (result.action, json.dumps(result.action_input, sort_keys=True))
                if call_key == prev_call and last_observation:
                    logger.warning(
                        "Duplicate tool call detected (%s), forcing final answer from observation",
                        result.action,
                    )
                    return self._answer_from_observation(last_observation)

                prev_call = call_key

                observation = await self._execute_tool(
                    result.action, result.action_input or {}
                )
                last_observation = observation

                # Append observation with a nudge to answer
                working_prompt += (
                    f"\n\nThought: {result.thought}\n"
                    f"Action: {result.action}\n"
                    f"Action Input: {json.dumps(result.action_input)}\n"
                    f"Observation: {observation}\n"
                    f"You now have the data. Respond with:\n"
                    f"Thought: I have the information.\n"
                    f"Final Answer: <answer based on the Observation above>"
                )
            else:
                # No action and no final answer — treat raw as answer
                return result.raw

        # Exhausted max steps — synthesize from last observation if available
        if last_observation:
            logger.warning("Max ReAct steps (%d) reached for %s, using last observation", max_steps, self.name)
            return self._answer_from_observation(last_observation)
        logger.warning("Max ReAct steps (%d) reached for %s", max_steps, self.name)
        return result.raw if result else "I was unable to complete the request."

    @staticmethod
    def _answer_from_observation(observation: str) -> str:
        """Extract a usable answer from a raw tool observation."""
        try:
            data = json.loads(observation)
            if isinstance(data, dict) and "error" not in data:
                return json.dumps(data, indent=2, default=str)
        except (json.JSONDecodeError, ValueError):
            pass
        return observation

    @staticmethod
    def _remap_args(spec: "ToolSpec", args: dict[str, Any]) -> dict[str, Any]:
        """Remap a generic 'input' key to the correct parameter name.

        When the LLM produces a plain string instead of JSON for Action Input,
        the parser wraps it as ``{"input": raw_string}``.  This method uses
        the tool's parameter schema to map it to the real parameter name.
        """
        if "input" not in args:
            return args

        properties = spec.parameters.get("properties", {})
        if "input" in properties:
            return args  # tool genuinely expects 'input'

        value = args["input"]
        rest = {k: v for k, v in args.items() if k != "input"}

        logger.debug("_remap_args: tool=%s raw_input=%r", spec.name, value)

        # 1) Single required parameter — use it directly
        required = spec.parameters.get("required", [])
        if len(required) == 1:
            return {**rest, required[0]: value}

        # 2) Search for entity ID patterns *anywhere* in the value
        if isinstance(value, str):
            _ENTITY_PATTERNS: list[tuple[str, str]] = [
                (r"POL\d+", "policy_number"),
                (r"CUST\d+", "customer_id"),
                (r"CLM\d+", "claim_id"),
            ]
            for pattern, param in _ENTITY_PATTERNS:
                m = re.search(pattern, value, re.IGNORECASE)
                if m and param in properties:
                    logger.info(
                        "_remap_args: mapped 'input' -> %s=%s",
                        param, m.group(),
                    )
                    return {**rest, param: m.group()}

        # 3) Only one property in the schema — safe to assume
        if len(properties) == 1:
            return {**rest, next(iter(properties)): value}

        # 4) Last resort — use the first property rather than crashing
        if properties:
            first_prop = next(iter(properties))
            logger.warning(
                "_remap_args: no pattern matched for tool %s, "
                "defaulting to '%s' with value %r",
                spec.name, first_prop, value,
            )
            return {**rest, first_prop: value}

        return args

    @staticmethod
    def _flatten_arg_values(args: dict[str, Any]) -> dict[str, Any]:
        """Ensure all arg values are simple types (str/int/float/bool).

        The Q3 quantised model sometimes produces nested dicts like
        ``{"policy_number": {"policy_number": "POL000004"}}`` which
        breaks SQL parameter binding.
        """
        flat: dict[str, Any] = {}
        for key, value in args.items():
            while isinstance(value, dict):
                # Unwrap: take the first value from the nested dict
                value = next(iter(value.values()))
            if isinstance(value, list):
                value = value[0] if value else ""
            flat[key] = value
        return flat

    async def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Look up tool in registry, inject DB session, call, and return result as string."""
        spec = self.registry.get(tool_name)
        if spec is None:
            msg = f"Unknown tool: {tool_name}"
            await self._log_event("tool_not_found", {"tool": tool_name})
            return json.dumps({"error": msg})

        args = self._remap_args(spec, args)
        args = self._flatten_arg_values(args)

        try:
            # All tool functions expect session as first arg
            result = await spec.fn(self.session, **args)
            await self._log_event(
                "tool_executed",
                {"tool": tool_name, "args": args, "success": True},
            )
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_name, exc)
            await self._log_event(
                "tool_executed",
                {"tool": tool_name, "args": args, "success": False, "error": str(exc)},
            )
            return json.dumps({"error": f"Tool execution failed: {exc}"})

    async def _log_event(self, event_type: str, payload: dict | None = None) -> None:
        """Write an event to the agent_events table."""
        try:
            await log_agent_event(
                self.session,
                conversation_id=self.conversation_id,
                agent_name=self.name,
                event_type=event_type,
                payload=payload,
            )
        except Exception as exc:
            logger.warning("Failed to log agent event: %s", exc)
