"""Tool registry: metadata + async callables, rendered as text for ReAct prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., Coroutine[Any, Any, Any]]


class ToolRegistry:
    """Register, look up, and format tools for text-based ReAct prompts."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def format_for_prompt(self) -> str:
        """Render tool descriptions as text for injection into ReAct prompts."""
        if not self._tools:
            return "No tools available."
        lines: list[str] = []
        for spec in self._tools.values():
            params_str = json.dumps(spec.parameters, indent=2)
            lines.append(
                f"Tool: {spec.name}\n"
                f"Description: {spec.description}\n"
                f"Parameters: {params_str}"
            )
        return "\n\n".join(lines)
