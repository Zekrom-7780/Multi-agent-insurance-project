"""LLM client abstraction for phi-3-mini via LM Studio (OpenAI-compatible API)."""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import settings
from app.utils.react_parser import ReactOutput, parse_react_output

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around AsyncOpenAI pointed at a local LM Studio server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self._model = model or settings.LLM_MODEL
        self._client = AsyncOpenAI(
            base_url=base_url or settings.LLM_BASE_URL,
            api_key=api_key or settings.LLM_API_KEY,
        )

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 800,
        frequency_penalty: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Send a single system-prompt completion and return the raw text."""
        logger.debug("LLM request (%d chars)", len(prompt))
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=[{"role": "system", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
        )
        if stop:
            kwargs["stop"] = stop
        resp = await self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        logger.debug("LLM response (%d chars, method=%s)", len(text), "raw")
        return text

    # Stop sequences that prevent the model from hallucinating
    # its own Observation or starting a second Thought/Action cycle.
    _REACT_STOP = ["Observation:", "observation:", "\n---", "\n\n\n"]

    async def react_step(
        self,
        prompt: str,
        known_tools: list[str] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 250,
    ) -> ReactOutput:
        """Complete and parse through the layered ReAct parser."""
        raw = await self.complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=self._REACT_STOP,
        )
        result = parse_react_output(raw, known_tools=known_tools)
        logger.info(
            "ReAct parse: method=%s action=%s final_answer=%s",
            result.parse_method,
            result.action,
            bool(result.final_answer),
        )
        return result
