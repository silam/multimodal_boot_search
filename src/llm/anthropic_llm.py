"""Anthropic adapter. All SDK details (retries, tool-use for JSON, tracing) stay in this file."""

from typing import Any, TypeVar, cast

import anthropic
from pydantic import BaseModel

from docqa.observability.logging import get_logger, timed

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=3)
        self._model = model
        self._max_tokens = max_tokens

    def complete_structured(self, system: str, user: str, schema: type[T]) -> T:
        # Force a single tool call whose input schema is our Pydantic model -> reliable JSON.
        tool: dict[str, Any] = {
            "name": "respond",
            "description": "Return the final answer in this structure.",
            "input_schema": schema.model_json_schema(),
        }
        with timed(log, "llm.call", model=self._model):
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[cast(Any, tool)],
                tool_choice={"type": "tool", "name": "respond"},
            )
        log.info(
            "llm.usage",
            extra={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )
        for block in resp.content:
            if block.type == "tool_use":
                return schema.model_validate(block.input)
        raise RuntimeError("Model did not return structured output")
