"""Provider-agnostic LLM interface. The rest of the app depends on this, never on an SDK."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def complete_structured(self, system: str, user: str, schema: type[T]) -> T:
        """Return the model's reply parsed and validated into `schema`."""
        ...
