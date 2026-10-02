"""Factory: the single place that decides which concrete LLM to build."""

from docqa.config import Settings
from docqa.llm.base import LLMClient


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Set ANTHROPIC_API_KEY (or DOCQA_ANTHROPIC_API_KEY) to use Anthropic.")
        from docqa.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_tokens=settings.max_tokens,
        )
    from docqa.llm.fake_llm import FakeLLM

    return FakeLLM()
