"""Central configuration. Every tunable knob lives here, read from env vars / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCQA_", extra="ignore")

    # LLM
    llm_provider: Literal["anthropic", "fake"] = "fake"
    anthropic_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("DOCQA_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )
    anthropic_model: str = "claude-sonnet-5-5"
    max_tokens: int = 800

    # Retrieval
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3
    embedding_dim: int = 512

    # Paths
    docs_dir: Path = Path("data/docs")
    index_path: Path = Path("data/index.json")

    # Prompts
    answer_prompt: str = "answer_v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
