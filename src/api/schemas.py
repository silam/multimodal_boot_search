"""HTTP request/response schemas (kept separate from domain models on purpose)."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Source(BaseModel):
    chunk_id: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    confident: bool
    citations: list[str]
    sources: list[Source]
