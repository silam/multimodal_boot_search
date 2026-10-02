"""Domain models. Pure data — no LLM, DB, or web imports here."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str
    source: str


class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    source: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class Answer(BaseModel):
    """Structured output we ask the LLM to return."""

    answer: str
    citations: list[str] = Field(default_factory=list, description="chunk ids used")
    confident: bool = True
