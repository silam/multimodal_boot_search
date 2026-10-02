"""Service tests use a stub LLM — no network, fully deterministic."""

from typing import TypeVar

from pydantic import BaseModel

from docqa.core.models import Answer
from docqa.prompts.loader import render
from docqa.retrieval.retriever import Retriever
from docqa.services.qa_service import QAService

T = TypeVar("T", bound=BaseModel)


class StubLLM:
    def __init__(self, reply: dict[str, object]) -> None:
        self.reply = reply
        self.last_user: str | None = None

    def complete_structured(self, system: str, user: str, schema: type[T]) -> T:
        self.last_user = user
        return schema.model_validate(self.reply)


def test_prompt_includes_context_and_question(retriever: Retriever) -> None:
    hits = retriever.retrieve("dogs")
    system, user = render("answer_v1", chunks=hits, question="Q?", max_words=50)
    assert "ONLY the provided context" in system
    assert 'id="dogs#0"' in user and user.endswith("Question: Q?")


def test_service_drops_hallucinated_citations(retriever: Retriever) -> None:
    llm = StubLLM({"answer": "Walk them daily.", "citations": ["dogs#0", "made-up#9"]})
    answer, _ = QAService(retriever, llm, "answer_v1").ask("do dogs need walks?")
    assert answer.citations == ["dogs#0"]
    assert llm.last_user is not None and "Question: do dogs need walks?" in llm.last_user


def test_service_returns_unknown_without_context(retriever: Retriever) -> None:
    llm = StubLLM({"answer": "should not be called"})
    answer, hits = QAService(retriever, llm, "answer_v1").ask("quantum chromodynamics")
    assert hits == []
    assert isinstance(answer, Answer) and answer.confident is False
    assert llm.last_user is None
