"""Orchestration: retrieve -> build prompt -> call LLM -> validate. No framework imports."""

from docqa.core.models import Answer, RetrievedChunk
from docqa.llm.base import LLMClient
from docqa.observability.logging import get_logger, timed
from docqa.prompts.loader import render
from docqa.retrieval.retriever import Retriever

log = get_logger(__name__)

NO_CONTEXT = Answer(answer="I don't know based on the provided documents.", confident=False)


class QAService:
    def __init__(self, retriever: Retriever, llm: LLMClient, prompt_name: str) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompt_name = prompt_name

    def ask(self, question: str) -> tuple[Answer, list[RetrievedChunk]]:
        with timed(log, "qa.retrieve"):
            hits = self._retriever.retrieve(question)
        if not hits:
            return NO_CONTEXT, []

        system, user = render(self._prompt_name, chunks=hits, question=question, max_words=120)
        answer = self._llm.complete_structured(system, user, Answer)

        # Guardrail: drop citations the model invented.
        valid_ids = {h.chunk.id for h in hits}
        answer.citations = [c for c in answer.citations if c in valid_ids]
        log.info("qa.answered", extra={"prompt": self._prompt_name, "citations": answer.citations})
        return answer, hits
