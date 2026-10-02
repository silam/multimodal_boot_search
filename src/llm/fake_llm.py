"""Deterministic fake LLM: lets you run, test and demo the app with no API key.

It is a tiny extractive QA model: it picks the sentence from the retrieved context with the
most word overlap with the question, and cites that chunk.
"""

import re
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_CHUNK_RE = re.compile(r'<chunk id="([^"]+)"[^>]*>\n(.*?)\n</chunk>', re.S)
_QUESTION_RE = re.compile(r"Question:\s*(.+)\s*$", re.S)
_WORD = re.compile(r"[a-z0-9$]+")
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "and",
    "is",
    "in",
    "for",
    "on",
    "what",
    "how",
    "do",
    "does",
    "i",
    "my",
    "you",
    "your",
    "which",
    "until",
    "much",
    "many",
    "long",
}


def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(text.lower()) if w not in _STOP}


class FakeLLM:
    def complete_structured(self, system: str, user: str, schema: type[T]) -> T:
        chunks = _CHUNK_RE.findall(user)
        q_match = _QUESTION_RE.search(user)
        if not chunks or not q_match:
            return schema.model_validate(
                {"answer": "I don't know based on the provided documents.", "confident": False}
            )
        q_words = _words(q_match.group(1))

        best = (0, "", "")  # (overlap, chunk_id, sentence)
        for chunk_id, text in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
                if sentence.startswith("#"):
                    continue
                overlap = len(q_words & _words(sentence))
                if overlap > best[0]:
                    best = (overlap, chunk_id, sentence.strip())

        if best[0] == 0:
            return schema.model_validate(
                {"answer": "I don't know based on the provided documents.", "confident": False}
            )
        return schema.model_validate(
            {"answer": f"[fake] {best[2]}", "citations": [best[1]], "confident": True}
        )
