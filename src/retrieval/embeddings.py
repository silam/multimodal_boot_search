"""Embedding interface + an offline default.

`HashingEmbedder` is a dependency-free bag-of-words embedder so the project runs anywhere.
In production, add e.g. `OpenAIEmbedder` / `VoyageEmbedder` implementing the same Protocol
and select it in a factory, exactly like `llm/factory.py`.
"""

import hashlib
import re
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "to", "and", "is", "in", "for", "on", "what", "how", "does", "do"}


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> Vector: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim

    @staticmethod
    def _stem(token: str) -> str:
        # Crude suffix stripping so "walks"/"walk", "returned"/"return" match.
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    def embed(self, texts: list[str]) -> Vector:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                if tok not in _STOP:
                    out[row, self._bucket(self._stem(tok))] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1, norms)).astype(np.float32)
