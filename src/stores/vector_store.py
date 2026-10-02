"""Vector store interface + a JSON-persisted in-memory implementation.

Replace with Qdrant / Cosmos DB / pgvector by implementing the same `VectorStore` Protocol.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from docqa.core.models import Chunk, RetrievedChunk
from docqa.retrieval.embeddings import Vector


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: Vector) -> None: ...
    def search(self, query: Vector, top_k: int) -> list[RetrievedChunk]: ...


class InMemoryVectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: Vector) -> None:
        if vectors.shape != (len(chunks), self.dim):
            raise ValueError(f"expected vectors of shape ({len(chunks)}, {self.dim})")
        self._chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: Vector, top_k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        scores = self._vectors @ query  # vectors are L2-normalised -> cosine similarity
        idx = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in idx]

    # --- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "chunks": [c.model_dump() for c in self._chunks],
            "vectors": self._vectors.tolist(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "InMemoryVectorStore":
        if not path.exists():
            raise FileNotFoundError(f"No index at {path}. Run the ingest pipeline first.")
        data = json.loads(path.read_text(encoding="utf-8"))
        store = cls(dim=data["dim"])
        store.add(
            [Chunk.model_validate(c) for c in data["chunks"]],
            np.asarray(data["vectors"], dtype=np.float32).reshape(-1, data["dim"]),
        )
        return store
