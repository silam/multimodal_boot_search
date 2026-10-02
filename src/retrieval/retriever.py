"""Retriever = embedder + vector store. Add reranking / hybrid search here later."""

from docqa.core.models import RetrievedChunk
from docqa.retrieval.embeddings import Embedder
from docqa.stores.vector_store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int, min_score: float = 0.05):
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        qvec = self._embedder.embed([query])[0]
        hits = self._store.search(qvec, self._top_k)
        return [h for h in hits if h.score >= self._min_score]
