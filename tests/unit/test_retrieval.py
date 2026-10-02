from pathlib import Path

from docqa.retrieval.retriever import Retriever
from docqa.stores.vector_store import InMemoryVectorStore


def test_retriever_ranks_relevant_doc_first(retriever: Retriever) -> None:
    hits = retriever.retrieve("which pet needs walks?")
    assert hits[0].chunk.doc_id == "dogs"


def test_store_roundtrip(store: InMemoryVectorStore, tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    store.save(path)
    loaded = InMemoryVectorStore.load(path)
    assert len(loaded) == len(store)
