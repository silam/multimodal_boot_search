import pytest

from docqa.core.models import Document
from docqa.retrieval.chunking import chunk_document
from docqa.retrieval.embeddings import HashingEmbedder
from docqa.retrieval.retriever import Retriever
from docqa.stores.vector_store import InMemoryVectorStore

DIM = 256


@pytest.fixture
def docs() -> list[Document]:
    return [
        Document(
            id="cats", text="Cats sleep most of the day and love warm windows.", source="cats.md"
        ),
        Document(
            id="dogs", text="Dogs need daily walks and enjoy playing fetch.", source="dogs.md"
        ),
    ]


@pytest.fixture
def store(docs: list[Document]) -> InMemoryVectorStore:
    emb = HashingEmbedder(DIM)
    chunks = [c for d in docs for c in chunk_document(d, size=200, overlap=20)]
    s = InMemoryVectorStore(DIM)
    s.add(chunks, emb.embed([c.text for c in chunks]))
    return s


@pytest.fixture
def retriever(store: InMemoryVectorStore) -> Retriever:
    return Retriever(HashingEmbedder(DIM), store, top_k=2)
