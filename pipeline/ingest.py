"""Offline ingestion pipeline: docs folder -> chunks -> embeddings -> index file.

Run:  uv run python -m pipelines.ingest
"""

import argparse
from pathlib import Path

from docqa.config import get_settings
from docqa.core.models import Document
from docqa.observability.logging import configure_logging, get_logger
from docqa.retrieval.chunking import chunk_document
from docqa.retrieval.embeddings import HashingEmbedder
from docqa.stores.vector_store import InMemoryVectorStore

log = get_logger("pipelines.ingest")


def load_documents(docs_dir: Path) -> list[Document]:
    paths = sorted(p for p in docs_dir.rglob("*") if p.suffix in {".md", ".txt"})
    return [
        Document(id=p.stem, text=p.read_text(encoding="utf-8"), source=str(p.relative_to(docs_dir)))
        for p in paths
    ]


def run(docs_dir: Path, index_path: Path) -> int:
    s = get_settings()
    docs = load_documents(docs_dir)
    chunks = [c for d in docs for c in chunk_document(d, s.chunk_size, s.chunk_overlap)]
    store = InMemoryVectorStore(dim=s.embedding_dim)
    if chunks:
        store.add(chunks, HashingEmbedder(s.embedding_dim).embed([c.text for c in chunks]))
    store.save(index_path)
    log.info(
        "ingest.done",
        extra={"documents": len(docs), "chunks": len(chunks), "index": str(index_path)},
    )
    return len(chunks)


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description="Build the vector index from a docs folder.")
    parser.add_argument("--docs", type=Path, default=s.docs_dir)
    parser.add_argument("--index", type=Path, default=s.index_path)
    args = parser.parse_args()
    configure_logging()
    run(args.docs, args.index)


if __name__ == "__main__":
    main()
