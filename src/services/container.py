"""Composition root: wires concrete implementations together from Settings."""

from docqa.config import Settings
from docqa.llm.factory import build_llm
from docqa.retrieval.embeddings import HashingEmbedder
from docqa.retrieval.retriever import Retriever
from docqa.services.qa_service import QAService
from docqa.stores.vector_store import InMemoryVectorStore


def build_qa_service(settings: Settings) -> QAService:
    store = InMemoryVectorStore.load(settings.index_path)
    retriever = Retriever(HashingEmbedder(settings.embedding_dim), store, top_k=settings.top_k)
    return QAService(retriever, build_llm(settings), prompt_name=settings.answer_prompt)
