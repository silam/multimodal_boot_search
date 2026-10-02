"""FastAPI app. Thin layer: validate input, call the service, shape output."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request

from docqa.api.schemas import AskRequest, AskResponse, Source
from docqa.config import get_settings
from docqa.observability.logging import configure_logging
from docqa.services.container import build_qa_service
from docqa.services.qa_service import QAService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    app.state.qa = build_qa_service(get_settings())  # build once, reuse per request
    yield


app = FastAPI(title="docqa", version="0.1.0", lifespan=lifespan)


def get_qa(request: Request) -> QAService:
    qa: QAService = request.app.state.qa
    return qa


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "llm_provider": get_settings().llm_provider}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, qa: Annotated[QAService, Depends(get_qa)]) -> AskResponse:
    answer, hits = qa.ask(body.question)
    return AskResponse(
        answer=answer.answer,
        confident=answer.confident,
        citations=answer.citations,
        sources=[
            Source(chunk_id=h.chunk.id, source=h.chunk.source, score=round(h.score, 3))
            for h in hits
        ],
    )
