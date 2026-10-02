"""End-to-end through HTTP: ingest a temp corpus, boot the app with the fake LLM, call /ask."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docqa.config import get_settings
from pipelines.ingest import run


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "returns.md").write_text("You can return unworn items within 30 days for a refund.")
    (docs / "shipping.md").write_text("Express shipping costs $15 and arrives in two days.")
    index = tmp_path / "index.json"

    monkeypatch.setenv("DOCQA_LLM_PROVIDER", "fake")
    monkeypatch.setenv("DOCQA_INDEX_PATH", str(index))
    get_settings.cache_clear()
    run(docs, index)

    from docqa.api.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["llm_provider"] == "fake"


def test_ask_returns_cited_answer(client: TestClient) -> None:
    r = client.post("/ask", json={"question": "How much is express shipping?"})
    assert r.status_code == 200
    body = r.json()
    assert "$15" in body["answer"]
    assert body["citations"] == ["shipping#0"]
    assert body["sources"][0]["source"] == "shipping.md"


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
