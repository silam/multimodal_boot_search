import pytest

from docqa.retrieval.chunking import chunk_text


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("   ", size=100, overlap=10) == []


def test_long_text_overlaps_and_covers_all_words() -> None:
    words = [f"w{i}" for i in range(200)]
    chunks = chunk_text(" ".join(words), size=60, overlap=15)
    assert len(chunks) > 1
    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w199"
    # consecutive chunks share at least one word
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)
