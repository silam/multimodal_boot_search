"""Text chunking. Pure functions -> trivial to unit test."""

from docqa.core.models import Chunk, Document


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~`size`-character chunks with ~`overlap` characters of overlap.

    Works on word boundaries so words are never cut in half.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []

    avg_word = max(1, round(len(" ".join(words)) / len(words)))
    size_w = max(1, size // avg_word)
    step = max(1, size_w - overlap // avg_word)

    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size_w]))
        if start + size_w >= len(words):
            break
    return chunks


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    return [
        Chunk(id=f"{doc.id}#{i}", doc_id=doc.id, text=piece, source=doc.source)
        for i, piece in enumerate(chunk_text(doc.text, size, overlap))
    ]
