"""Unit tests for src/ingestion/chunker.py."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion.chunker import chunk_document, _token_len
from src.ingestion.parser import ParsedDocument


def _make_doc(body: str, file_path: str = "/vault/note.md") -> ParsedDocument:
    import hashlib
    return ParsedDocument(
        file_path=file_path,
        title="Test Note",
        body=body,
        frontmatter={},
        tags=[],
        obsidian_links=[],
        has_tasks=False,
        open_tasks=[],
        mtime=0.0,
        content_hash=hashlib.md5(body.encode()).hexdigest(),
    )


def test_empty_body_produces_no_chunks():
    doc = _make_doc("")
    chunks = chunk_document(doc)
    assert chunks == []


def test_short_body_produces_one_chunk():
    doc = _make_doc("A short note with only a few words.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert "short note" in chunks[0].text


def test_long_body_is_split_into_multiple_chunks():
    # Generate body that exceeds 512 tokens
    paragraph = "This is a sentence about machine learning and observability. " * 20
    body = "\n\n".join([paragraph] * 5)
    doc = _make_doc(body)
    chunks = chunk_document(doc, chunk_size=128, overlap=16)
    assert len(chunks) > 1


def test_each_chunk_is_within_token_limit():
    paragraph = "Word " * 1000
    doc = _make_doc(paragraph)
    chunks = chunk_document(doc, chunk_size=256, overlap=32)
    for chunk in chunks:
        assert _token_len(chunk.text) <= 256 + 10  # small tolerance for merging


def test_chunk_metadata_inherited_from_document():
    doc = _make_doc("Some content.", file_path="/vault/projects/rag.md")
    doc.tags = ["mlops", "rag"]
    doc.has_tasks = True
    doc.open_tasks = ["Write tests"]
    chunks = chunk_document(doc)
    assert chunks[0].tags == ["mlops", "rag"]
    assert chunks[0].has_tasks is True
    assert chunks[0].open_tasks == ["Write tests"]
    assert chunks[0].file_path == "/vault/projects/rag.md"


def test_chunk_ids_are_unique():
    body = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    doc = _make_doc(body)
    chunks = chunk_document(doc, chunk_size=10, overlap=2)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_index_is_sequential():
    body = "Lots of text. " * 200
    doc = _make_doc(body)
    chunks = chunk_document(doc, chunk_size=64, overlap=8)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
