"""Unit tests for src/ingestion/parser.py."""

from pathlib import Path

import pytest

from src.ingestion.parser import parse, _extract_tags, _extract_wikilinks, _clean_body


@pytest.fixture
def tmp_md(tmp_path):
    """Factory: create a .md file in a temp dir and return its Path."""
    def _make(content: str, name: str = "note.md") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p
    return _make


def test_title_from_frontmatter(tmp_md):
    p = tmp_md("---\ntitle: My Note\n---\nBody text.")
    doc = parse(p)
    assert doc.title == "My Note"


def test_title_from_h1(tmp_md):
    p = tmp_md("# First Heading\n\nSome body.")
    doc = parse(p)
    assert doc.title == "First Heading"


def test_title_from_filename_fallback(tmp_md):
    p = tmp_md("No heading here.", name="project-alpha.md")
    doc = parse(p)
    assert doc.title == "project-alpha"


def test_frontmatter_tags_list(tmp_md):
    p = tmp_md("---\ntags:\n  - python\n  - mlops\n---\nBody.")
    doc = parse(p)
    assert "python" in doc.tags
    assert "mlops" in doc.tags


def test_inline_tags(tmp_md):
    p = tmp_md("Some note about #RAG and #LLMs in practice.")
    doc = parse(p)
    assert "RAG" in doc.tags
    assert "LLMs" in doc.tags


def test_wikilinks_extracted(tmp_md):
    p = tmp_md("See [[Project Alpha]] and [[RAG Systems|RAG]] for context.")
    doc = parse(p)
    assert "Project Alpha" in doc.obsidian_links
    assert "RAG Systems" in doc.obsidian_links


def test_open_tasks_detected(tmp_md):
    content = "## Tasks\n\n- [ ] Write tests\n- [x] Set up Qdrant\n- [ ] Add hybrid search"
    p = tmp_md(content)
    doc = parse(p)
    assert doc.has_tasks is True
    assert "Write tests" in doc.open_tasks
    assert "Add hybrid search" in doc.open_tasks
    assert len(doc.open_tasks) == 2


def test_no_tasks(tmp_md):
    p = tmp_md("Just prose, no tasks here.")
    doc = parse(p)
    assert doc.has_tasks is False
    assert doc.open_tasks == []


def test_content_hash_is_deterministic(tmp_md):
    content = "Same content every time."
    p = tmp_md(content)
    doc1 = parse(p)
    doc2 = parse(p)
    assert doc1.content_hash == doc2.content_hash


def test_clean_body_removes_wikilinks(tmp_md):
    # [[page|alias]] → alias, [[page]] → page
    p = tmp_md("See [[RAG Systems|RAG]] and [[Obsidian]].")
    doc = parse(p)
    assert "[[" not in doc.body
    assert "RAG" in doc.body
    assert "Obsidian" in doc.body


def test_empty_body_is_handled(tmp_md):
    p = tmp_md("---\ntitle: Empty\n---\n")
    doc = parse(p)
    assert doc.body == ""
