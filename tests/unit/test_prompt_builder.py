"""Unit tests for src/generation/prompt_builder.py."""

import pytest

from src.generation.prompt_builder import build_prompt, _format_chunk, _build_context
from src.retrieval.searcher import SearchResult


def _make_result(title: str = "Test Note", text: str = "Some content.", score: float = 0.85) -> SearchResult:
    return SearchResult(
        chunk_id="abc123",
        file_path=f"/vault/{title}.md",
        title=title,
        text=text,
        score=score,
        tags=["test"],
        has_tasks=False,
        open_tasks=[],
        chunk_index=0,
    )


def test_build_prompt_returns_two_strings():
    system, user = build_prompt("What is RAG?", "factual", [_make_result()])
    assert isinstance(system, str)
    assert isinstance(user, str)
    assert len(system) > 0
    assert len(user) > 0


def test_factual_mode_prompt_mentions_precision():
    system, _ = build_prompt("query", "factual", [])
    assert "precise" in system.lower() or "directly" in system.lower()


def test_task_mode_prompt_mentions_tasks():
    system, _ = build_prompt("query", "task", [])
    assert "task" in system.lower()


def test_synthesis_mode_prompt_mentions_structured():
    system, _ = build_prompt("query", "synthesis", [])
    assert "synthes" in system.lower() or "structured" in system.lower()


def test_reasoning_mode_prompt_mentions_connections():
    system, _ = build_prompt("query", "reasoning", [])
    assert "connection" in system.lower() or "relation" in system.lower() or "analytical" in system.lower()


def test_user_message_contains_query():
    query = "What did I write about Qdrant?"
    _, user = build_prompt(query, "factual", [_make_result()])
    assert query in user


def test_user_message_contains_chunk_title():
    result = _make_result(title="Vector Databases")
    _, user = build_prompt("query", "factual", [result])
    assert "Vector Databases" in user


def test_empty_results_produces_no_found_message():
    _, user = build_prompt("query", "factual", [])
    assert "No relevant" in user or "not found" in user.lower()


def test_context_respects_token_limit():
    # Create many large results — context should be truncated
    big_text = "token " * 500
    results = [_make_result(title=f"Note {i}", text=big_text) for i in range(10)]
    _, user = build_prompt("query", "synthesis", results, max_context_tokens=512)
    # Should not contain all 10 notes
    assert user.count("Note ") < 10


def test_format_chunk_includes_score():
    result = _make_result(score=0.75)
    formatted = _format_chunk(1, result)
    assert "0.75" in formatted


def test_format_chunk_includes_tags():
    result = _make_result()
    result.tags = ["mlops", "rag"]
    formatted = _format_chunk(1, result)
    assert "mlops" in formatted
    assert "rag" in formatted
