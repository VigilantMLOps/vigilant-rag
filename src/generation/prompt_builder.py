"""Build mode-aware prompts from a query and retrieved chunks."""

from __future__ import annotations

from typing import Literal

from src.retrieval.searcher import SearchResult

QueryMode = Literal["factual", "synthesis", "task", "reasoning"]

_SYSTEM_PROMPTS: dict[str, str] = {
    "factual": (
        "You are a precise assistant answering questions from personal notes. "
        "Answer directly and concisely. Cite the note title in parentheses after each claim. "
        "If the answer is not in the provided context, say 'Not found in my notes.'"
    ),
    "synthesis": (
        "You are a thoughtful assistant synthesizing information across multiple personal notes. "
        "Provide a structured, comprehensive answer. Group related ideas. "
        "Cite note titles in parentheses. Acknowledge if coverage is incomplete."
    ),
    "task": (
        "You are a task-focused assistant extracting action items and todos from personal notes. "
        "List open tasks clearly with their source note. "
        "Format tasks as a bulleted list. Include the note title for each task."
    ),
    "reasoning": (
        "You are an analytical assistant identifying connections across personal notes. "
        "Reason about relationships between topics. Be explicit about which notes support each connection. "
        "Highlight surprising or non-obvious links. Cite note titles throughout."
    ),
}

_CONTEXT_HEADER = "## Retrieved Notes\n\n"
_SEPARATOR = "\n---\n"


def build_prompt(
    query: str,
    mode: QueryMode,
    results: list[SearchResult],
    max_context_tokens: int = 3000,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) ready for Ollama chat format."""
    system = _SYSTEM_PROMPTS.get(mode, _SYSTEM_PROMPTS["factual"])
    context = _build_context(results, max_context_tokens)
    user = f"{context}\n\n## Question\n\n{query}"
    return system, user


def _build_context(results: list[SearchResult], max_tokens: int) -> str:
    if not results:
        return _CONTEXT_HEADER + "_No relevant notes found._"

    # Import inline to avoid circular imports at module level
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    sections: list[str] = []
    used_tokens = len(enc.encode(_CONTEXT_HEADER))

    for i, r in enumerate(results, start=1):
        excerpt = _format_chunk(i, r)
        chunk_tokens = len(enc.encode(excerpt))
        if used_tokens + chunk_tokens > max_tokens:
            break
        sections.append(excerpt)
        used_tokens += chunk_tokens

    return _CONTEXT_HEADER + _SEPARATOR.join(sections)


def _format_chunk(idx: int, result: SearchResult) -> str:
    tags_str = f" · tags: {', '.join(result.tags)}" if result.tags else ""
    score_str = f"{result.score:.2f}"
    return (
        f"### [{idx}] {result.title}{tags_str} (score: {score_str})\n\n"
        f"{result.text}"
    )
