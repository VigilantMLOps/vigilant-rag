"""Recursive character text splitter with tiktoken-based token counting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import tiktoken

from src.ingestion.parser import ParsedDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Split hierarchy: prefer paragraph breaks, fall back to smaller units
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str           # deterministic: md5(file_path + chunk_index)
    file_path: str
    title: str
    text: str
    chunk_index: int
    tags: list[str]
    frontmatter: dict
    has_tasks: bool
    open_tasks: list[str]
    obsidian_links: list[str]
    mtime: float
    file_hash: str          # the whole-file content_hash from ParsedDocument


def chunk_document(doc: ParsedDocument, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    if not doc.body.strip():
        return []

    texts = _split(doc.body, chunk_size, overlap)
    chunks = []
    for i, text in enumerate(texts):
        chunk_id = hashlib.md5(f"{doc.file_path}:{i}".encode()).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                file_path=doc.file_path,
                title=doc.title,
                text=text,
                chunk_index=i,
                tags=doc.tags,
                frontmatter=doc.frontmatter,
                has_tasks=doc.has_tasks,
                open_tasks=doc.open_tasks,
                obsidian_links=doc.obsidian_links,
                mtime=doc.mtime,
                file_hash=doc.content_hash,
            )
        )
    return chunks


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursively split on separators until all pieces fit within chunk_size."""
    return _merge(_recursive_split(text, chunk_size, _SEPARATORS), chunk_size, overlap)


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if _token_len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    next_seps = separators[1:]

    parts = text.split(sep) if sep else list(text)
    result = []
    for part in parts:
        if not part.strip():
            continue
        if _token_len(part) <= chunk_size:
            result.append(part)
        elif next_seps:
            result.extend(_recursive_split(part, chunk_size, next_seps))
        else:
            # Last resort: hard-cut by tokens
            result.extend(_hard_cut(part, chunk_size))
    return result


def _hard_cut(text: str, chunk_size: int) -> list[str]:
    tokens = _ENCODING.encode(text)
    return [
        _ENCODING.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks up to chunk_size, with overlap."""
    chunks: list[str] = []
    current_tokens: list[int] = []

    for piece in pieces:
        piece_tokens = _ENCODING.encode(piece)
        if current_tokens and len(current_tokens) + len(piece_tokens) > chunk_size:
            chunks.append(_ENCODING.decode(current_tokens))
            # Start next chunk with overlap from the end of current
            current_tokens = current_tokens[-overlap:] if overlap else []
        current_tokens.extend(piece_tokens)

    if current_tokens:
        chunks.append(_ENCODING.decode(current_tokens))

    return [c for c in chunks if c.strip()]
