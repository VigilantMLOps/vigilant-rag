"""Parse Obsidian Markdown files into structured documents."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter


@dataclass
class ParsedDocument:
    file_path: str
    title: str
    body: str
    frontmatter: dict
    tags: list[str]
    obsidian_links: list[str]   # resolved basenames from [[wikilinks]]
    has_tasks: bool
    open_tasks: list[str]
    mtime: float
    content_hash: str           # MD5 of raw file content for dedup


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_TAG_RE = re.compile(r"(?<!\w)#([a-zA-Z][a-zA-Z0-9_/-]*)")
_OPEN_TASK_RE = re.compile(r"^- \[ \] (.+)$", re.MULTILINE)
_TASK_RE = re.compile(r"^- \[[ x]\] .+$", re.MULTILINE)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def parse(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    content_hash = hashlib.md5(raw.encode()).hexdigest()
    mtime = path.stat().st_mtime

    post = frontmatter.loads(raw)
    body: str = post.content
    meta: dict = dict(post.metadata)

    title = _extract_title(meta, body, path)
    tags = _extract_tags(meta, body)
    obsidian_links = _extract_wikilinks(body)
    open_tasks = _OPEN_TASK_RE.findall(body)
    has_tasks = bool(_TASK_RE.search(body))

    # Strip Obsidian syntax from body before chunking so embeddings are clean
    clean_body = _clean_body(body)

    return ParsedDocument(
        file_path=str(path),
        title=title,
        body=clean_body,
        frontmatter=meta,
        tags=tags,
        obsidian_links=obsidian_links,
        has_tasks=has_tasks,
        open_tasks=open_tasks,
        mtime=mtime,
        content_hash=content_hash,
    )


def _extract_title(meta: dict, body: str, path: Path) -> str:
    if "title" in meta and meta["title"]:
        return str(meta["title"])
    m = _H1_RE.search(body)
    if m:
        return m.group(1).strip()
    return path.stem


def _extract_tags(meta: dict, body: str) -> list[str]:
    tags: set[str] = set()
    # YAML frontmatter tags (list or space-separated string)
    fm_tags = meta.get("tags", [])
    if isinstance(fm_tags, str):
        fm_tags = fm_tags.split()
    for t in fm_tags:
        tags.add(str(t).lstrip("#"))
    # Inline #tags from body
    for m in _TAG_RE.finditer(body):
        tags.add(m.group(1))
    return sorted(tags)


def _extract_wikilinks(body: str) -> list[str]:
    seen: set[str] = set()
    links = []
    for m in _WIKILINK_RE.finditer(body):
        target = m.group(1).strip()
        if target not in seen:
            seen.add(target)
            links.append(target)
    return links


def _clean_body(body: str) -> str:
    """Remove Obsidian-specific syntax that would pollute embeddings."""
    # Remove wikilinks, keep display text if present: [[page|alias]] → alias, [[page]] → page
    body = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]*)?\|([^\]]+)\]\]", r"\2", body)
    body = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]*)?\]\]", r"\1", body)
    # Remove block references ^id
    body = re.sub(r"\^[a-zA-Z0-9-]+$", "", body, flags=re.MULTILINE)
    # Collapse excessive whitespace
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()
