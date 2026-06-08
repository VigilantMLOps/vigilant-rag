"""Golden dataset schema and loader for offline RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalEntry:
    query: str
    mode: str
    expected_sources: list[str]   # path suffixes; matched via str.endswith()
    expected_answer_contains: list[str]
    difficulty: str = "medium"
    notes: str = ""


def load_dataset(path: Path) -> list[EvalEntry]:
    entries: list[EvalEntry] = []
    with open(path) as fh:
        for line in fh:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            data = json.loads(raw)
            entries.append(EvalEntry(**data))
    return entries
