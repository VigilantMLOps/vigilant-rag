"""Cross-encoder reranker: re-scores top-K candidates and returns top-N."""

from __future__ import annotations

import asyncio
from functools import partial

from loguru import logger
from sentence_transformers import CrossEncoder

from src.retrieval.searcher import SearchResult


class Reranker:
    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, top_n: int = 5) -> None:
        logger.info(f"Loading cross-encoder: {self.MODEL}")
        self._model = CrossEncoder(self.MODEL)
        self._top_n = top_n
        logger.info("Cross-encoder ready")

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None, partial(self._model.predict, pairs)
        )

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        n = top_n or self._top_n
        for score, candidate in ranked[:n]:
            candidate.score = float(score)
        return [c for _, c in ranked[:n]]
