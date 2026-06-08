"""Qdrant dense vector search. Phase 2 will add sparse (BM25) hybrid search."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint


@dataclass
class SearchResult:
    chunk_id: str
    file_path: str
    title: str
    text: str
    score: float
    tags: list[str]
    has_tasks: bool
    open_tasks: list[str]
    chunk_index: int


class Searcher:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        collection: str,
        top_k: int = 20,
        score_threshold: float = 0.3,
    ) -> None:
        self._qdrant = qdrant
        self._collection = collection
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def search(
        self,
        query_vector: list[float],
        *,
        top_k: int | None = None,
        filter_tags: list[str] | None = None,
        filter_has_tasks: bool | None = None,
    ) -> list[SearchResult]:
        """Dense ANN search with optional payload filters."""
        must_conditions = []
        if filter_tags:
            for tag in filter_tags:
                must_conditions.append(FieldCondition(key="tags", match=MatchValue(value=tag)))
        if filter_has_tasks is not None:
            must_conditions.append(
                FieldCondition(key="has_tasks", match=MatchValue(value=filter_has_tasks))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        response = await self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k or self._top_k,
            score_threshold=self._score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [_to_result(h) for h in response.points]


def _to_result(hit: ScoredPoint) -> SearchResult:
    p = hit.payload or {}
    return SearchResult(
        chunk_id=str(hit.id),
        file_path=p.get("file_path", ""),
        title=p.get("title", ""),
        text=p.get("text", ""),
        score=hit.score,
        tags=p.get("tags", []),
        has_tasks=p.get("has_tasks", False),
        open_tasks=p.get("open_tasks", []),
        chunk_index=p.get("chunk_index", 0),
    )
