"""BM25 sparse embedder via fastembed — used for hybrid search."""

from __future__ import annotations

import asyncio
from functools import partial

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


class SparseEmbedder:
    """Wraps fastembed BM25 to produce Qdrant SparseVectors."""

    MODEL = "Qdrant/bm25"

    def __init__(self) -> None:
        # Downloads model on first instantiation (~1 MB from HuggingFace)
        self._model = SparseTextEmbedding(model_name=self.MODEL)

    def _embed_sync(self, text: str) -> SparseVector:
        embedding = next(self._model.embed([text]))
        return SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

    def _embed_batch_sync(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.embed(texts)
        ]

    async def embed(self, text: str) -> SparseVector:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[SparseVector]:
        loop = asyncio.get_event_loop()
        fn = partial(self._embed_batch_sync, texts)
        return await loop.run_in_executor(None, fn)
