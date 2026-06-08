"""Manage Qdrant upsert and delete operations for document chunks."""

from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.ingestion.chunker import Chunk, chunk_document
from src.ingestion.parser import parse

VECTOR_SIZE = 768  # nomic-embed-text output dimensions


class Indexer:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        collection: str,
        embedder,  # retrieval.embedder.Embedder — injected to avoid circular import
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self._qdrant = qdrant
        self._collection = collection
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ensure_collection(self) -> None:
        exists = await self._qdrant.collection_exists(self._collection)
        if not exists:
            await self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {self._collection}")

    async def index_file(self, path: Path) -> int:
        """Parse, chunk, embed, and upsert a single .md file. Returns chunk count."""
        doc = parse(path)
        chunks = chunk_document(doc, self._chunk_size, self._chunk_overlap)
        if not chunks:
            logger.debug(f"No chunks produced for {path} (empty body?)")
            return 0

        # Check if this file is unchanged since last index
        if await self._file_unchanged(path, doc.content_hash):
            logger.debug(f"Skipping unchanged file: {path.name}")
            return 0

        # Delete stale chunks for this file before upserting new ones
        await self.delete_file(path)

        vectors = await self._embedder.embed_batch([c.text for c in chunks])
        points = [_make_point(chunk, vec) for chunk, vec in zip(chunks, vectors)]
        await self._qdrant.upsert(collection_name=self._collection, points=points)
        logger.info(f"Indexed {len(points)} chunks for {path.name}")
        return len(points)

    async def delete_file(self, path: Path) -> None:
        """Remove all chunks belonging to a file from Qdrant."""
        await self._qdrant.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="file_path", match=MatchValue(value=str(path)))]
            ),
        )
        logger.debug(f"Deleted Qdrant points for: {path.name}")

    async def index_vault(self, vault_path: Path) -> int:
        """Full re-index of all .md files in the vault. Returns total chunk count."""
        total = 0
        md_files = list(vault_path.rglob("*.md"))
        logger.info(f"Full re-index: {len(md_files)} files in {vault_path}")
        for md_file in md_files:
            try:
                total += await self.index_file(md_file)
            except Exception:
                logger.exception(f"Failed to index {md_file}")
        logger.info(f"Full re-index complete: {total} total chunks")
        return total

    async def _file_unchanged(self, path: Path, current_hash: str) -> bool:
        """Return True if Qdrant already has a point with this file's current content_hash."""
        results, _ = await self._qdrant.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="file_path", match=MatchValue(value=str(path))),
                    FieldCondition(key="file_hash", match=MatchValue(value=current_hash)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0


def _make_point(chunk: Chunk, vector: list[float]) -> PointStruct:
    # Deterministic UUID from the chunk_id (MD5 hex string)
    point_id = str(uuid.UUID(chunk.chunk_id))
    return PointStruct(
        id=point_id,
        vector=vector,
        payload={
            "file_path": chunk.file_path,
            "title": chunk.title,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "tags": chunk.tags,
            "frontmatter": chunk.frontmatter,
            "has_tasks": chunk.has_tasks,
            "open_tasks": chunk.open_tasks,
            "obsidian_links": chunk.obsidian_links,
            "mtime": chunk.mtime,
            "file_hash": chunk.file_hash,
        },
    )
