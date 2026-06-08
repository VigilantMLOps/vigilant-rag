"""Async Ollama embedding client using nomic-embed-text."""

from __future__ import annotations

import httpx
from loguru import logger


class Embedder:
    def __init__(self, ollama_url: str, model: str = "nomic-embed-text") -> None:
        self._url = ollama_url.rstrip("/") + "/api/embeddings"
        self._model = model
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> list[float]:
        """Embed a single text. Returns a 768-dim float vector."""
        response = await self._client.post(
            self._url,
            json={"model": self._model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts sequentially (Ollama doesn't batch natively)."""
        vectors = []
        for text in texts:
            vectors.append(await self.embed(text))
        return vectors

    async def warmup(self) -> None:
        """Send a dummy request so Ollama loads the model into memory before first query."""
        logger.info(f"Warming up embedding model: {self._model}")
        await self.embed("warmup")
        logger.info("Embedding model ready")

    async def aclose(self) -> None:
        await self._client.aclose()
