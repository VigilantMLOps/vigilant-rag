"""Fire-and-forget trace emitter → vigilant-api POST /api/v1/telemetry/rag-trace."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

import httpx
from loguru import logger


@dataclass
class RagTrace:
    query_text: str
    query_mode: str
    n_retrieved: int
    top_retrieval_score: float
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    model_name: str
    sources: list[str]              # list of file paths from retrieved chunks
    trace_id: str = ""
    prompt_version: str = "v1"

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())


class Tracer:
    def __init__(self, vigilant_api_url: str, enabled: bool = True) -> None:
        self._url = vigilant_api_url.rstrip("/") + "/api/v1/telemetry/rag-trace"
        self._enabled = enabled
        self._client = httpx.AsyncClient(timeout=2.0)

    async def emit(self, trace: RagTrace) -> None:
        """Emit a trace. Swallows all errors — must never block the query path."""
        if not self._enabled:
            return
        try:
            response = await self._client.post(self._url, json=asdict(trace))
            if response.status_code >= 400:
                logger.debug(f"Trace emit HTTP {response.status_code}: {response.text[:200]}")
        except Exception as exc:
            # vigilant-api may not be running locally — this is non-fatal
            logger.debug(f"Trace emit failed (non-fatal): {exc}")

    async def aclose(self) -> None:
        await self._client.aclose()
