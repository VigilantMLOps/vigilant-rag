"""GET /health — checks connectivity to Qdrant and Ollama."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from src.api.dependencies import get_qdrant, get_settings

router = APIRouter()


class DependencyStatus(BaseModel):
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    qdrant: DependencyStatus
    ollama: DependencyStatus
    collection: str
    vault_path: str


@router.get("/health", response_model=HealthResponse)
async def health(
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
    settings=Depends(get_settings),
) -> HealthResponse:
    qdrant_status = await _check_qdrant(qdrant, settings.collection_name)
    ollama_status = await _check_ollama(settings.ollama_url)
    overall = "ok" if qdrant_status.status == "ok" and ollama_status.status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        ollama=ollama_status,
        collection=settings.collection_name,
        vault_path=str(settings.vault_path),
    )


async def _check_qdrant(qdrant: AsyncQdrantClient, collection: str) -> DependencyStatus:
    try:
        info = await qdrant.get_collection(collection)
        count = info.points_count or 0
        return DependencyStatus(status="ok", detail=f"{count} points indexed")
    except Exception as e:
        return DependencyStatus(status="error", detail=str(e))


async def _check_ollama(ollama_url: str) -> DependencyStatus:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(ollama_url.rstrip("/") + "/api/version")
            resp.raise_for_status()
            version = resp.json().get("version", "unknown")
            return DependencyStatus(status="ok", detail=f"version {version}")
    except Exception as e:
        return DependencyStatus(status="error", detail=str(e))
