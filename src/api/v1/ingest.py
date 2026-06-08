"""POST /api/v1/ingest — manual full vault re-index trigger."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_indexer, get_settings
from src.ingestion.indexer import Indexer

router = APIRouter()


class IngestResponse(BaseModel):
    status: str
    total_chunks: int
    vault_path: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    indexer: Indexer = Depends(get_indexer),
    settings=Depends(get_settings),
) -> IngestResponse:
    try:
        total = await indexer.index_vault(settings.vault_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")

    return IngestResponse(
        status="ok",
        total_chunks=total,
        vault_path=str(settings.vault_path),
    )
