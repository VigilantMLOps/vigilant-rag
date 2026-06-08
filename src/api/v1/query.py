"""POST /api/v1/query — main RAG query endpoint."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.api.dependencies import (
    get_embedder,
    get_llm,
    get_reranker,
    get_searcher,
    get_sparse_embedder,
    get_tracer,
)
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import QueryMode, build_prompt
from src.observability.tracer import RagTrace, Tracer
from src.retrieval.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.searcher import SearchResult, Searcher
from src.retrieval.sparse_embedder import SparseEmbedder

router = APIRouter()


class QueryFilters(BaseModel):
    tags: list[str] = Field(default_factory=list)
    has_tasks: bool | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: QueryMode = "factual"
    filters: QueryFilters = Field(default_factory=QueryFilters)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceDoc(BaseModel):
    title: str
    file_path: str
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    trace_id: str
    retrieval_latency_ms: int
    generation_latency_ms: int
    total_latency_ms: int


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    embedder: Embedder = Depends(get_embedder),
    sparse_embedder: SparseEmbedder = Depends(get_sparse_embedder),
    searcher: Searcher = Depends(get_searcher),
    reranker: Reranker = Depends(get_reranker),
    llm: LLMClient = Depends(get_llm),
    tracer: Tracer = Depends(get_tracer),
) -> QueryResponse:
    t_total_start = time.monotonic()

    # 1. Embed query — dense (Ollama) + sparse (BM25) in parallel
    t_ret_start = time.monotonic()
    try:
        query_vector, sparse_vector = await asyncio.gather(
            embedder.embed(request.query),
            sparse_embedder.embed(request.query),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {e}")

    # 2. Hybrid retrieval: RRF fusion of dense + sparse (fetch top_k_dense candidates)
    try:
        candidates: list[SearchResult] = await searcher.search(
            query_vector,
            sparse_vector,
            filter_tags=request.filters.tags or None,
            filter_has_tasks=request.filters.has_tasks,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector search unavailable: {e}")

    # 3. Cross-encoder rerank candidates → top_k final results
    try:
        results = await reranker.rerank(request.query, candidates, top_n=request.top_k)
    except Exception as e:
        logger.warning(f"Reranking failed, falling back to raw retrieval: {e}")
        results = candidates[: request.top_k]

    retrieval_latency_ms = int((time.monotonic() - t_ret_start) * 1000)

    # 4. Build prompt and generate
    system, user = build_prompt(request.query, request.mode, results)
    t_gen_start = time.monotonic()
    try:
        generation = await llm.generate(system, user)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    generation_latency_ms = int((time.monotonic() - t_gen_start) * 1000)
    total_latency_ms = int((time.monotonic() - t_total_start) * 1000)

    # 5. Build response
    sources = [
        SourceDoc(
            title=r.title,
            file_path=r.file_path,
            excerpt=r.text[:300],
            score=round(r.score, 4),
        )
        for r in results
    ]
    top_score = results[0].score if results else 0.0

    # 5. Fire-and-forget trace
    trace = RagTrace(
        query_text=request.query,
        query_mode=request.mode,
        n_retrieved=len(results),
        top_retrieval_score=top_score,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
        prompt_tokens=generation.prompt_tokens,
        completion_tokens=generation.completion_tokens,
        model_name=llm._model,
        sources=[r.file_path for r in results],
    )
    asyncio.create_task(tracer.emit(trace))

    return QueryResponse(
        answer=generation.text,
        sources=sources,
        trace_id=trace.trace_id,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
    )
