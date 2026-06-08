"""FastAPI dependency providers — all singletons stored in app.state."""

from __future__ import annotations

from fastapi import Request

from src.generation.llm_client import LLMClient
from src.ingestion.indexer import Indexer
from src.observability.tracer import Tracer
from src.retrieval.embedder import Embedder
from src.retrieval.reranker import Reranker
from src.retrieval.searcher import Searcher
from src.retrieval.sparse_embedder import SparseEmbedder


def get_settings(request: Request):
    return request.app.state.settings


def get_qdrant(request: Request):
    return request.app.state.qdrant


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_searcher(request: Request) -> Searcher:
    return request.app.state.searcher


def get_indexer(request: Request) -> Indexer:
    return request.app.state.indexer


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


def get_tracer(request: Request) -> Tracer:
    return request.app.state.tracer


def get_sparse_embedder(request: Request) -> SparseEmbedder:
    return request.app.state.sparse_embedder


def get_reranker(request: Request) -> Reranker:
    return request.app.state.reranker
