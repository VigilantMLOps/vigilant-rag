"""FastAPI application factory with lifespan: starts watcher, warms up models."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from qdrant_client import AsyncQdrantClient

from config.settings import Settings, load_rag_config
from src.api.v1 import health, ingest, query
from src.generation.llm_client import LLMClient
from src.ingestion.indexer import Indexer
from src.ingestion.watcher import VaultWatcher
from src.observability.tracer import Tracer
from src.retrieval.embedder import Embedder
from src.retrieval.searcher import Searcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    rag_cfg = load_rag_config()

    # 1. Connect to Qdrant
    qdrant = AsyncQdrantClient(url=settings.qdrant_url)

    # 2. Build service graph
    embedder = Embedder(settings.ollama_url, settings.embed_model)
    searcher = Searcher(
        qdrant=qdrant,
        collection=settings.collection_name,
        top_k=rag_cfg["retrieval"]["top_k_dense"],
        score_threshold=rag_cfg["retrieval"]["score_threshold"],
    )
    indexer = Indexer(
        qdrant=qdrant,
        collection=settings.collection_name,
        embedder=embedder,
        chunk_size=rag_cfg["ingestion"]["chunk_size"],
        chunk_overlap=rag_cfg["ingestion"]["chunk_overlap"],
    )
    llm = LLMClient(settings.ollama_url, settings.llm_model)
    tracer = Tracer(settings.vigilant_api_url, enabled=settings.emit_traces)

    # 3. Ensure Qdrant collection exists
    await indexer.ensure_collection()

    # 4. Warm up models (non-blocking — we log errors but don't fail startup)
    try:
        await embedder.warmup()
        await llm.warmup()
    except Exception:
        logger.warning("Model warmup failed — Ollama may not be running yet")

    # 5. Store singletons in app state for dependency injection
    app.state.settings = settings
    app.state.qdrant = qdrant
    app.state.embedder = embedder
    app.state.searcher = searcher
    app.state.indexer = indexer
    app.state.llm = llm
    app.state.tracer = tracer

    # 6. Start file watcher and index consumer
    watcher = VaultWatcher(settings.vault_path)
    loop = asyncio.get_event_loop()
    watcher.start(loop)
    consumer_task = asyncio.create_task(_index_consumer(watcher, indexer))

    logger.info("vigilant-rag is ready")
    yield

    # 7. Shutdown: stop watcher, cancel consumer, close HTTP clients
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    watcher.stop()
    await embedder.aclose()
    await llm.aclose()
    await tracer.aclose()
    await qdrant.close()
    logger.info("vigilant-rag shutdown complete")


async def _index_consumer(watcher: VaultWatcher, indexer: Indexer) -> None:
    """Drain the watcher queue and process index events."""
    while True:
        event = await watcher.get_event()
        try:
            if event.type == "upsert":
                await indexer.index_file(event.path)
            elif event.type == "delete":
                await indexer.delete_file(event.path)
        except Exception:
            logger.exception(f"Failed to process index event: {event}")
        finally:
            watcher.task_done()


def create_app() -> FastAPI:
    app = FastAPI(
        title="vigilant-rag",
        description="Production RAG over Obsidian notes with VigilantMLOps observability",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")
    return app


app = create_app()
