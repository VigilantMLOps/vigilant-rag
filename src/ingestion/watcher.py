"""File system watcher: bridges watchdog events into an asyncio queue for the indexer."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from loguru import logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer


@dataclass
class IndexEvent:
    type: Literal["upsert", "delete"]
    path: Path


class _VaultEventHandler(FileSystemEventHandler):
    """Receives watchdog callbacks (sync thread) and enqueues async IndexEvents."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[IndexEvent]) -> None:
        super().__init__()
        self._loop = loop
        self._queue = queue
        # Debounce: track last event time per path to skip rapid-fire saves
        self._last_seen: dict[str, float] = {}
        self._debounce_seconds = 2.0

    def _enqueue(self, event_type: Literal["upsert", "delete"], src: str) -> None:
        now = time.monotonic()
        key = f"{event_type}:{src}"
        if now - self._last_seen.get(key, 0) < self._debounce_seconds:
            return
        self._last_seen[key] = now
        asyncio.run_coroutine_threadsafe(
            self._queue.put(IndexEvent(type=event_type, path=Path(src))),
            self._loop,
        )

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(".md"):
            self._enqueue("upsert", event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(".md"):
            self._enqueue("upsert", event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and str(event.src_path).endswith(".md"):
            self._enqueue("delete", event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            if str(event.src_path).endswith(".md"):
                self._enqueue("delete", event.src_path)
            if str(event.dest_path).endswith(".md"):
                self._enqueue("upsert", event.dest_path)


class VaultWatcher:
    """Wraps watchdog Observer and exposes an asyncio queue of IndexEvents."""

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path
        self._queue: asyncio.Queue[IndexEvent] = asyncio.Queue()
        self._observer: Observer | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        handler = _VaultEventHandler(loop, self._queue)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._vault_path), recursive=True)
        self._observer.start()
        logger.info(f"Watching vault: {self._vault_path}")

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join()
            logger.info("Vault watcher stopped")

    async def get_event(self) -> IndexEvent:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()
