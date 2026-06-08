"""Async Ollama generation client with streaming support."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx
from loguru import logger


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


@dataclass
class StreamChunk:
    content: str = ""
    done: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(self, ollama_url: str, model: str = "llama3.2:3b") -> None:
        self._chat_url = ollama_url.rstrip("/") + "/api/chat"
        self._model = model
        self._client = httpx.AsyncClient(timeout=120.0)

    async def generate(self, system: str, user: str) -> GenerationResult:
        """Non-streaming generation. Returns full text + token counts."""
        t0 = time.monotonic()
        response = await self._client.post(
            self._chat_url,
            json={
                "model": self._model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = int((time.monotonic() - t0) * 1000)

        text = data["message"]["content"]
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    async def stream(self, system: str, user: str):
        """Async generator yielding StreamChunk objects. Final chunk has done=True and token counts."""
        async with self._client.stream(
            "POST",
            self._chat_url,
            json={
                "model": self._model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if content := chunk.get("message", {}).get("content", ""):
                        yield StreamChunk(content=content)
                    if chunk.get("done"):
                        yield StreamChunk(
                            done=True,
                            prompt_tokens=chunk.get("prompt_eval_count", 0),
                            completion_tokens=chunk.get("eval_count", 0),
                        )
                        break
                except json.JSONDecodeError:
                    logger.warning(f"Unparseable Ollama stream line: {line!r}")

    async def warmup(self) -> None:
        """Pre-load the LLM into Ollama memory to avoid cold-start on first query."""
        logger.info(f"Warming up LLM: {self._model}")
        await self.generate("You are helpful.", "Say 'ready'.")
        logger.info("LLM ready")

    async def aclose(self) -> None:
        await self._client.aclose()
