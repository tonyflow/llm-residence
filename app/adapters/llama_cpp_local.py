from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

from llama_cpp import Llama

from app.adapters.base import GenerationResult, RuntimeAdapter
from app.schemas import ChatMessage


class LocalLlamaCppAdapter(RuntimeAdapter):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._llm: Llama | None = None

    async def load(self) -> None:
        if self._llm is not None:
            return

        self._llm = await asyncio.to_thread(
            Llama,
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            verbose=False,
        )

    async def unload(self) -> None:
        self._llm = None

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        if self._llm is None:
            raise RuntimeError("Local llama.cpp adapter not loaded")

        payload: dict[str, Any] = {
            "messages": [m.model_dump() for m in messages],
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = await asyncio.to_thread(self._llm.create_chat_completion, **payload)

        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError("No completion choices returned by local runtime")

        first_choice = choices[0]
        content = first_choice.get("message", {}).get("content", "")
        finish_reason = first_choice.get("finish_reason") or "stop"

        usage = response.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        return GenerationResult(
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        if self._llm is None:
            raise RuntimeError("Local llama.cpp adapter not loaded")

        payload: dict[str, Any] = {
            "messages": [m.model_dump() for m in messages],
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        queue: asyncio.Queue[str | None] = asyncio.Queue()

        loop = asyncio.get_running_loop()

        def _thread_worker() -> None:
            assert self._llm is not None
            for chunk in self._llm.create_chat_completion(**payload):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    loop.call_soon_threadsafe(queue.put_nowait, piece)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        thread = threading.Thread(target=_thread_worker, daemon=True)
        thread.start()
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
