from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from app.adapters.base import GenerationResult, RuntimeAdapter
from app.schemas import ChatMessage


class OllamaAdapter(RuntimeAdapter):
    def __init__(self, base_url: str, ollama_model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._ollama_model = ollama_model
        self._loaded = False
        self._client = httpx.AsyncClient(timeout=120.0)

    async def load(self) -> None:
        payload = {"model": self._ollama_model}
        response = await self._client.post(f"{self._base_url}/api/show", json=payload)

        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed loading model metadata from Ollama: {response.status_code} {response.text}"
            )

        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        if not self._loaded:
            raise RuntimeError("Ollama adapter not loaded")

        payload: dict[str, Any] = {
            "model": self._ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
        }

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        response = await self._client.post(f"{self._base_url}/api/chat", json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama generate failed: {response.status_code} {response.text}")

        data = response.json()
        content = data.get("message", {}).get("content", "")
        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        done_reason = data.get("done_reason", "stop")

        return GenerationResult(
            content=content,
            finish_reason=done_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        if not self._loaded:
            raise RuntimeError("Ollama adapter not loaded")

        payload: dict[str, Any] = {
            "model": self._ollama_model,
            "messages": [m.model_dump() for m in messages],
            "stream": True,
        }

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

        async with self._client.stream(
            "POST", f"{self._base_url}/api/chat", json=payload
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise RuntimeError(
                    f"Ollama stream failed: {response.status_code} {body.decode(errors='ignore')}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content")
                if piece:
                    yield piece

    async def close(self) -> None:
        await self._client.aclose()
