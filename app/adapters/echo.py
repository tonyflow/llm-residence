from __future__ import annotations

import asyncio
from typing import AsyncIterator

from app.adapters.base import GenerationResult, RuntimeAdapter
from app.schemas import ChatMessage


class EchoAdapter(RuntimeAdapter):
    def __init__(self, prefix: str = "Echo") -> None:
        self._loaded = False
        self._prefix = prefix

    async def load(self) -> None:
        await asyncio.sleep(0)
        self._loaded = True

    async def unload(self) -> None:
        await asyncio.sleep(0)
        self._loaded = False

    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        if not self._loaded:
            raise RuntimeError("Echo adapter not loaded")

        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        content = f"{self._prefix}: {last_user}".strip()
        completion_tokens = len(content.split())
        prompt_tokens = sum(len(m.content.split()) for m in messages)

        return GenerationResult(
            content=content,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def stream_generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        result = await self.generate(messages, temperature, max_tokens)
        for token in result.content.split():
            await asyncio.sleep(0.03)
            yield token + " "
