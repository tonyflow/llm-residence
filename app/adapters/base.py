from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from app.schemas import ChatMessage


@dataclass
class GenerationResult:
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class RuntimeAdapter(ABC):
    @abstractmethod
    async def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def unload(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    async def stream_generate(
        self,
        messages: list[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        raise NotImplementedError
