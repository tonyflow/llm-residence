from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from app.model_manager import ModelManager
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    Usage,
)


class ChatService:
    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def create_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        adapter = await self._model_manager.get_adapter(request.model)
        result = await adapter.generate(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:16]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(content=result.content),
                    finish_reason=(
                        result.finish_reason
                        if result.finish_reason in {"stop", "length", "error"}
                        else "stop"
                    ),
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    async def stream_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[bytes]:
        adapter = await self._model_manager.get_adapter(request.model)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
        created = int(time.time())

        async for piece in adapter.stream_generate(
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")

        final_payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_payload)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
