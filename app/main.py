from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.model_manager import ModelManager
from app.model_registry import ModelRegistry
from app.schemas import ChatCompletionRequest, ModelsResponse, ModelInfo
from app.services.chat_service import ChatService

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger(__name__)


_logger.info("Starting Mini Inference Server with settings: %s", settings)
registry = ModelRegistry(settings.model_registry_path)
manager = ModelManager(
    registry=registry,
    eviction_check_interval_seconds=settings.eviction_check_interval_seconds,
)
chat_service = ChatService(model_manager=manager)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await manager.start()
    try:
        yield
    finally:
        await manager.shutdown()


app = FastAPI(title="Mini Inference Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    models = [ModelInfo(id=m.id, runtime=m.runtime) for m in registry.list_models()]
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        if request.stream:
            return StreamingResponse(
                chat_service.stream_completion(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return await chat_service.create_completion(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
