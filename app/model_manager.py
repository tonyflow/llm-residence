from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.adapters.base import RuntimeAdapter
from app.adapters.echo import EchoAdapter
from app.adapters.ollama import OllamaAdapter
from app.model_registry import ModelConfig, ModelRegistry


@dataclass
class LoadedModel:
    adapter: RuntimeAdapter
    loaded_at: float
    last_used_at: float


class ModelManager:
    def __init__(self, registry: ModelRegistry, eviction_check_interval_seconds: int = 20) -> None:
        self._registry = registry
        self._lock = asyncio.Lock()
        self._loaded: dict[str, LoadedModel] = {}
        self._eviction_task: asyncio.Task[None] | None = None
        self._eviction_check_interval_seconds = eviction_check_interval_seconds

    async def start(self) -> None:
        self._eviction_task = asyncio.create_task(self._eviction_loop())

    async def shutdown(self) -> None:
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for loaded in self._loaded.values():
                await loaded.adapter.unload()
                close = getattr(loaded.adapter, "close", None)
                if callable(close):
                    await close()
            self._loaded.clear()

    async def get_adapter(self, model_id: str) -> RuntimeAdapter:
        model_cfg = self._registry.get(model_id)

        async with self._lock:
            existing = self._loaded.get(model_id)
            if existing:
                existing.last_used_at = time.time()
                return existing.adapter

            adapter = self._build_adapter(model_cfg)
            await adapter.load()
            now = time.time()
            self._loaded[model_id] = LoadedModel(
                adapter=adapter,
                loaded_at=now,
                last_used_at=now,
            )
            return adapter

    def _build_adapter(self, cfg: ModelConfig) -> RuntimeAdapter:
        if cfg.runtime == "echo":
            return EchoAdapter(prefix=str(cfg.params.get("prefix", "Echo")))
        if cfg.runtime == "ollama":
            base_url = str(cfg.params.get("base_url", "http://localhost:11434"))
            ollama_model = str(cfg.params["ollama_model"])
            return OllamaAdapter(base_url=base_url, ollama_model=ollama_model)
        raise ValueError(f"Unsupported runtime '{cfg.runtime}'")

    async def _eviction_loop(self) -> None:
        while True:
            await asyncio.sleep(self._eviction_check_interval_seconds)
            await self._evict_idle_models()

    async def _evict_idle_models(self) -> None:
        now = time.time()

        async with self._lock:
            to_evict: list[str] = []
            for model_id, loaded in self._loaded.items():
                cfg = self._registry.get(model_id)
                idle_seconds = now - loaded.last_used_at
                if idle_seconds > cfg.warm_ttl_seconds:
                    to_evict.append(model_id)

            for model_id in to_evict:
                loaded = self._loaded.pop(model_id)
                await loaded.adapter.unload()
                close = getattr(loaded.adapter, "close", None)
                if callable(close):
                    await close()
