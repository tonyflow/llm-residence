from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.adapters.base import RuntimeAdapter
from app.adapters.echo import EchoAdapter
from app.adapters.llama_cpp_local import LocalLlamaCppAdapter
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
                    maybe_coro = close()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
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
        if cfg.runtime == "llama_cpp_local":
            model_path = str(cfg.params["model_path"])
            n_ctx = int(cfg.params.get("n_ctx", 4096))
            n_gpu_layers = int(cfg.params.get("n_gpu_layers", 0))
            n_threads_raw = cfg.params.get("n_threads")
            n_threads = int(n_threads_raw) if n_threads_raw is not None else None
            return LocalLlamaCppAdapter(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
            )
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
                    maybe_coro = close()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
