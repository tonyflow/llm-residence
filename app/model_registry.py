from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    id: str
    runtime: str
    params: dict[str, Any]
    warm_ttl_seconds: int


class ModelRegistry:
    def __init__(self, registry_path: str) -> None:
        self._registry_path = Path(registry_path)
        self._models: dict[str, ModelConfig] = {}
        self.reload()

    def reload(self) -> None:
        if not self._registry_path.exists():
            raise FileNotFoundError(f"Model registry not found: {self._registry_path}")

        raw = yaml.safe_load(self._registry_path.read_text()) or {}
        models = raw.get("models", [])
        parsed: dict[str, ModelConfig] = {}

        for entry in models:
            model_id = entry["id"]
            runtime = entry["runtime"]
            ttl = int(entry.get("warm_ttl_seconds", 300))

            params = {k: v for k, v in entry.items() if k not in {"id", "runtime", "warm_ttl_seconds"}}
            parsed[model_id] = ModelConfig(
                id=model_id,
                runtime=runtime,
                params=params,
                warm_ttl_seconds=ttl,
            )

        self._models = parsed

    def get(self, model_id: str) -> ModelConfig:
        try:
            return self._models[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model '{model_id}'") from exc

    def list_models(self) -> list[ModelConfig]:
        return sorted(self._models.values(), key=lambda m: m.id)
