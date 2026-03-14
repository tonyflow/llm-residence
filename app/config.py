from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    model_registry_path: str = os.getenv("MODEL_REGISTRY_PATH", "models.yaml")
    eviction_check_interval_seconds: int = int(
        os.getenv("EVICTION_CHECK_INTERVAL_SECONDS", "20")
    )


settings = Settings()
