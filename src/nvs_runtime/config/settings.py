"""Runtime configuration — applied only at the composition root."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """Edge configuration for NVS Runtime."""

    name: str = "nvs-runtime"
    max_concurrent: int = 32
    max_cost_per_tenant: int = 10_000
    default_timeout_ms: int = 5_000
    enable_kernel: bool = False
    enable_hekb: bool = True
    enable_mcp: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_config(data: dict[str, Any] | None = None) -> RuntimeConfig:
    if not data:
        return RuntimeConfig()
    return RuntimeConfig.model_validate(data)
