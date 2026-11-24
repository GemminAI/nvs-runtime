"""Kernel provider — optional; runtime compiles and runs without Kernel."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import CapabilityNotFoundError
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider


class KernelProvider(BaseProvider):
    """Geometry / kernel integration through the provider boundary only.

    When no kernel endpoint is available, returns DEFERRED (no local reimplementation).
    """

    kind = "kernel"

    def __init__(
        self,
        provider_id: str = "kernel",
        *,
        available: bool = False,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(provider_id)
        self._available = available
        self._endpoint = endpoint
        self.register_capability(
            Capability(
                id="kernel.geometry",
                name="Kernel Geometry",
                provider_id=self.provider_id,
                input_type="geometry_request",
                output_type="geometry_result",
                tags=["kernel", "geometry"],
                cost_units=5,
                latency_ms=10,
            )
        )

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if capability_id != "kernel.geometry":
            raise CapabilityNotFoundError(capability_id)
        if not self._available:
            return {
                "type": "geometry_result",
                "status": "DEFERRED",
                "reason": "kernel unavailable",
                "request": payload,
                "provider": self.provider_id,
            }
        return {
            "type": "geometry_result",
            "status": "OK",
            "endpoint": self._endpoint,
            "request": payload,
            "provider": self.provider_id,
            "mode": "simulated",
        }

    async def health(self) -> dict[str, Any]:
        base = await super().health()
        base["available"] = self._available
        base["endpoint"] = self._endpoint
        return base
