"""Provider base helpers — composition over inheritance."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.models import Capability


class BaseProvider:
    """Optional convenience base. Prefer implementing the Provider protocol."""

    provider_id: str = "base"
    kind: str = "base"

    def __init__(self, provider_id: str | None = None) -> None:
        if provider_id:
            self.provider_id = provider_id
        self._capabilities: list[Capability] = []
        self._healthy = True

    def register_capability(self, capability: Capability) -> None:
        self._capabilities.append(capability)

    def discover(self) -> list[dict[str, Any]]:
        return [c.model_dump(mode="json") for c in self._capabilities]

    def capabilities(self) -> list[dict[str, Any]]:
        return self.discover()

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def health(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "kind": self.kind,
            "healthy": self._healthy,
            "capability_count": len(self._capabilities),
        }

    async def shutdown(self) -> None:
        self._healthy = False
