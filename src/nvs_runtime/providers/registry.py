"""Provider registry — runtime-owned, provider-agnostic."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import CapabilityNotFoundError, ProviderNotFoundError
from nvs_runtime.core.models import Capability
from nvs_runtime.core.protocols import Provider


class ProviderRegistry:
    """Registers providers and aggregates their capabilities."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        self._providers[provider.provider_id] = provider

    def unregister(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def get(self, provider_id: str) -> Provider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ProviderNotFoundError(provider_id) from exc

    def list_providers(self) -> list[dict[str, Any]]:
        return [
            {
                "provider_id": p.provider_id,
                "kind": p.kind,
                "capability_count": len(p.capabilities()),
            }
            for p in self._providers.values()
        ]

    def all_capabilities(self) -> list[Capability]:
        caps: list[Capability] = []
        for provider in self._providers.values():
            for raw in provider.capabilities():
                caps.append(Capability.model_validate(raw))
        return caps

    def find_capability(self, capability_id: str) -> tuple[Provider, Capability]:
        for provider in self._providers.values():
            for raw in provider.capabilities():
                cap = Capability.model_validate(raw)
                if cap.id == capability_id:
                    return provider, cap
        raise CapabilityNotFoundError(capability_id)

    def providers(self) -> list[Provider]:
        return list(self._providers.values())
