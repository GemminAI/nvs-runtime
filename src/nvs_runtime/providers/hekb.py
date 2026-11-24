"""HEKB knowledge provider — runtime talks only through ObjectStore adapters."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import CapabilityNotFoundError
from nvs_runtime.core.hext import MemoryObjectStore
from nvs_runtime.core.models import Capability
from nvs_runtime.core.protocols import ObjectStore
from nvs_runtime.providers.base import BaseProvider


class HekbProvider(BaseProvider):
    """Knowledge provider backed by an ObjectStore protocol implementation."""

    kind = "hekb"

    def __init__(
        self,
        provider_id: str = "hekb",
        *,
        store: ObjectStore | None = None,
    ) -> None:
        super().__init__(provider_id)
        self._store = store or MemoryObjectStore()
        self.register_capability(
            Capability(
                id="hekb.put",
                name="HEKB Put Object",
                provider_id=self.provider_id,
                input_type="object",
                output_type="object_ref",
                tags=["hekb", "store"],
                cost_units=1,
                latency_ms=1,
            )
        )
        self.register_capability(
            Capability(
                id="hekb.get",
                name="HEKB Get Object",
                provider_id=self.provider_id,
                input_type="object_ref",
                output_type="object",
                tags=["hekb", "store"],
                cost_units=1,
                latency_ms=1,
            )
        )

    @property
    def store(self) -> ObjectStore:
        return self._store

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if capability_id == "hekb.put":
            object_id = self._store.put_object(
                dict(payload.get("data") or payload),
                labels=dict(payload.get("labels") or {}),
            )
            return {
                "type": "object_ref",
                "object_id": object_id,
                "provider": self.provider_id,
            }
        if capability_id == "hekb.get":
            object_id = str(payload.get("object_id") or "")
            obj = self._store.get_object(object_id)
            return {
                "type": "object",
                "object_id": object_id,
                "found": obj is not None,
                "data": obj,
                "provider": self.provider_id,
            }
        raise CapabilityNotFoundError(capability_id)
