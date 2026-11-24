"""HTTP provider — offline by default via injectable transport."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import CapabilityNotFoundError, ProviderError
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider
from nvs_runtime.transport.memory import InMemoryTransport


class HttpProvider(BaseProvider):
    """Invokes remote HTTP-shaped endpoints through a Transport."""

    kind = "http"

    def __init__(
        self,
        provider_id: str = "http",
        *,
        transport: InMemoryTransport | None = None,
        endpoint: str = "http://local/invoke",
    ) -> None:
        super().__init__(provider_id)
        self._transport = transport
        self._endpoint = endpoint
        self.register_capability(
            Capability(
                id="http.invoke",
                name="HTTP Invoke",
                provider_id=self.provider_id,
                input_type="any",
                output_type="any",
                tags=["http"],
                cost_units=2,
                latency_ms=5,
            )
        )

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if capability_id != "http.invoke":
            raise CapabilityNotFoundError(capability_id)
        if self._transport is None:
            # Deterministic offline stub — no network.
            return {
                "type": "http_response",
                "status": 200,
                "body": payload,
                "mode": "offline_stub",
                "provider": self.provider_id,
            }
        try:
            return await self._transport.send(
                self._endpoint,
                {"capability_id": capability_id, "payload": payload},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(str(exc), code="HTTP_TRANSPORT_ERROR") from exc
