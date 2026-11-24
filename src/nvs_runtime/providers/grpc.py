"""gRPC provider — offline stub with optional in-memory transport."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import CapabilityNotFoundError, ProviderError
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider
from nvs_runtime.transport.memory import InMemoryTransport


class GrpcProvider(BaseProvider):
    """gRPC-shaped provider. Core never embeds a gRPC client."""

    kind = "grpc"

    def __init__(
        self,
        provider_id: str = "grpc",
        *,
        transport: InMemoryTransport | None = None,
        endpoint: str = "grpc://local/Invoke",
    ) -> None:
        super().__init__(provider_id)
        self._transport = transport
        self._endpoint = endpoint
        self.register_capability(
            Capability(
                id="grpc.invoke",
                name="gRPC Invoke",
                provider_id=self.provider_id,
                input_type="any",
                output_type="any",
                tags=["grpc"],
                cost_units=2,
                latency_ms=4,
            )
        )

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if capability_id != "grpc.invoke":
            raise CapabilityNotFoundError(capability_id)
        if self._transport is None:
            return {
                "type": "grpc_response",
                "ok": True,
                "message": payload,
                "mode": "offline_stub",
                "provider": self.provider_id,
            }
        try:
            return await self._transport.send(
                self._endpoint,
                {"capability_id": capability_id, "payload": payload},
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(str(exc), code="GRPC_TRANSPORT_ERROR") from exc
