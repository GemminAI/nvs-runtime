"""Local provider — in-process deterministic capabilities."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import CapabilityNotFoundError, ProviderError
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider


class LocalProvider(BaseProvider):
    """Default offline provider used by the reference runtime."""

    kind = "local"

    def __init__(self, provider_id: str = "local") -> None:
        super().__init__(provider_id)
        self.register_capability(
            Capability(
                id="local.echo",
                name="Echo",
                provider_id=self.provider_id,
                input_type="text",
                output_type="text",
                tags=["util", "echo"],
                cost_units=1,
                latency_ms=1,
            )
        )
        self.register_capability(
            Capability(
                id="local.transform",
                name="Transform",
                provider_id=self.provider_id,
                input_type="text",
                output_type="structured",
                tags=["util", "transform"],
                cost_units=2,
                latency_ms=2,
            )
        )
        self.register_capability(
            Capability(
                id="local.summarize",
                name="Summarize",
                provider_id=self.provider_id,
                input_type="structured",
                output_type="summary",
                tags=["util", "summarize"],
                cost_units=3,
                latency_ms=3,
            )
        )

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._healthy:
            raise ProviderError("local provider is shut down", code="PROVIDER_SHUTDOWN")
        ids = {c.id for c in self._capabilities}
        if capability_id not in ids:
            raise CapabilityNotFoundError(capability_id)

        if capability_id == "local.echo":
            text = str(payload.get("text", payload.get("input", "")))
            return {"type": "text", "text": text, "provider": self.provider_id}

        if capability_id == "local.transform":
            text = str(payload.get("text", payload.get("input", "")))
            return {
                "type": "structured",
                "tokens": text.split(),
                "length": len(text),
                "provider": self.provider_id,
            }

        if capability_id == "local.summarize":
            tokens = payload.get("tokens")
            if tokens is None:
                text = str(payload.get("text", ""))
                tokens = text.split()
            return {
                "type": "summary",
                "summary": f"{len(tokens)} tokens",
                "head": list(tokens)[:5],
                "provider": self.provider_id,
            }

        raise CapabilityNotFoundError(capability_id)
