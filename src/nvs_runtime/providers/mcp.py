"""MCP provider — MCP is one provider among many, never the runtime center."""

from __future__ import annotations

from typing import Any

from nvs_runtime.core.errors import CapabilityNotFoundError, ProviderError
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider


class McpProvider(BaseProvider):
    """Exposes MCP tools as runtime capabilities.

    Optional dependency: install with ``pip install nvs-runtime[mcp]``.
    Without the MCP SDK, this provider operates in offline stub mode.
    """

    kind = "mcp"

    def __init__(
        self,
        provider_id: str = "mcp",
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(provider_id)
        self._tools = tools or [
            {
                "name": "mcp.echo",
                "description": "Echo tool (offline stub)",
                "input_type": "text",
                "output_type": "text",
            }
        ]
        for tool in self._tools:
            self.register_capability(
                Capability(
                    id=str(tool["name"]),
                    name=str(tool.get("description") or tool["name"]),
                    provider_id=self.provider_id,
                    input_type=str(tool.get("input_type", "any")),
                    output_type=str(tool.get("output_type", "any")),
                    tags=["mcp", "tool"],
                    cost_units=2,
                    latency_ms=3,
                    metadata={"mcp_tool": True},
                )
            )

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        ids = {c.id for c in self._capabilities}
        if capability_id not in ids:
            raise CapabilityNotFoundError(capability_id)
        if not self._healthy:
            raise ProviderError("mcp provider is shut down", code="PROVIDER_SHUTDOWN")
        # Offline deterministic stub — no MCP SDK / stdio required.
        return {
            "type": "mcp_tool_result",
            "tool": capability_id,
            "content": payload,
            "mode": "offline_stub",
            "provider": self.provider_id,
        }


def create_mcp_bridge(runtime: Any) -> dict[str, Any]:
    """Optional helper: project Runtime API into MCP tool descriptors.

    Returns descriptors only. Actual MCP server wiring lives outside core.
    """
    discovered = runtime.discover()
    tools = [
        {
            "name": cap.id,
            "description": cap.name,
            "inputSchema": {
                "type": "object",
                "properties": {"payload": {"type": "object"}},
            },
        }
        for cap in discovered.capabilities
    ]
    return {"tools": tools, "runtime": "nvs-runtime"}
