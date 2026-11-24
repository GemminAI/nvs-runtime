"""Capability Graph — DAG of capabilities, never hardcoded pipelines."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from nvs_runtime.core.models import Capability


class GraphNode(BaseModel):
    capability_id: str
    provider_id: str
    input_type: str
    output_type: str
    tags: list[str] = Field(default_factory=list)
    cost_units: int = 1
    latency_ms: int = 1


class GraphEdge(BaseModel):
    source: str
    target: str
    data_format: str


class CapabilityGraph:
    """Rebuildable capability DAG sourced from the provider registry."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}

    def replace(self, capabilities: list[Capability]) -> None:
        self._capabilities = {c.id: c for c in capabilities}

    def upsert(self, capability: Capability) -> None:
        self._capabilities[capability.id] = capability

    def remove_provider(self, provider_id: str) -> None:
        self._capabilities = {
            cid: cap for cid, cap in self._capabilities.items() if cap.provider_id != provider_id
        }

    def get(self, capability_id: str) -> Capability:
        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {capability_id}") from exc

    def all(self) -> list[Capability]:
        return list(self._capabilities.values())

    def build(self) -> dict[str, Any]:
        caps = self.all()
        nodes = [
            GraphNode(
                capability_id=c.id,
                provider_id=c.provider_id,
                input_type=c.input_type,
                output_type=c.output_type,
                tags=list(c.tags),
                cost_units=c.cost_units,
                latency_ms=c.latency_ms,
            )
            for c in caps
        ]
        edges: list[GraphEdge] = []
        for src in caps:
            for dst in caps:
                if src.id == dst.id:
                    continue
                if _types_compatible(src.output_type, dst.input_type):
                    edges.append(
                        GraphEdge(
                            source=src.id,
                            target=dst.id,
                            data_format=src.output_type,
                        )
                    )
        return {
            "nodes": [n.model_dump(mode="json") for n in nodes],
            "edges": [e.model_dump(mode="json") for e in edges],
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def neighbors(self, capability_id: str) -> list[Capability]:
        src = self.get(capability_id)
        return [
            dst
            for dst in self.all()
            if dst.id != src.id and _types_compatible(src.output_type, dst.input_type)
        ]


def _types_compatible(output_type: str, input_type: str) -> bool:
    if input_type in {"any", "*"} or output_type in {"any", "*"}:
        return True
    return output_type == input_type
