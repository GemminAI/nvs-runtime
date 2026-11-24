"""Lightweight observability query store."""

from __future__ import annotations

from typing import Any


class ObservabilityStore:
    """Records timeline / topology / plan events for observe()."""

    def __init__(self, *, max_items: int = 2_000) -> None:
        self._max = max_items
        self._timeline: list[dict[str, Any]] = []
        self._topology: list[dict[str, Any]] = []
        self._plans: list[dict[str, Any]] = []

    def record_timeline(self, item: dict[str, Any]) -> None:
        self._append(self._timeline, item)

    def record_topology(self, item: dict[str, Any]) -> None:
        self._append(self._topology, item)

    def record_plan(self, item: dict[str, Any]) -> None:
        self._append(self._plans, item)

    def query(self, kind: str, *, limit: int = 50) -> list[dict[str, Any]]:
        if kind == "topology":
            return self._topology[-limit:]
        if kind == "plans":
            return self._plans[-limit:]
        if kind == "statistics":
            return [
                {
                    "timeline_count": len(self._timeline),
                    "topology_count": len(self._topology),
                    "plan_count": len(self._plans),
                }
            ]
        return self._timeline[-limit:]

    def _append(self, bucket: list[dict[str, Any]], item: dict[str, Any]) -> None:
        bucket.append(item)
        if len(bucket) > self._max:
            del bucket[: len(bucket) - self._max]
