"""In-process runtime metrics — no Prometheus required."""

from __future__ import annotations

from typing import Any


class MetricsCollector:
    """Counters and latency samples for offline observability."""

    def __init__(self) -> None:
        self.invokes = 0
        self.invoke_failures = 0
        self.plans = 0
        self.plan_failures = 0
        self.sessions_created = 0
        self.checkpoints_saved = 0
        self._latencies_ms: list[float] = []

    def record_invoke(self, *, ok: bool, latency_ms: float) -> None:
        self.invokes += 1
        if not ok:
            self.invoke_failures += 1
        self._latencies_ms.append(latency_ms)
        if len(self._latencies_ms) > 1_000:
            self._latencies_ms = self._latencies_ms[-1_000:]

    def record_plan(self, *, executable: bool) -> None:
        self.plans += 1
        if not executable:
            self.plan_failures += 1

    def record_session_created(self) -> None:
        self.sessions_created += 1

    def record_checkpoint(self) -> None:
        self.checkpoints_saved += 1

    def snapshot(self) -> dict[str, Any]:
        samples = self._latencies_ms
        avg = sum(samples) / len(samples) if samples else 0.0
        p95 = _percentile(samples, 0.95)
        return {
            "invokes": self.invokes,
            "invoke_failures": self.invoke_failures,
            "plans": self.plans,
            "plan_failures": self.plan_failures,
            "sessions_created": self.sessions_created,
            "checkpoints_saved": self.checkpoints_saved,
            "invoke_latency_ms": {
                "count": len(samples),
                "avg": round(avg, 3),
                "p95": round(p95, 3),
            },
        }


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]
