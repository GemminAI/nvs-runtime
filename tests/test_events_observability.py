"""Event bus, metrics, observability, transport."""

from __future__ import annotations

import pytest

from nvs_runtime.core.clock import ControllableClock
from nvs_runtime.core.errors import TransportError
from nvs_runtime.events.bus import EventBus
from nvs_runtime.events.types import EventType
from nvs_runtime.metrics.collector import MetricsCollector
from nvs_runtime.observability.store import ObservabilityStore
from nvs_runtime.transport.memory import InMemoryTransport


def test_event_bus_publish_subscribe() -> None:
    clock = ControllableClock()
    bus = EventBus(clock=clock)
    seen = []
    bus.subscribe(lambda e: seen.append(e.type))
    bus.publish(EventType.PLAN_CREATED, {"plan_id": "p1"})
    assert seen == [EventType.PLAN_CREATED]
    hist = bus.history(limit=10, event_type=EventType.PLAN_CREATED)
    assert len(hist) == 1


def test_metrics_and_observability() -> None:
    m = MetricsCollector()
    m.record_invoke(ok=True, latency_ms=1.0)
    m.record_invoke(ok=False, latency_ms=2.0)
    m.record_plan(executable=True)
    m.record_plan(executable=False)
    m.record_session_created()
    m.record_checkpoint()
    snap = m.snapshot()
    assert snap["invokes"] == 2
    assert snap["invoke_failures"] == 1
    assert snap["plan_failures"] == 1

    obs = ObservabilityStore(max_items=3)
    for i in range(5):
        obs.record_timeline({"i": i})
    items = obs.query("timeline", limit=10)
    assert len(items) == 3
    assert items[0]["i"] == 2
    obs.record_topology({"n": 1})
    obs.record_plan({"p": 1})
    assert obs.query("topology")[0]["n"] == 1
    assert obs.query("statistics")[0]["timeline_count"] == 3


@pytest.mark.asyncio
async def test_memory_transport() -> None:
    t = InMemoryTransport()
    t.register("ep", lambda packet: {"ok": packet["x"]})
    out = await t.send("ep", {"x": 7})
    assert out["ok"] == 7
    with pytest.raises(TransportError):
        await t.send("missing", {})
