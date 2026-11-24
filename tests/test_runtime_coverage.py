"""Additional runtime paths for coverage and edge cases."""

from __future__ import annotations

import pytest

from nvs_runtime.core.clock import ControllableClock
from nvs_runtime.core.errors import (
    CapabilityNotFoundError,
    GovernanceError,
    ProviderError,
    ProviderNotFoundError,
)
from nvs_runtime.core.hext import MemoryObjectStore
from nvs_runtime.core.models import Capability
from nvs_runtime.governance.engine import GovernanceEngine
from nvs_runtime.providers.base import BaseProvider
from nvs_runtime.providers.grpc import GrpcProvider
from nvs_runtime.providers.http import HttpProvider
from nvs_runtime.providers.local import LocalProvider
from nvs_runtime.providers.registry import ProviderRegistry
from nvs_runtime.transport.memory import InMemoryTransport


@pytest.mark.asyncio
async def test_observe_kinds_and_session_filters(runtime, token) -> None:
    plan = runtime.plan(
        {
            "goal": "summarize",
            "input_type": "text",
            "target_outputs": ["summary"],
            "token": token,
        }
    )
    session = runtime.session({"action": "create", "plan_id": plan.plan_id})
    await runtime.invoke(
        {
            "plan_id": plan.plan_id,
            "session_id": session["session_id"],
            "payload": {"text": "one two three"},
            "token": token,
        }
    )

    topo = runtime.observe({"kind": "topology"})
    assert topo.items[0]["node_count"] >= 1

    sessions = runtime.observe(
        {"kind": "sessions", "session_id": session["session_id"]}
    )
    assert len(sessions.items) == 1

    timeline = runtime.observe({"kind": "timeline", "limit": 20})
    assert timeline.items

    filtered = runtime.observe(
        {"kind": "timeline", "session_id": session["session_id"], "plan_id": plan.plan_id}
    )
    assert isinstance(filtered.items, list)

    got = runtime.session({"action": "get", "session_id": session["session_id"]})
    assert got["session_id"] == session["session_id"]

    listed = runtime.checkpoint({"action": "list", "session_id": session["session_id"]})
    assert "checkpoints" in listed


@pytest.mark.asyncio
async def test_invoke_with_provider_mismatch_and_auth(runtime, token) -> None:
    bad = await runtime.invoke(
        {
            "capability_id": "local.echo",
            "provider_id": "not-local",
            "payload": {"text": "x"},
        }
    )
    assert not bad.ok

    limited = runtime.cbac.issue(
        tenant_id="t",
        scopes={"scope:invoke"},
        max_cost_units=0,
    )
    denied = await runtime.invoke(
        {
            "capability_id": "local.echo",
            "payload": {"text": "x"},
            "token": limited,
        }
    )
    assert not denied.ok

    ok = await runtime.invoke(
        {
            "capability_id": "local.echo",
            "provider_id": "local",
            "payload": {"text": "x"},
            "token": token,
        }
    )
    assert ok.ok


@pytest.mark.asyncio
async def test_plan_invoke_auto_session_and_cancel(runtime) -> None:
    plan = runtime.plan(
        {
            "goal": "summarize",
            "input_type": "text",
            "target_outputs": ["summary"],
        }
    )
    runtime.governance.cancel_plan(plan.plan_id)
    result = await runtime.invoke(
        {"plan_id": plan.plan_id, "payload": {"text": "hello"}}
    )
    assert not result.ok
    assert result.error_code == "PLAN_CANCELLED"


@pytest.mark.asyncio
async def test_register_provider_and_double_shutdown(runtime) -> None:
    class Extra(BaseProvider):
        kind = "custom"

        def __init__(self) -> None:
            super().__init__("extra")
            self.register_capability(
                Capability(
                    id="extra.work",
                    name="Extra",
                    provider_id="extra",
                    input_type="text",
                    output_type="text",
                )
            )

        async def invoke(self, capability_id: str, payload: dict) -> dict:
            return {"type": "text", "text": "extra"}

    runtime.register_provider(Extra())
    assert any(c.id == "extra.work" for c in runtime.discover().capabilities)
    await runtime.shutdown()
    await runtime.shutdown()  # idempotent


@pytest.mark.asyncio
async def test_provider_errors_and_registry() -> None:
    reg = ProviderRegistry()
    local = LocalProvider()
    reg.register(local)
    reg.unregister("local")
    with pytest.raises(ProviderNotFoundError):
        reg.get("local")

    p = LocalProvider()
    with pytest.raises(CapabilityNotFoundError):
        await p.invoke("missing", {})
    await p.shutdown()
    with pytest.raises(ProviderError):
        await p.invoke("local.echo", {"text": "x"})

    http = HttpProvider()
    with pytest.raises(CapabilityNotFoundError):
        await http.invoke("nope", {})
    transport = InMemoryTransport()

    def boom(_packet):
        raise RuntimeError("fail")

    transport.register("http://local/invoke", boom)
    http2 = HttpProvider(transport=transport)
    with pytest.raises(ProviderError):
        await http2.invoke("http.invoke", {})

    grpc = GrpcProvider()
    with pytest.raises(CapabilityNotFoundError):
        await grpc.invoke("nope", {})
    transport2 = InMemoryTransport()
    transport2.register("grpc://local/Invoke", boom)
    grpc2 = GrpcProvider(transport=transport2)
    with pytest.raises(ProviderError):
        await grpc2.invoke("grpc.invoke", {})


def test_governance_missing_plan_and_hext_list() -> None:
    store = MemoryObjectStore()
    gov = GovernanceEngine(store=store)
    with pytest.raises(GovernanceError):
        gov.get_plan("missing")
    assert gov.list_plans() == []
    assert store.list_ids() == []


def test_clock_set_and_models_observe_dict() -> None:
    clock = ControllableClock()
    clock.set(42)
    assert clock() == 42
    from nvs_runtime.core.models import ObserveResult

    assert ObserveResult(kind="timeline").to_dict()["kind"] == "timeline"


def test_graph_upsert_remove_get() -> None:
    from nvs_runtime.graph.capability_graph import CapabilityGraph

    g = CapabilityGraph()
    cap = Capability(
        id="c1",
        name="C",
        provider_id="p1",
        input_type="a",
        output_type="b",
    )
    g.upsert(cap)
    assert g.get("c1").id == "c1"
    g.remove_provider("p1")
    with pytest.raises(KeyError):
        g.get("c1")


def test_session_advance_and_cbac_helpers(runtime) -> None:
    s = runtime.session({"action": "create"})
    runtime.sessions.start(s["session_id"])
    advanced = runtime.sessions.advance(s["session_id"])
    assert advanced.current_step == 1
    runtime.sessions.create(plan_id="p", total_steps=2, detail={"k": 1})
    token = runtime.cbac.issue()
    assert runtime.cbac.audit_hash(token)
    assert runtime.cbac.require_scopes(None, {"scope:invoke"}) is None
    runtime.cbac.check_budget(None, 999)


@pytest.mark.asyncio
async def test_hekb_and_base_provider_defaults() -> None:
    from nvs_runtime.providers.base import BaseProvider
    from nvs_runtime.providers.hekb import HekbProvider
    from nvs_runtime.providers.kernel import KernelProvider

    base = BaseProvider()
    with pytest.raises(NotImplementedError):
        await base.invoke("x", {})

    hekb = HekbProvider()
    assert hekb.store is not None
    with pytest.raises(CapabilityNotFoundError):
        await hekb.invoke("hekb.missing", {})

    k = KernelProvider()
    with pytest.raises(CapabilityNotFoundError):
        await k.invoke("missing", {})


def test_planner_high_throughput_and_remote_filter() -> None:
    from nvs_runtime.core.models import PlanRequest
    from nvs_runtime.governance.engine import GovernanceEngine
    from nvs_runtime.graph.capability_graph import CapabilityGraph
    from nvs_runtime.planner.planner import Planner
    from nvs_runtime.security.cbac import CBACEngine
    from nvs_runtime.security.signer import HmacSigner

    graph = CapabilityGraph()
    graph.replace(
        [
            Capability(
                id="local.a",
                name="A",
                provider_id="local",
                input_type="text",
                output_type="summary",
                latency_ms=5,
                cost_units=1,
            ),
            Capability(
                id="remote.a",
                name="R",
                provider_id="remote:node",
                input_type="text",
                output_type="summary",
                latency_ms=1,
                cost_units=1,
            ),
        ]
    )
    planner = Planner(
        graph=graph,
        governance=GovernanceEngine(store=MemoryObjectStore()),
        cbac=CBACEngine(signer=HmacSigner(), clock=ControllableClock()),
    )
    plan = planner.plan(
        PlanRequest(
            goal="g",
            input_type="text",
            target_outputs=["summary"],
            optimization="HIGH_THROUGHPUT",
            allow_remote=False,
        )
    )
    assert plan.executable
    assert plan.steps[0].capability_id == "local.a"
