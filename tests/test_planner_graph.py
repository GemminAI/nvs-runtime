"""Capability graph and planner — deterministic path search."""

from __future__ import annotations

from nvs_runtime.core.clock import ControllableClock
from nvs_runtime.core.hext import MemoryObjectStore
from nvs_runtime.core.models import Capability, PlanRequest
from nvs_runtime.governance.engine import GovernanceEngine
from nvs_runtime.graph.capability_graph import CapabilityGraph
from nvs_runtime.planner.planner import Planner
from nvs_runtime.security.cbac import CBACEngine
from nvs_runtime.security.signer import HmacSigner


def _planner() -> Planner:
    graph = CapabilityGraph()
    graph.replace(
        [
            Capability(
                id="a",
                name="A",
                provider_id="local",
                input_type="text",
                output_type="structured",
                cost_units=1,
                latency_ms=1,
            ),
            Capability(
                id="b",
                name="B",
                provider_id="local",
                input_type="structured",
                output_type="summary",
                cost_units=2,
                latency_ms=2,
            ),
            Capability(
                id="cheap",
                name="Cheap",
                provider_id="local",
                input_type="text",
                output_type="summary",
                cost_units=1,
                latency_ms=10,
            ),
        ]
    )
    store = MemoryObjectStore()
    gov = GovernanceEngine(store=store)
    clock = ControllableClock()
    cbac = CBACEngine(signer=HmacSigner(), clock=clock)
    return Planner(graph=graph, governance=gov, cbac=cbac)


def test_graph_edges() -> None:
    graph = CapabilityGraph()
    graph.replace(
        [
            Capability(
                id="a",
                name="A",
                provider_id="p",
                input_type="text",
                output_type="structured",
            ),
            Capability(
                id="b",
                name="B",
                provider_id="p",
                input_type="structured",
                output_type="summary",
            ),
        ]
    )
    built = graph.build()
    assert built["node_count"] == 2
    assert built["edge_count"] == 1
    assert built["edges"][0]["source"] == "a"
    assert built["edges"][0]["target"] == "b"


def test_planner_low_latency() -> None:
    planner = _planner()
    plan = planner.plan(
        PlanRequest(
            goal="g",
            input_type="text",
            target_outputs=["summary"],
            optimization="LOW_LATENCY",
        )
    )
    assert plan.executable
    # path a->b latency 3 vs cheap latency 10
    assert [s.capability_id for s in plan.steps] == ["a", "b"]


def test_planner_low_cost() -> None:
    planner = _planner()
    plan = planner.plan(
        PlanRequest(
            goal="g",
            input_type="text",
            target_outputs=["summary"],
            optimization="LOW_COST",
        )
    )
    assert plan.executable
    assert plan.steps[0].capability_id == "cheap"


def test_planner_budget_exceeded() -> None:
    planner = _planner()
    plan = planner.plan(
        PlanRequest(
            goal="g",
            input_type="text",
            target_outputs=["summary"],
            max_latency_ms=1,
            max_cost_units=100,
        )
    )
    assert not plan.executable
    assert plan.error_code == "BUDGET_EXCEEDED"


def test_planner_auth_failure() -> None:
    planner = _planner()
    plan = planner.plan(
        PlanRequest(
            goal="g",
            input_type="text",
            target_outputs=["summary"],
            token="not.a.valid.token",
        )
    )
    assert not plan.executable
    assert plan.error_code in {"TOKEN_MALFORMED", "TOKEN_INVALID"}
