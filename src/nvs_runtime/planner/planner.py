"""Graph-search planner — no hardcoded routes."""

from __future__ import annotations

from collections import deque

from nvs_runtime.core.errors import AuthorizationError
from nvs_runtime.core.models import (
    Capability,
    ExecutionPlan,
    Placement,
    PlanRequest,
    PlanStep,
)
from nvs_runtime.governance.engine import GovernanceEngine
from nvs_runtime.graph.capability_graph import CapabilityGraph
from nvs_runtime.security.cbac import CBACEngine


class Planner:
    """Searches the Capability Graph and admits plans via Governance."""

    def __init__(
        self,
        *,
        graph: CapabilityGraph,
        governance: GovernanceEngine,
        cbac: CBACEngine,
    ) -> None:
        self._graph = graph
        self._governance = governance
        self._cbac = cbac

    def plan(self, request: PlanRequest) -> ExecutionPlan:
        try:
            if request.token:
                self._cbac.require_scopes(request.token, {"scope:plan"})
                self._cbac.check_budget(request.token, request.max_cost_units)
        except AuthorizationError as exc:
            return ExecutionPlan(
                goal=request.goal,
                executable=False,
                error_code=exc.code,
                detail=exc.message,
            )

        targets = request.target_outputs or ["any"]
        paths = self._search(request.input_type, targets, allow_remote=request.allow_remote)
        if not paths:
            return ExecutionPlan(
                goal=request.goal,
                executable=False,
                error_code="NO_PATH",
                detail="no path in capability graph for goal",
            )

        ranked = self._rank(paths, request.optimization)
        chosen = ranked[0]
        steps = [
            PlanStep(
                capability_id=c.id,
                provider_id=c.provider_id,
                input_type=c.input_type,
                output_type=c.output_type,
                estimated_latency_ms=c.latency_ms,
                estimated_cost_units=c.cost_units,
            )
            for c in chosen
        ]
        latency = sum(s.estimated_latency_ms for s in steps)
        cost = sum(s.estimated_cost_units for s in steps)
        if latency > request.max_latency_ms or cost > request.max_cost_units:
            return ExecutionPlan(
                goal=request.goal,
                executable=False,
                steps=steps,
                estimated_latency_ms=latency,
                estimated_cost_units=cost,
                error_code="BUDGET_EXCEEDED",
                detail="chosen path exceeds request budget",
            )

        providers = {s.provider_id for s in steps}
        placement = Placement.LOCAL if len(providers) == 1 else Placement.HYBRID
        if any(s.provider_id.startswith("remote:") for s in steps):
            placement = Placement.REMOTE if placement != Placement.HYBRID else Placement.HYBRID

        plan = ExecutionPlan(
            goal=request.goal,
            executable=True,
            steps=steps,
            placement=placement,
            estimated_latency_ms=latency,
            estimated_cost_units=cost,
            metadata={"optimization": request.optimization},
        )
        self._governance.store_plan(plan)
        return plan

    def _search(
        self,
        input_type: str,
        targets: list[str],
        *,
        allow_remote: bool,
    ) -> list[list[Capability]]:
        starts = [
            c
            for c in self._graph.all()
            if _accepts(c.input_type, input_type)
            and (allow_remote or not c.provider_id.startswith("remote:"))
        ]
        found: list[list[Capability]] = []
        for start in starts:
            queue: deque[list[Capability]] = deque([[start]])
            seen: set[str] = {start.id}
            while queue:
                path = queue.popleft()
                last = path[-1]
                if _produces(last.output_type, targets):
                    found.append(path)
                    continue
                if len(path) >= 8:
                    continue
                for nxt in self._graph.neighbors(last.id):
                    if nxt.id in seen:
                        continue
                    if not allow_remote and nxt.provider_id.startswith("remote:"):
                        continue
                    seen.add(nxt.id)
                    queue.append([*path, nxt])
        return found

    def _rank(self, paths: list[list[Capability]], optimization: str) -> list[list[Capability]]:
        if optimization == "LOW_COST":
            return sorted(paths, key=lambda p: (sum(c.cost_units for c in p), len(p)))
        if optimization == "HIGH_THROUGHPUT":
            return sorted(paths, key=lambda p: (len(p), sum(c.latency_ms for c in p)))
        # LOW_LATENCY default
        return sorted(paths, key=lambda p: (sum(c.latency_ms for c in p), len(p)))


def _accepts(cap_input: str, request_input: str) -> bool:
    if cap_input in {"any", "*"} or request_input in {"any", "*"}:
        return True
    return cap_input == request_input


def _produces(output_type: str, targets: list[str]) -> bool:
    """Match outputs to targets.

    Wildcard outputs (``any`` / ``*``) only satisfy wildcard targets.
    This prevents catch-all providers (HTTP/gRPC) from short-circuiting
    every concrete plan goal.
    """
    if "any" in targets or "*" in targets:
        return output_type in targets or output_type in {"any", "*"}
    return output_type in targets
