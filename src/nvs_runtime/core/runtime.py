"""NVSRuntime — stable, provider-agnostic public API."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nvs_runtime.core.clock import elapsed_ms
from nvs_runtime.core.errors import (
    AuthorizationError,
    CapabilityNotFoundError,
    GovernanceError,
    ProviderError,
    ProviderNotFoundError,
    RuntimeError,
    ShutdownError,
)
from nvs_runtime.core.models import (
    CheckpointRequest,
    DiscoverQuery,
    DiscoverResult,
    ExecutionPlan,
    InvokeRequest,
    InvokeResult,
    ObserveQuery,
    ObserveResult,
    PlanRequest,
    SessionAction,
    SessionRequest,
)
from nvs_runtime.events.bus import EventBus
from nvs_runtime.events.types import EventType
from nvs_runtime.governance.engine import GovernanceEngine
from nvs_runtime.graph.capability_graph import CapabilityGraph
from nvs_runtime.metrics.collector import MetricsCollector
from nvs_runtime.observability.store import ObservabilityStore
from nvs_runtime.planner.planner import Planner
from nvs_runtime.providers.registry import ProviderRegistry
from nvs_runtime.security.cbac import CBACEngine
from nvs_runtime.session.checkpoint import CheckpointManager
from nvs_runtime.session.manager import SessionManager


@dataclass
class NVSRuntime:
    """Reference NVS Runtime — applications never talk to providers directly."""

    providers: ProviderRegistry
    graph: CapabilityGraph
    planner: Planner
    sessions: SessionManager
    checkpoints: CheckpointManager
    governance: GovernanceEngine
    metrics_collector: MetricsCollector
    observability: ObservabilityStore
    events: EventBus
    cbac: CBACEngine
    clock: Callable[[], int]
    _shut_down: bool = field(default=False, init=False, repr=False)

    def discover(
        self,
        query: DiscoverQuery | dict[str, Any] | None = None,
    ) -> DiscoverResult:
        self._ensure_alive()
        self._refresh_graph()
        q = (
            DiscoverQuery()
            if query is None
            else query
            if isinstance(query, DiscoverQuery)
            else DiscoverQuery.model_validate(query)
        )
        caps = self.graph.all()
        if q.provider_id:
            caps = [c for c in caps if c.provider_id == q.provider_id]
        if q.tag:
            caps = [c for c in caps if q.tag in c.tags]
        if q.input_type:
            caps = [c for c in caps if c.input_type in {q.input_type, "any", "*"}]
        if q.output_type:
            caps = [c for c in caps if c.output_type in {q.output_type, "any", "*"}]
        result = DiscoverResult(
            capabilities=caps,
            providers=self.providers.list_providers(),
        )
        self.events.publish(
            EventType.CAPABILITY_DISCOVERED,
            {"count": len(caps)},
        )
        return result

    def plan(self, request: PlanRequest | dict[str, Any]) -> ExecutionPlan:
        self._ensure_alive()
        self._refresh_graph()
        req = request if isinstance(request, PlanRequest) else PlanRequest.model_validate(request)
        plan = self.planner.plan(req)
        self.metrics_collector.record_plan(executable=plan.executable)
        self.observability.record_plan(plan.to_dict())
        self.events.publish(
            EventType.PLAN_CREATED,
            {"plan_id": plan.plan_id, "executable": plan.executable},
        )
        topo = self.graph.build()
        self.observability.record_topology(
            {"node_count": topo["node_count"], "edge_count": topo["edge_count"]}
        )
        return plan

    async def invoke(self, request: InvokeRequest | dict[str, Any]) -> InvokeResult:
        self._ensure_alive()
        self._refresh_graph()
        req = (
            request if isinstance(request, InvokeRequest) else InvokeRequest.model_validate(request)
        )
        t0 = self.clock()
        tenant_id = "anonymous"
        try:
            if req.token:
                payload = self.cbac.require_scopes(req.token, {"scope:invoke"})
                if payload:
                    tenant_id = payload.tenant_id

            if req.plan_id:
                return await self._invoke_plan(req, tenant_id=tenant_id, t0=t0)

            if not req.capability_id:
                raise CapabilityNotFoundError("<missing>")

            provider, cap = self.providers.find_capability(req.capability_id)
            if req.provider_id and req.provider_id != provider.provider_id:
                raise ProviderNotFoundError(req.provider_id)

            if req.token:
                self.cbac.check_budget(req.token, cap.cost_units)

            self.governance.quota.admit(tenant_id, cost=cap.cost_units)
            try:
                self.events.publish(
                    EventType.INVOKE_STARTED,
                    {"capability_id": cap.id, "provider_id": provider.provider_id},
                )
                if req.session_id:
                    self.sessions.start(req.session_id)
                output = await provider.invoke(cap.id, dict(req.payload))
                latency = elapsed_ms(t0, self.clock)
                result = InvokeResult(
                    ok=True,
                    capability_id=cap.id,
                    provider_id=provider.provider_id,
                    session_id=req.session_id,
                    output=output,
                    latency_ms=latency,
                )
                self.metrics_collector.record_invoke(ok=True, latency_ms=latency)
                self.events.publish(EventType.INVOKE_COMPLETED, result.to_dict())
                self.observability.record_timeline(
                    {"kind": "invoke", **result.to_dict()}
                )
                if req.session_id:
                    self.sessions.advance(req.session_id)
                return result
            finally:
                self.governance.release(tenant_id)
        except (
            AuthorizationError,
            CapabilityNotFoundError,
            ProviderError,
            ProviderNotFoundError,
            GovernanceError,
            RuntimeError,
            KeyError,
        ) as exc:
            latency = elapsed_ms(t0, self.clock)
            code = getattr(exc, "code", "INVOKE_ERROR")
            message = getattr(exc, "message", str(exc))
            result = InvokeResult(
                ok=False,
                capability_id=req.capability_id,
                provider_id=req.provider_id,
                session_id=req.session_id,
                plan_id=req.plan_id,
                error_code=code,
                detail=message,
                latency_ms=latency,
            )
            self.metrics_collector.record_invoke(ok=False, latency_ms=latency)
            self.events.publish(EventType.INVOKE_FAILED, result.to_dict())
            self.observability.record_timeline({"kind": "invoke_failed", **result.to_dict()})
            return result

    def observe(self, query: ObserveQuery | dict[str, Any] | None = None) -> ObserveResult:
        self._ensure_alive()
        q = (
            ObserveQuery()
            if query is None
            else query
            if isinstance(query, ObserveQuery)
            else ObserveQuery.model_validate(query)
        )
        if q.kind == "sessions":
            items = self.sessions.list_sessions()
            if q.session_id:
                items = [s for s in items if s.get("session_id") == q.session_id]
            return ObserveResult(kind="sessions", items=items)

        if q.kind == "topology":
            self._refresh_graph()
            topo = self.graph.build()
            return ObserveResult(kind="topology", items=[topo])

        if q.kind == "statistics":
            return ObserveResult(
                kind="statistics",
                statistics={
                    **self.metrics_collector.snapshot(),
                    "events": len(self.events.history(limit=10_000)),
                },
                items=self.observability.query("statistics", limit=q.limit),
            )

        items = self.observability.query("timeline", limit=q.limit)
        if q.session_id:
            items = [i for i in items if i.get("session_id") == q.session_id]
        if q.plan_id:
            items = [i for i in items if i.get("plan_id") == q.plan_id]
        # Merge event bus history for richer observe()
        events = self.events.history(limit=q.limit)
        return ObserveResult(
            kind=q.kind,
            items=items or events,
            statistics=self.metrics_collector.snapshot(),
        )

    def session(self, request: SessionRequest | dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure_alive()
        req = (
            SessionRequest()
            if request is None
            else request
            if isinstance(request, SessionRequest)
            else SessionRequest.model_validate(request)
        )
        result = self.sessions.handle(req)
        if req.action == SessionAction.CREATE:
            self.metrics_collector.record_session_created()
            self.events.publish(EventType.SESSION_CREATED, result)
        else:
            self.events.publish(EventType.SESSION_UPDATED, result)
        return result

    def checkpoint(
        self,
        request: CheckpointRequest | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._ensure_alive()
        req = (
            CheckpointRequest()
            if request is None
            else request
            if isinstance(request, CheckpointRequest)
            else CheckpointRequest.model_validate(request)
        )
        result = self.checkpoints.handle(req)
        if req.action.value == "save":
            self.metrics_collector.record_checkpoint()
            self.events.publish(EventType.CHECKPOINT_SAVED, result)
        elif req.action.value == "restore":
            self.events.publish(EventType.CHECKPOINT_RESTORED, result)
        return result

    def metrics(self) -> dict[str, Any]:
        self._ensure_alive()
        snap = self.metrics_collector.snapshot()
        snap["providers"] = self.providers.list_providers()
        snap["plans"] = len(self.governance.list_plans())
        return snap

    async def shutdown(self) -> None:
        if self._shut_down:
            return
        for provider in self.providers.providers():
            await provider.shutdown()
        self.events.publish(EventType.RUNTIME_SHUTDOWN, {})
        self._shut_down = True

    def register_provider(self, provider: Any) -> None:
        self._ensure_alive()
        self.providers.register(provider)
        self._refresh_graph()
        self.events.publish(
            EventType.PROVIDER_REGISTERED,
            {"provider_id": provider.provider_id, "kind": provider.kind},
        )
        self.events.publish(EventType.TOPOLOGY_CHANGED, self.graph.build())

    # --- internals ---

    async def _invoke_plan(
        self,
        req: InvokeRequest,
        *,
        tenant_id: str,
        t0: int,
    ) -> InvokeResult:
        assert req.plan_id is not None
        plan = self.governance.get_plan(req.plan_id)
        self.governance.admit(tenant_id=tenant_id, plan=plan)
        session_id = req.session_id
        # A caller-supplied session_id means the caller owns that session's lifecycle
        # (it created it, and will complete/cancel it itself -- e.g. one long-lived
        # session invoked by plan_id once per token). Only a session this call creates
        # for itself (session_id was None) is also this call's to complete -- the
        # single-shot pattern (README "Quick Start": invoke({"plan_id":...}) with no
        # session_id) is unchanged.
        session_owned_by_this_call = session_id is None
        try:
            if session_id is None:
                session = self.sessions.create(
                    plan_id=plan.plan_id,
                    total_steps=len(plan.steps),
                )
                session_id = session.session_id
                self.metrics_collector.record_session_created()
            self.sessions.start(session_id)

            output: dict[str, Any] = dict(req.payload)
            last_cap = None
            last_provider = None
            for idx, step in enumerate(plan.steps):
                if self.governance.is_cancelled(plan.plan_id):
                    raise GovernanceError("plan cancelled during invoke", code="PLAN_CANCELLED")
                provider, cap = self.providers.find_capability(step.capability_id)
                last_cap = cap.id
                last_provider = provider.provider_id
                self.events.publish(
                    EventType.INVOKE_STARTED,
                    {
                        "capability_id": cap.id,
                        "provider_id": provider.provider_id,
                        "plan_id": plan.plan_id,
                        "step": idx,
                    },
                )
                output = await provider.invoke(cap.id, output)
                self.sessions.advance(session_id, step=idx + 1)

            if session_owned_by_this_call:
                self.sessions.complete(session_id)
            latency = elapsed_ms(t0, self.clock)
            result = InvokeResult(
                ok=True,
                capability_id=last_cap,
                provider_id=last_provider,
                session_id=session_id,
                plan_id=plan.plan_id,
                output=output,
                latency_ms=latency,
            )
            self.metrics_collector.record_invoke(ok=True, latency_ms=latency)
            self.events.publish(EventType.INVOKE_COMPLETED, result.to_dict())
            self.observability.record_timeline({"kind": "invoke_plan", **result.to_dict()})
            return result
        finally:
            self.governance.release(tenant_id)

    def _refresh_graph(self) -> None:
        self.graph.replace(self.providers.all_capabilities())

    def _ensure_alive(self) -> None:
        if self._shut_down:
            raise ShutdownError()
