"""DI composition root — no module singletons."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nvs_runtime.config.settings import RuntimeConfig, load_config
from nvs_runtime.core.clock import system_clock_ns
from nvs_runtime.core.hext import MemoryObjectStore
from nvs_runtime.core.protocols import ObjectStore, Signer
from nvs_runtime.core.runtime import NVSRuntime
from nvs_runtime.events.bus import EventBus
from nvs_runtime.governance.engine import GovernanceEngine
from nvs_runtime.governance.quota import QuotaEngine
from nvs_runtime.graph.capability_graph import CapabilityGraph
from nvs_runtime.metrics.collector import MetricsCollector
from nvs_runtime.observability.store import ObservabilityStore
from nvs_runtime.planner.planner import Planner
from nvs_runtime.providers.hekb import HekbProvider
from nvs_runtime.providers.http import HttpProvider
from nvs_runtime.providers.kernel import KernelProvider
from nvs_runtime.providers.local import LocalProvider
from nvs_runtime.providers.mcp import McpProvider
from nvs_runtime.providers.registry import ProviderRegistry
from nvs_runtime.security.cbac import CBACEngine
from nvs_runtime.security.signer import HmacSigner
from nvs_runtime.session.checkpoint import CheckpointManager
from nvs_runtime.session.manager import SessionManager


def build_runtime(
    *,
    config: RuntimeConfig | dict[str, Any] | None = None,
    clock: Callable[[], int] | None = None,
    signer: Signer | None = None,
    store: ObjectStore | None = None,
    register_defaults: bool = True,
) -> NVSRuntime:
    """Compose a fully wired NVSRuntime.

    Deterministic by default: injectable clock, in-memory store, offline providers.
    """
    cfg = (
        config
        if isinstance(config, RuntimeConfig)
        else load_config(config if isinstance(config, dict) else None)
    )
    clock_fn = clock or system_clock_ns
    signer_impl = signer or HmacSigner()
    object_store = store or MemoryObjectStore()

    registry = ProviderRegistry()
    graph = CapabilityGraph()
    events = EventBus(clock=clock_fn)
    cbac = CBACEngine(signer=signer_impl, clock=clock_fn)
    quota = QuotaEngine(
        max_concurrent=cfg.max_concurrent,
        max_cost_per_tenant=cfg.max_cost_per_tenant,
    )
    governance = GovernanceEngine(store=object_store, quota=quota)
    planner = Planner(graph=graph, governance=governance, cbac=cbac)
    sessions = SessionManager(clock=clock_fn)
    checkpoints = CheckpointManager(store=object_store, clock=clock_fn)
    metrics = MetricsCollector()
    observability = ObservabilityStore()

    runtime = NVSRuntime(
        providers=registry,
        graph=graph,
        planner=planner,
        sessions=sessions,
        checkpoints=checkpoints,
        governance=governance,
        metrics_collector=metrics,
        observability=observability,
        events=events,
        cbac=cbac,
        clock=clock_fn,
    )

    if register_defaults:
        runtime.register_provider(LocalProvider())
        runtime.register_provider(HttpProvider())
        if cfg.enable_hekb:
            runtime.register_provider(HekbProvider(store=object_store))
        if cfg.enable_kernel:
            runtime.register_provider(KernelProvider(available=False))
        if cfg.enable_mcp:
            runtime.register_provider(McpProvider())

    return runtime
