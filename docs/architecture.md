# Architecture

## Vision

NVS Runtime is the execution substrate for layer-oriented AI systems.  
Providers are replaceable. The Runtime API is not.

## Layering

```
Application
    ↓
NVS Runtime (stable API)
├── Planner              graph search → ExecutionPlan
├── Capability Graph     capability DAG (never hardcoded)
├── Session Manager      pause / resume / cancel
├── Governance           admit / cancel / quota
├── Metrics              counters + latency samples
├── Observability        timeline / topology queries
├── Event Bus            pub/sub + history
├── Transport Layer      swappable message delivery
└── Provider Registry    provider lifecycle
    ↓
Providers
├── LocalProvider
├── HttpProvider
├── GrpcProvider
├── McpProvider          ← MCP lives here only
├── KernelProvider       ← optional
└── HekbProvider         ← knowledge via ObjectStore
```

## Stable API contract

```python
runtime.discover()
runtime.plan(...)
runtime.invoke(...)
runtime.observe(...)
runtime.session(...)
runtime.checkpoint(...)
runtime.metrics()
runtime.shutdown()
```

Applications depend only on these methods.

## Provider contract

Every provider implements:

| Method | Role |
|--------|------|
| `discover()` | Advertise endpoints / tools |
| `capabilities()` | Structured capability descriptors |
| `invoke(capability_id, payload)` | Execute work |
| `health()` | Liveness |
| `shutdown()` | Cleanup |

No provider is privileged. MCP is not special-cased in the core.

## HEXT

HEXT is treated as a **Semantic Object Standard**.

Runtime depends only on:

- `SemanticObject` protocol (`object_id`, `object_type`, `to_dict()`)
- `ObjectStore` protocol (`put_object`, `get_object`)

Never HEXT SDK internals.

## HEKB

HEKB is a **Knowledge Provider**.

Runtime communicates only through `ObjectStore` adapters  
(default: in-memory `MemoryObjectStore` for offline operation).

## Kernel

Kernel is optional.

- Runtime compiles and operates without Kernel
- Integration occurs exclusively through `KernelProvider`
- Unavailable kernel → `DEFERRED` status (no local geometry reimplementation)

## Dependency injection

`build_runtime()` is the composition root.

- No module-level singletons
- Clock, signer, store, and providers are injectable
- Tests construct a fresh graph per case

## Determinism

| Boundary | Default offline double |
|----------|------------------------|
| Clock | `ControllableClock` / system clock |
| Object store | `MemoryObjectStore` |
| Transport | `InMemoryTransport` |
| HTTP / gRPC / MCP | offline stubs |
| Kernel | `DEFERRED` |

## Event flow (happy path)

```
discover → plan → session(create)
       → invoke(plan_id)
       → observe / metrics
       → checkpoint(save)
       → shutdown
```
