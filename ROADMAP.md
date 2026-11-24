# Roadmap

NVS Runtime evolves the **runtime substrate**, not any single provider.

Milestones below are directional. Dates are not commitments.

## v1.0.0 — Current

- Stable Runtime API surface
- Capability Graph + Planner
- Sessions, checkpoints, governance, metrics, observability, event bus
- Providers: Local, HTTP, gRPC, MCP, Kernel, HEKB (offline-capable)
- Offline deterministic test suite (≥90% coverage)

## v1.1 — Provider ecosystem

- **Provider SDK** — packaging helpers and contract tests for third-party providers
- **Provider Registry** — richer discovery, health, and lifecycle semantics
- **Streaming Providers** — incremental / streamed invoke results
- **Plugin Discovery** — load providers from entry points or config without core changes

## v1.2 — Distributed runtime

- **Distributed Runtime** — multi-node capability graphs and placement
- **Remote Runtime** — first-class remote invoke / observe over swappable transports

## v2.0 — Stable Runtime ABI

- Freeze a versioned **Runtime ABI** for long-lived application compatibility
- Stronger guarantees around plan / session / checkpoint durability
- Production-hardened Kernel / HEKB adapters without privileging them in core

## Non-goals

- Becoming an MCP-only product
- Embedding LLM or RAG frameworks in the core
- Reimplementing HEXT / Kernel / HEKB internals inside the runtime
