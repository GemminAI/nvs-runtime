# NVS Runtime v1.0.0

**The Runtime for Layer-Oriented AI Systems**

First public release of the provider-agnostic NVS Runtime.

- **Repository:** https://github.com/GemminAI/nvs-runtime  
- **License:** Apache-2.0  
- **Python:** ≥3.11  

## Overview

NVS Runtime is a clean-room execution substrate for layer-oriented AI systems.

Applications use one stable Runtime API.  
Providers (MCP, HTTP, gRPC, Local, Kernel, HEKB, and future adapters) plug in underneath.

This release is **not** an MCP runtime, LLM framework, or RAG framework.

## What's new

- Stable Runtime API: `discover` · `plan` · `invoke` · `observe` · `session` · `checkpoint` · `metrics`
- Capability Graph and graph-search Planner (no hardcoded pipelines)
- Sessions, checkpoints, governance quotas, event bus, metrics, observability
- Dependency-injected composition root (`build_runtime`)
- Offline, deterministic test suite (no API keys)
- MCP isolated under `providers/` — not central to the runtime

## Install

```bash
pip install -e ".[dev]"
pytest --cov=nvs_runtime
python examples/plan_and_invoke.py
```

## Runtime API

```python
runtime = build_runtime()
runtime.discover()
runtime.plan(...)
await runtime.invoke(...)
runtime.observe(...)
runtime.session(...)
runtime.checkpoint(...)
runtime.metrics()
await runtime.shutdown()
```

## Providers

| Provider | Notes |
|----------|-------|
| Local | Default deterministic pipeline capabilities |
| HTTP | Offline stub or injectable transport |
| gRPC | Offline stub or injectable transport |
| MCP | Optional; tool projection helper included |
| Kernel | Optional; returns `DEFERRED` when unavailable |
| HEKB | Knowledge provider via `ObjectStore` protocol |

## Architecture

```
Application → NVS Runtime → Providers
```

Core: Planner, Capability Graph, Session Manager, Governance, Metrics, Observability, Event Bus, Transport, Provider Registry.

HEXT is consumed only through semantic object / store interfaces.  
Kernel and HEKB integrate only through providers and adapters.

## Known limitations

- Distributed remote placement is minimal
- HTTP / gRPC / MCP default paths are offline stubs unless wired with transports or SDKs
- Kernel does not reimplement geometry locally when unavailable
- Metrics and observability are in-process only (no Prometheus exporter yet)

## Roadmap

See [ROADMAP.md](../ROADMAP.md) for v1.1 / v1.2 / v2.0 milestones.

## Citation

See [CITATION.cff](../CITATION.cff).
