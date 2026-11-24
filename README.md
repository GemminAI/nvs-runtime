# NVS Runtime

**The Runtime for Layer-Oriented AI Systems**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/GemminAI/nvs-runtime/releases)
[![Tests](https://github.com/GemminAI/nvs-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/GemminAI/nvs-runtime/actions/workflows/ci.yml)

NVS Runtime is a **provider-agnostic runtime substrate** for layer-oriented AI systems.

It is not an MCP runtime.  
It is not an LLM framework.  
It is not a RAG framework.

Applications talk only to a stable Runtime API. Providers are interchangeable.

```
Application
    ↓
NVS Runtime
    ↓
Providers (MCP · HTTP · gRPC · Local · Kernel · HEKB · …)
```

## Installation

```bash
# nvs-observation is not on PyPI yet, so install it from GitHub first
pip install "git+https://github.com/GemminAI/nvs-observation.git"
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[mcp,http]"
```

Requires Python **≥3.12** (supported: 3.12, 3.13, 3.14). Licensed under **Apache-2.0**.

## Quick Start

```python
import asyncio
from nvs_runtime import build_runtime

async def main():
    runtime = build_runtime()

    runtime.discover()
    plan = runtime.plan({
        "goal": "summarize text",
        "input_type": "text",
        "target_outputs": ["summary"],
    })
    result = await runtime.invoke({
        "plan_id": plan.plan_id,
        "payload": {"text": "hello nvs runtime"},
    })
    print(result.output)
    print(runtime.observe({"kind": "statistics"}).statistics)
    print(runtime.metrics())
    await runtime.shutdown()

asyncio.run(main())
```

Or run the examples:

```bash
python examples/hello_runtime.py
python examples/plan_and_invoke.py
```

## Runtime API

| Method | Purpose |
|--------|---------|
| `runtime.discover()` | List providers and capabilities |
| `runtime.plan(...)` | Search the capability graph |
| `runtime.invoke(...)` | Execute a capability or plan |
| `runtime.observe(...)` | Timeline / topology / statistics |
| `runtime.session(...)` | Create / pause / resume / cancel |
| `runtime.checkpoint(...)` | Save / restore durable state |
| `runtime.metrics()` | In-process counters and latency |
| `runtime.shutdown()` | Stop providers cleanly |

## Architecture

```
NVSRuntime
├── Planner
├── Capability Graph
├── Session Manager
├── Governance
├── Metrics
├── Observability
├── Event Bus
├── Transport Layer
└── Provider Registry
        ↓
   Local · HTTP · gRPC · MCP · Kernel · HEKB
```

See [docs/architecture.md](docs/architecture.md).

## Provider Architecture

| Provider | Kind | Role |
|----------|------|------|
| `LocalProvider` | `local` | Deterministic in-process capabilities |
| `HttpProvider` | `http` | HTTP-shaped invoke (offline stub by default) |
| `GrpcProvider` | `grpc` | gRPC-shaped invoke (offline stub by default) |
| `McpProvider` | `mcp` | MCP tools as capabilities (optional SDK) |
| `KernelProvider` | `kernel` | Optional kernel geometry (`DEFERRED` if unavailable) |
| `HekbProvider` | `hekb` | Knowledge store via `ObjectStore` |

No provider is privileged. MCP is one adapter among many.

See [docs/provider-guide.md](docs/provider-guide.md).

## What is here, and what is not

This repository is the public runtime. It holds the Runtime API, the providers, the observation
capture path (capture records, the capture archive, the Gate and Formal Store clients, the `observe`
and `log` commands), and the readers of what was stored: the record schema, the read-back of a
stored run, the item-set entry, the comparison of a stored Risk and Control record with a replay,
and the `verify` command.

Measuring the K path of a capture and checking it again are done by a separate package, the
**Developer Tools**, which this repository does not contain. It plugs in through the
`nvs_runtime.k_measurer` entry point. Without it the runtime still captures, stores and reads, but a
K path is recorded as not measured (`edge_unavailable`), and `verify` reports how many positions with
a measured K it could not check (`k_unchecked`; the exit code does not change).

The code that drives experiments (replays, variants, end-to-end runners, analysis scripts) is not
part of the public nvs-runtime either; it lives with the Developer Tools. The tests here read stored
measured results from `tests/fixtures/measured_k`, which were made by the real kernel and are only
read, so they run without the kernel or the Developer Tools.

## Documentation

| Order | Guide |
|-------|--------|
| 1 | [Architecture](docs/architecture.md) |
| 2 | [Runtime Guide](docs/runtime-guide.md) |
| 3 | [Provider Guide](docs/provider-guide.md) |
| 4 | [Developer Guide](docs/developer-guide.md) |
| 5 | [Contribution Guide](docs/contribution-guide.md) |

Also: [Roadmap](ROADMAP.md) · [Release Notes v1.0.0](.github/RELEASE_NOTES_v1.0.0.md) · [Support](SUPPORT.md)

## Testing

```bash
pytest --cov=nvs_runtime --cov-report=term-missing
```

- Offline and deterministic
- No external APIs or API keys
- Coverage target ≥ 90%. The 90% gate applies where the NVS-Kernel edge and sensos are installed; the public CI reports coverage without gating on it.

## Roadmap

High-level milestones live in [ROADMAP.md](ROADMAP.md).

- **v1.1** — Provider SDK, registry hardening, streaming, plugin discovery  
- **v1.2** — Distributed / remote runtime  
- **v2.0** — Stable Runtime ABI  

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

## Philosophy

> We are not building an MCP runtime.  
> We are building the runtime that future AI systems execute on.  
> MCP is only one provider among many.  
> The runtime must remain stable even if every provider changes.
