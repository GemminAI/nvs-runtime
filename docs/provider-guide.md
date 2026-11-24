# Provider Guide

## Contract

Implement the `Provider` protocol:

```python
class Provider(Protocol):
    provider_id: str
    kind: str

    def discover(self) -> list[dict]: ...
    def capabilities(self) -> list[dict]: ...
    async def invoke(self, capability_id: str, payload: dict) -> dict: ...
    async def health(self) -> dict: ...
    async def shutdown(self) -> None: ...
```

Optional convenience base: `nvs_runtime.providers.base.BaseProvider`.

## Registering

```python
from nvs_runtime import build_runtime
from nvs_runtime.providers import LocalProvider

runtime = build_runtime(register_defaults=False)
runtime.register_provider(LocalProvider())
```

## Built-in providers

| Provider | Kind | Notes |
|----------|------|-------|
| `LocalProvider` | `local` | Deterministic in-process pipeline |
| `HttpProvider` | `http` | Offline stub or `InMemoryTransport` |
| `GrpcProvider` | `grpc` | Offline stub or transport |
| `McpProvider` | `mcp` | MCP tools as capabilities (optional SDK) |
| `KernelProvider` | `kernel` | Optional; `DEFERRED` when unavailable |
| `HekbProvider` | `hekb` | Knowledge via `ObjectStore` |

## Writing a custom provider

```python
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider

class MyProvider(BaseProvider):
    kind = "custom"

    def __init__(self):
        super().__init__("my")
        self.register_capability(
            Capability(
                id="my.work",
                name="Do work",
                provider_id=self.provider_id,
                input_type="text",
                output_type="text",
                tags=["custom"],
            )
        )

    async def invoke(self, capability_id: str, payload: dict) -> dict:
        return {"type": "text", "text": payload.get("text", "").upper()}
```

## Rules

1. Never import MCP / Kernel / HEKB internals into `core/`
2. Prefer transports and protocols over SDKs in the provider boundary
3. Provide an offline stub path so tests need no network or API keys
4. Advertise accurate `input_type` / `output_type` so the planner can connect edges

## MCP specifically

MCP is **only** `providers/mcp.py`.

- Core API does not know about MCP tools or stdio
- `create_mcp_bridge(runtime)` projects discover() into tool descriptors
- Optional dependency (nvs-runtime is not on PyPI yet, and neither is nvs-observation, so both come
  from GitHub): install nvs-observation first, then the `mcp` extra:

  ```bash
  pip install "git+https://github.com/GemminAI/nvs-observation.git"
  pip install "nvs-runtime[mcp] @ git+https://github.com/GemminAI/nvs-runtime.git"
  ```
