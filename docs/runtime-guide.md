# Runtime Guide

## Creating a runtime

```python
from nvs_runtime import build_runtime

runtime = build_runtime()
```

Custom composition:

```python
from nvs_runtime import build_runtime
from nvs_runtime.core.clock import ControllableClock

runtime = build_runtime(
    clock=ControllableClock(),
    config={"enable_kernel": False, "enable_mcp": False},
)
```

## discover()

```python
result = runtime.discover()
result = runtime.discover({"tag": "echo", "provider_id": "local"})
```

Returns capabilities and provider summaries.

## plan()

```python
plan = runtime.plan({
    "goal": "summarize text",
    "input_type": "text",
    "target_outputs": ["summary"],
    "optimization": "LOW_LATENCY",  # or LOW_COST / HIGH_THROUGHPUT
    "max_cost_units": 100,
    "max_latency_ms": 200,
})
```

Plans are stored by Governance and identified by `plan.plan_id`.

## invoke()

Single capability:

```python
result = await runtime.invoke({
    "capability_id": "local.echo",
    "payload": {"text": "hello"},
})
```

Full plan:

```python
result = await runtime.invoke({
    "plan_id": plan.plan_id,
    "payload": {"text": "hello nvs"},
})
```

## session()

```python
s = runtime.session({"action": "create", "plan_id": plan.plan_id})
runtime.session({"action": "pause", "session_id": s["session_id"]})
runtime.session({"action": "resume", "session_id": s["session_id"]})
runtime.session({"action": "cancel", "session_id": s["session_id"]})
```

## checkpoint()

```python
ckpt = runtime.checkpoint({
    "action": "save",
    "session_id": s["session_id"],
    "state": {"step": 1},
})
runtime.checkpoint({
    "action": "restore",
    "checkpoint_id": ckpt["checkpoint_id"],
})
```

## observe() / metrics()

```python
runtime.observe({"kind": "timeline"})
runtime.observe({"kind": "topology"})
runtime.observe({"kind": "statistics"})
runtime.metrics()
```

## shutdown()

```python
await runtime.shutdown()
```

After shutdown, API calls raise `ShutdownError`.

## Capability tokens (optional)

```python
token = runtime.cbac.issue(
    tenant_id="acme",
    scopes={"scope:invoke", "scope:plan", "scope:session"},
    max_cost_units=100,
)
runtime.plan({..., "token": token})
await runtime.invoke({..., "token": token})
```
