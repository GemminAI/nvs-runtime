"""Stable Runtime API — discover / plan / invoke / observe / session / checkpoint / metrics."""

from __future__ import annotations

import pytest

from nvs_runtime import NVSRuntime, build_runtime
from nvs_runtime.core.errors import ShutdownError


@pytest.mark.asyncio
async def test_public_api_surface(runtime: NVSRuntime, token: str) -> None:
    discovered = runtime.discover()
    assert discovered.capabilities
    assert any(p["kind"] == "local" for p in discovered.providers)

    plan = runtime.plan(
        {
            "goal": "summarize text",
            "input_type": "text",
            "target_outputs": ["summary"],
            "token": token,
        }
    )
    assert plan.executable
    assert plan.steps
    assert plan.steps[0].capability_id == "local.transform"
    assert plan.steps[-1].capability_id == "local.summarize"

    session = runtime.session({"action": "create", "plan_id": plan.plan_id})
    assert session["session_id"].startswith("sess_")

    result = await runtime.invoke(
        {
            "plan_id": plan.plan_id,
            "session_id": session["session_id"],
            "payload": {"text": "hello world from nvs"},
            "token": token,
        }
    )
    assert result.ok
    assert result.output["type"] == "summary"
    assert "tokens" in result.output["summary"]

    ckpt = runtime.checkpoint(
        {
            "action": "save",
            "session_id": session["session_id"],
            "state": {"step": 2, "note": "mid"},
            "label": "test",
        }
    )
    assert ckpt["checkpoint_id"].startswith("ckpt_")

    restored = runtime.checkpoint(
        {"action": "restore", "checkpoint_id": ckpt["checkpoint_id"]}
    )
    assert restored["restored"] is True
    assert restored["state"]["note"] == "mid"

    observed = runtime.observe({"kind": "statistics"})
    assert observed.kind == "statistics"
    assert observed.statistics["invokes"] >= 1

    metrics = runtime.metrics()
    assert metrics["invokes"] >= 1
    assert metrics["plans"] >= 1

    await runtime.shutdown()
    with pytest.raises(ShutdownError):
        runtime.discover()


@pytest.mark.asyncio
async def test_single_capability_invoke(runtime: NVSRuntime) -> None:
    result = await runtime.invoke(
        {"capability_id": "local.echo", "payload": {"text": "ping"}}
    )
    assert result.ok
    assert result.output["text"] == "ping"


@pytest.mark.asyncio
async def test_discover_filters(runtime: NVSRuntime) -> None:
    by_tag = runtime.discover({"tag": "echo"})
    assert all("echo" in c.tags for c in by_tag.capabilities)
    by_provider = runtime.discover({"provider_id": "local"})
    assert all(c.provider_id == "local" for c in by_provider.capabilities)


def test_plan_no_path(runtime: NVSRuntime) -> None:
    plan = runtime.plan(
        {
            "goal": "impossible",
            "input_type": "nonexistent_type_xyz",
            "target_outputs": ["also_nonexistent"],
        }
    )
    assert not plan.executable
    assert plan.error_code == "NO_PATH"


@pytest.mark.asyncio
async def test_invoke_unknown_capability(runtime: NVSRuntime) -> None:
    result = await runtime.invoke({"capability_id": "does.not.exist"})
    assert not result.ok
    assert result.error_code == "CAPABILITY_NOT_FOUND"


def test_build_runtime_minimal() -> None:
    rt = build_runtime(register_defaults=False)
    assert isinstance(rt, NVSRuntime)
    assert rt.discover().capabilities == []
