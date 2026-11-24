"""Provider contract tests — all offline, no API keys."""

from __future__ import annotations

import pytest

from nvs_runtime.providers import (
    GrpcProvider,
    HekbProvider,
    HttpProvider,
    KernelProvider,
    LocalProvider,
    McpProvider,
)
from nvs_runtime.providers.mcp import create_mcp_bridge
from nvs_runtime.transport.memory import InMemoryTransport


@pytest.mark.asyncio
async def test_local_provider_pipeline() -> None:
    p = LocalProvider()
    echo = await p.invoke("local.echo", {"text": "a b c"})
    assert echo["text"] == "a b c"
    structured = await p.invoke("local.transform", {"text": "a b c"})
    assert structured["tokens"] == ["a", "b", "c"]
    summary = await p.invoke("local.summarize", structured)
    assert summary["type"] == "summary"
    health = await p.health()
    assert health["healthy"] is True
    await p.shutdown()
    health2 = await p.health()
    assert health2["healthy"] is False


@pytest.mark.asyncio
async def test_http_offline_stub() -> None:
    p = HttpProvider()
    out = await p.invoke("http.invoke", {"x": 1})
    assert out["mode"] == "offline_stub"
    assert out["status"] == 200


@pytest.mark.asyncio
async def test_http_with_memory_transport() -> None:
    transport = InMemoryTransport()

    async def handler(packet):
        return {"echo": packet["payload"]}

    transport.register("http://local/invoke", handler)
    p = HttpProvider(transport=transport)
    out = await p.invoke("http.invoke", {"ok": True})
    assert out["echo"]["ok"] is True


@pytest.mark.asyncio
async def test_grpc_offline_and_transport() -> None:
    p = GrpcProvider()
    stub = await p.invoke("grpc.invoke", {"n": 1})
    assert stub["mode"] == "offline_stub"

    transport = InMemoryTransport()
    transport.register("grpc://local/Invoke", lambda packet: {"rpc": packet})
    p2 = GrpcProvider(transport=transport)
    out = await p2.invoke("grpc.invoke", {"n": 2})
    assert out["rpc"]["payload"]["n"] == 2


@pytest.mark.asyncio
async def test_mcp_provider_offline() -> None:
    p = McpProvider()
    out = await p.invoke("mcp.echo", {"text": "hi"})
    assert out["mode"] == "offline_stub"
    assert out["tool"] == "mcp.echo"


@pytest.mark.asyncio
async def test_kernel_deferred_without_daemon() -> None:
    p = KernelProvider(available=False)
    out = await p.invoke("kernel.geometry", {"shape": "point"})
    assert out["status"] == "DEFERRED"
    health = await p.health()
    assert health["available"] is False

    p2 = KernelProvider(available=True, endpoint="http://kernel")
    out2 = await p2.invoke("kernel.geometry", {"shape": "point"})
    assert out2["status"] == "OK"


@pytest.mark.asyncio
async def test_hekb_put_get() -> None:
    p = HekbProvider()
    put = await p.invoke("hekb.put", {"data": {"type": "note", "text": "hello"}})
    assert put["object_id"]
    got = await p.invoke("hekb.get", {"object_id": put["object_id"]})
    assert got["found"] is True
    assert got["data"]["payload"]["text"] == "hello"


def test_mcp_bridge(runtime) -> None:
    bridge = create_mcp_bridge(runtime)
    assert "tools" in bridge
    assert bridge["runtime"] == "nvs-runtime"
