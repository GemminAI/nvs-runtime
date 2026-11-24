"""EdgeProvider: wraps the Edge of nvs_edge; DEFERRED, never a local
reimplementation, when nvs_edge is missing or the kernel is unreachable."""

from __future__ import annotations

from typing import Any

import pytest

import nvs_runtime.providers.edge as edge_module
from nvs_runtime.providers.edge import EdgeProvider

#: nvs_edge (the NVS-Kernel edge package) is an optional dependency that a plain CI
#: runner may not have -- tests that need a *real* Edge (to simulate a transport failure, or to
#: exercise the OK path end to end) are skipped without it. Only
#: test_deferred_when_nvs_edge_is_not_installed is exempt: it monkeypatches
#: `_NVS_EDGE_AVAILABLE` itself, so it needs no real package either way.
needs_real_nvs_edge = pytest.mark.skipif(
    not edge_module._NVS_EDGE_AVAILABLE, reason="nvs_edge (nvs-kernel) is not installed"
)

POLICY = "a" * 64


def _synthetic_observations(salt: float = 0.0) -> list[dict[str, Any]]:
    """One plausible result per Core port (written out here, so these
    tests depend only on the public nvs_edge package)."""
    base = {
        "context_distance": 0.25 + salt, "gated_force": [0.1, 0.2],
        "top_eigenvalue": 1.5, "mean_step": 0.3,
        "residual": 0.01, "reduced_magnitude": 0.7, "reduced": 0.4, "accumulated_state": 0.2,
        "decay_variance": 0.05, "accumulated_shift": 0.6, "gated_magnitudes": [0.3, 0.1],
        "variance_of_reductions": 0.02, "projection": 0.9, "force_magnitude": 0.8, "v_norm": 1.0,
        "distance": 0.33, "aggregate": 0.5, "integrated_state": 0.45, "gated": [0.2, 0.0],
        "eigenvector": [0.6, 0.8],
        "nearest_distance": 0.12, "sample_magnitudes": [0.1, 0.3], "convolved": [0.4, 0.2],
        "hole_count": 3,
        "mean_vector": [0.3, 0.4], "binned": 2, "tracked": 0.15, "summary": 0.77,
        "smallest_eigenvalue": 0.001,
        "alignment": [0.9, 0.8], "top_magnitude": 0.55,
    }  # fmt: skip
    return [
        {"port_id": f"{pid}_Port", "status": "OK", "result": dict(base)}
        for pid in (f"P{n:02d}" for n in range(1, 39))
    ]


def _full_response(token_index: int) -> dict[str, Any]:
    # imported here, not at module level: nvs_edge is absent on a plain CI runner and the
    # module must still collect (the tests that do not need it still run).
    from nvs_edge.time_policy import kernel_time_policy_sha256
    from nvs_edge.wiring import wiring_policy_sha256

    return {
        "anchor_policy_sha256": POLICY,
        "wiring_policy_sha256": wiring_policy_sha256(),
        "time_policy_sha256": kernel_time_policy_sha256(),
        "anchor_counts": {"config_bootstrap": 1, "dynamic": 0, "manual": 0},
        "port_observations": _synthetic_observations(token_index * 1e-3),
        "cycle_provenance": [{"step": token_index}],
        "trajectory_hash": "x" * 64,
        "cycles": [{"session_id": "sess", "cycle": token_index + 1, "geometry": {}, "belief": {},
                     "field": {}, "trajectory": {}, "prediction": {}, "risk": {}, "control": {},
                     "hext": {}, "memory_size": 0, "crystallized": False}],
    }


async def test_deferred_when_nvs_edge_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(edge_module, "_NVS_EDGE_AVAILABLE", False)
    provider = EdgeProvider(base_url="http://unused", anchor_policy_sha256=POLICY)
    assert provider.available is False

    result = await provider.invoke(
        "edge.observe", {"session_id": "s1", "token_index": 0, "hidden_state": [0.0] * 4096}
    )
    assert result["status"] == "DEFERRED"
    assert "nvs_edge" in result["reason"]


@needs_real_nvs_edge
async def test_deferred_when_kernel_is_unreachable() -> None:
    def broken_transport(_path: str, _body: dict[str, Any]) -> dict[str, Any]:
        from nvs_edge.edge import EdgeTransportError

        raise EdgeTransportError("connection refused")

    provider = EdgeProvider(
        base_url="http://unused", anchor_policy_sha256=POLICY, transport=broken_transport
    )
    assert provider.available is True

    result = await provider.invoke(
        "edge.observe", {"session_id": "s1", "token_index": 0, "hidden_state": [0.0] * 4096}
    )
    assert result["status"] == "DEFERRED"
    assert "unreachable" in result["reason"]

    # once a session's kernel call has failed, further tokens in that session also
    # defer immediately rather than retrying (Edge does not retry internally either).
    result2 = await provider.invoke(
        "edge.observe", {"session_id": "s1", "token_index": 1, "hidden_state": [0.0] * 4096}
    )
    assert result2["status"] == "DEFERRED"
    assert "earlier in this session" in result2["reason"]


# ---- the CDM's session key ------------------------------------------------------------------


def _transport(_path: str, body: dict[str, Any]) -> dict[str, Any]:
    return _full_response(body["events"][0]["step"])


def _provider() -> EdgeProvider:
    return EdgeProvider(base_url="http://unused", anchor_policy_sha256=POLICY, transport=_transport)


async def _first_record(
    provider: EdgeProvider, session_id: str, token_index: int = 0
) -> dict[str, Any]:
    result = await provider.invoke(
        "edge.observe",
        {"session_id": session_id, "token_index": token_index, "hidden_state": [0.0] * 4096},
    )
    assert result["status"] == "OK"
    record: dict[str, Any] = result["record"]
    return record


@needs_real_nvs_edge
async def test_the_key_cannot_change_once_the_session_has_a_token() -> None:
    provider = _provider()
    await _first_record(provider, "s1")
    with pytest.raises(ValueError, match="already observed"):
        provider.set_session_key("s1", b"k" * 32)


@pytest.mark.parametrize("bad", [b"", "ab" * 32, None, 12])
def test_a_session_key_must_be_non_empty_bytes(bad: Any) -> None:
    provider = EdgeProvider(base_url="http://unused", anchor_policy_sha256=POLICY)
    with pytest.raises(ValueError, match="non-empty bytes"):
        provider.set_session_key("s1", bad)


async def test_the_key_does_not_appear_in_a_deferred_result() -> None:
    key = b"\x01\x02" * 16
    provider = EdgeProvider(base_url=None, anchor_policy_sha256=POLICY)
    provider.set_session_key("s1", key)
    result = await provider.invoke(
        "edge.observe", {"session_id": "s1", "token_index": 0, "hidden_state": [0.0] * 4096}
    )
    assert result["status"] == "DEFERRED"
    assert key.hex() not in repr(result) and repr(key) not in repr(result)


# ---- the CDM tier ----------------------------------------------------------------------------


async def _observe(provider: EdgeProvider, session_id: str) -> dict[str, Any]:
    result: dict[str, Any] = await provider.invoke(
        "edge.observe", {"session_id": session_id, "token_index": 0, "hidden_state": [0.0] * 4096}
    )
    assert result["status"] == "OK"
    return result


@needs_real_nvs_edge
async def test_basic_tier_returns_no_causal_log_and_pro_returns_it() -> None:
    provider = _provider()
    provider.set_session_tier("basic-session", "basic")
    provider.set_session_tier("pro-session", "pro")
    basic = await _observe(provider, "basic-session")
    pro = await _observe(provider, "pro-session")
    untiered = await _observe(provider, "plain-session")
    assert basic["causal_log"] == {} and basic["record"]["seq"] == 0
    assert len(pro["causal_log"]["ports"]) == 38 and len(untiered["causal_log"]["ports"]) == 38
    # the board side is the same in both tiers
    assert basic["record"]["causal_log_ref"] == pro["record"]["causal_log_ref"]


@needs_real_nvs_edge
async def test_the_tier_cannot_change_once_the_session_has_a_token() -> None:
    provider = _provider()
    await _first_record(provider, "s1")
    with pytest.raises(ValueError, match="already observed"):
        provider.set_session_tier("s1", "basic")


@pytest.mark.parametrize("bad", ["", "gold", "Basic", None])
def test_an_unknown_tier_is_refused(bad: Any) -> None:
    provider = EdgeProvider(base_url="http://unused", anchor_policy_sha256=POLICY)
    with pytest.raises(ValueError, match="tier"):
        provider.set_session_tier("s1", bad)


def test_an_in_process_provider_needs_no_base_url() -> None:
    in_process = EdgeProvider(in_process=True, anchor_policy_sha256=POLICY)
    assert in_process.available is edge_module._NVS_EDGE_AVAILABLE
    assert EdgeProvider(anchor_policy_sha256=POLICY).available is False


@needs_real_nvs_edge
async def test_the_tier_token_reaches_the_edge_and_an_unverifiable_one_is_basic() -> None:
    provider = _provider()
    provider.set_session_tier_token("token-session", "not.a-token")
    result = await _observe(provider, "token-session")
    # this Edge has no public key to verify the token with
    assert provider.session_tier("token-session") == "basic"
    assert result["causal_log"] == {}
    untiered = await _observe(provider, "plain-token-session")
    assert provider.session_tier("plain-token-session") is None
    assert len(untiered["causal_log"]["ports"]) == 38


@needs_real_nvs_edge
async def test_the_tier_token_cannot_change_once_the_session_has_a_token() -> None:
    provider = _provider()
    await _first_record(provider, "s1")
    with pytest.raises(ValueError, match="already observed"):
        provider.set_session_tier_token("s1", "x.y")
