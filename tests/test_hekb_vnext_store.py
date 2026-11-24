"""HekbVNextObjectStore: maps put_object() onto hekb-vnext's real /v3/* wire format.

Offline: http_put/http_post are injected fakes (no network, no real hekb-vnext process).
The blob encoder import is real (a pure function, no network either) -- pointed at the
actual hekb-vnext checkout via HEKB_VNEXT_ROOT, skipped if that checkout is not present
in this environment.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from nvs_runtime.integrations.hekb_vnext import HekbExperienceRejected, HekbVNextObjectStore

HEKB_VNEXT_ROOT = Path(os.environ.get("HEKB_VNEXT_ROOT", Path.home() / "GemminAI" / "hekb-vnext"))
pytestmark = pytest.mark.skipif(
    not (HEKB_VNEXT_ROOT / "storage" / "trajectory_blob.py").is_file(),
    reason=f"hekb-vnext checkout not found at {HEKB_VNEXT_ROOT}",
)


class _FakeTransport:
    """Simulates hekb-vnext's own content-addressing closely enough to test THIS
    module's determinism: the same payload bytes always get the same object_id. Real
    dedup/signature verification is hekb-vnext's own responsibility (read, not
    re-tested here -- see store_service.py's create_experience)."""

    def __init__(self) -> None:
        self.put_calls: list[tuple[str, bytes]] = []
        self.post_calls: list[tuple[str, bytes]] = []
        self._blobs: set[str] = set()

    def put(self, url: str, *, headers: dict[str, str], data: bytes) -> tuple[int, str]:
        self.put_calls.append((url, data))
        sha = hashlib.sha256(data).hexdigest()
        self._blobs.add(sha)
        return 201, json.dumps({"sha256": sha, "bytes": len(data)})

    def post(self, url: str, *, headers: dict[str, str], data: bytes) -> tuple[int, str]:
        self.post_calls.append((url, data))
        if url.endswith("/v3/space-id"):
            space_id = hashlib.sha256(data).hexdigest()
            return 200, json.dumps({"space_id": space_id})
        if url.endswith("/v3/trajectories"):
            object_id = hashlib.sha256(data).hexdigest()
            return 201, json.dumps({"object_id": object_id})
        raise AssertionError(f"unexpected POST {url}")


def _store(transport: _FakeTransport) -> HekbVNextObjectStore:
    return HekbVNextObjectStore(
        base_url="http://unused",
        bearer_token="test-token",
        sign=lambda message: b"\x00" * 64,
        hekb_vnext_root=HEKB_VNEXT_ROOT,
        http_put=transport.put,
        http_post=transport.post,
    )


def _valid_data(**overrides: Any) -> dict[str, Any]:
    data = {
        "session_id": "p11-fake-0001",
        "port_ids": [f"P{n:02d}" for n in range(1, 33)],
        "quant_step_log2": -30,
        "board_format": "36cell/1",
        "rows": [[i] * 32 for i in range(3)],
        "causal_log_records": [
            {"token_index": i, "seq": i, "ports": {}, "cycle_raw": {}} for i in range(3)
        ],
        "space": {
            "model_ref": "不明", "layer": 32,
            "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
            "anchor_policy_sha256": "a" * 64,
        },
        "title": "p11-fake-0001",
        "tags": ["edge-hekb-slice"],
    }
    data.update(overrides)
    return data


def test_missing_causal_log_is_rejected_not_silently_stored() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    with pytest.raises(HekbExperienceRejected, match="causal_log_records is required"):
        store.put_object(_valid_data(causal_log_records=None))
    assert transport.put_calls == [] and transport.post_calls == []


def test_causal_log_length_mismatch_is_rejected() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    short_log = [{"token_index": 0, "seq": 0, "ports": {}, "cycle_raw": {}}]
    with pytest.raises(HekbExperienceRejected, match="has 1 entries, rows has 3"):
        store.put_object(_valid_data(causal_log_records=short_log))


def test_causal_log_record_missing_a_field_is_rejected() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    bad = [{"token_index": i, "seq": i, "ports": {}} for i in range(3)]  # cycle_raw missing
    with pytest.raises(HekbExperienceRejected, match="missing field"):
        store.put_object(_valid_data(causal_log_records=bad))


def test_successful_put_sends_two_blobs_then_one_trajectory_post() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    object_id = store.put_object(_valid_data(), labels={"category": "natural"})
    assert isinstance(object_id, str) and len(object_id) == 64
    assert len(transport.put_calls) == 2  # ports blob, causal_log blob
    assert any(url.endswith("/v3/trajectories") for url, _ in transport.post_calls)


def test_identical_data_is_idempotent_same_object_id_twice() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    data = _valid_data()
    first = store.put_object(data, labels={"category": "natural"})
    second = store.put_object(data, labels={"category": "natural"})
    assert first == second


def test_missing_required_top_level_field_is_rejected() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    data = _valid_data()
    del data["board_format"]
    with pytest.raises(HekbExperienceRejected, match="missing field"):
        store.put_object(data)


def test_blob_rejection_from_the_server_surfaces_as_hekb_experience_rejected() -> None:
    class RejectingTransport(_FakeTransport):
        def post(self, url: str, *, headers: dict[str, str], data: bytes) -> tuple[int, str]:
            if url.endswith("/v3/trajectories"):
                return 422, "payload.ext.hekbv3.causal_log.blob_sha256 not found"
            return super().post(url, headers=headers, data=data)

    transport = RejectingTransport()
    store = _store(transport)
    with pytest.raises(HekbExperienceRejected, match="422"):
        store.put_object(_valid_data())


def test_a_non_integer_row_is_rejected_before_the_blob_is_written() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    rows: list[list[object]] = [[i] * 32 for i in range(3)]
    rows[1][0] = 1.5
    with pytest.raises(HekbExperienceRejected, match="non-integer"):
        store.put_object(_valid_data(rows=rows))
    assert transport.put_calls == []


def test_excluded_columns_follow_ingest_boardrecords() -> None:
    transport = _FakeTransport()
    store = _store(transport)
    store.put_object(
        _valid_data(
            port_ids=["P01"],
            rows=[[1], [2], [3]],
            excluded_ports=["P20", "P15"],
        )
    )
    body = next(data for url, data in transport.post_calls if url.endswith("/v3/trajectories"))
    hekbv3 = json.loads(body)["payload"]["ext"]["hekbv3"]
    assert hekbv3["space"]["excluded_columns"] == "P15=valueless,P20=valueless"
    assert hekbv3["ports"]["excluded_columns"] == [
        {"port_id": "P20", "reason": "valueless"},
        {"port_id": "P15", "reason": "valueless"},
    ]
    assert hekbv3["ports"]["dim"] == 1
    assert hekbv3["ports"]["port_ids"] == ["P01"]
