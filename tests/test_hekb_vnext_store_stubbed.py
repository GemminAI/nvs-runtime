"""HekbVNextObjectStore without a hekb-vnext checkout.

tests/test_hekb_vnext_store.py needs the real ``storage.trajectory_blob`` of the sibling repo and is
skipped on a plain CI runner. Here that one function is replaced by a stub (the binary blob format
is hekb-vnext's, not what is tested): the request sequence, headers, signature, payload shape and
error mapping of this adapter are. Also the default urllib PUT/POST helpers (urlopen patched)."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import types
import urllib.error
from pathlib import Path
from typing import Any

import pytest

import nvs_runtime.integrations.hekb_vnext as hv
from nvs_runtime.integrations.hekb_vnext import (
    HekbExperienceRejected,
    HekbVNextObjectStore,
    HekbVNextUnavailable,
)


@pytest.fixture
def stub_blob_module(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def encode_trajectory_blob(rows: Any, **kw: Any) -> tuple[bytes, dict[str, Any]]:
        calls.append({"rows": rows, **kw})
        return b"BLOB" + json.dumps(rows).encode(), {}

    pkg = types.ModuleType("storage")
    mod = types.ModuleType("storage.trajectory_blob")
    mod.DTYPE_INT64 = "int64"  # type: ignore[attr-defined]
    mod.encode_trajectory_blob = encode_trajectory_blob  # type: ignore[attr-defined]
    pkg.trajectory_blob = mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "storage", pkg)
    monkeypatch.setitem(sys.modules, "storage.trajectory_blob", mod)
    return calls


class _Transport:
    def __init__(self, *, put_status: int = 201, post_status: int = 201, space_status: int = 200):
        self.put_calls: list[tuple[str, dict[str, str], bytes]] = []
        self.post_calls: list[tuple[str, dict[str, str], bytes]] = []
        self.put_status, self.post_status, self.space_status = put_status, post_status, space_status

    def put(self, url: str, *, headers: dict[str, str], data: bytes) -> tuple[int, str]:
        self.put_calls.append((url, headers, data))
        return self.put_status, "{}"

    def post(self, url: str, *, headers: dict[str, str], data: bytes) -> tuple[int, str]:
        self.post_calls.append((url, headers, data))
        if url.endswith("/v3/space-id"):
            return self.space_status, json.dumps({"space_id": "S" * 64})
        return self.post_status, json.dumps({"object_id": "O" * 64})


def _store(t: _Transport, root: Path = Path("/nonexistent")) -> HekbVNextObjectStore:
    return HekbVNextObjectStore(
        base_url="http://h/", bearer_token="tok", sign=lambda m: b"\x01" * 64,
        hekb_vnext_root=root, http_put=t.put, http_post=t.post,
    )


def _data(**over: Any) -> dict[str, Any]:
    d: dict[str, Any] = {
        "session_id": "s1", "port_ids": ["P01", "P02"], "quant_step_log2": -30,
        "board_format": "36cell/1", "rows": [[1, 2], [3, 4]],
        "causal_log_records": [
            {"token_index": i, "seq": i, "ports": {}, "cycle_raw": {}} for i in range(2)
        ],
        "space": {"model_ref": "m", "layer": 1},
    }
    d.update(over)
    return d


def test_put_sequence_payload_signature_and_headers(stub_blob_module: list[dict[str, Any]]) -> None:
    t = _Transport()
    oid = _store(t).put_object(_data(tags=["x"]), labels={"category": "natural"})
    assert oid == "O" * 64
    assert stub_blob_module[0]["created_at_ms"] == 0 and stub_blob_module[0]["dimensions"] == 2

    ports_sha = hashlib.sha256(t.put_calls[0][2]).hexdigest()
    assert t.put_calls[0][0] == f"http://h/v3/blobs/{ports_sha}"  # base_url trailing "/" stripped
    assert t.put_calls[0][1]["Authorization"] == "Bearer tok"
    assert [u.rsplit("/v3/", 1)[1] for u, _, _ in t.post_calls] == ["space-id", "trajectories"]

    url, headers, body = t.post_calls[1]
    payload = json.loads(body)["payload"]
    ext = payload["ext"]["hekbv3"]
    assert payload["title"] == "s1" and payload["tags"] == ["x"]
    assert ext["space"]["space_id"] == "S" * 64 and ext["space"]["board_format"] == "36cell/1"
    assert ext["ports"] == {
        "blob_sha256": ports_sha, "n_steps": 2, "dim": 2, "port_ids": ["P01", "P02"],
        "quant_step_log2": -30,
    }
    assert ext["causal_log"]["blob_sha256"] == hashlib.sha256(t.put_calls[1][2]).hexdigest()
    assert ext["labels"] == {"category": "natural"}
    assert base64.b64decode(headers["X-Audit-Signature"]) == b"\x01" * 64
    assert headers["Content-Type"] == "application/json"


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"rows": []}, "zero tokens"),
        ({"causal_log_records": []}, "causal_log_records is required"),
        ({"causal_log_records": [{"token_index": 0, "seq": 0, "ports": {}, "cycle_raw": {}}]},
         "has 1 entries, rows has 2"),
    ],
)
def test_incomplete_experiences_are_rejected_before_any_http(
    stub_blob_module: list[dict[str, Any]], data: dict[str, Any], match: str
) -> None:
    t = _Transport()
    with pytest.raises(HekbExperienceRejected, match=match):
        _store(t).put_object(_data(**data))
    assert t.put_calls == [] and t.post_calls == []


def test_missing_top_level_field_and_zero_record_causal_log() -> None:
    d = _data()
    del d["space"]
    with pytest.raises(HekbExperienceRejected, match="missing field"):
        _store(_Transport()).put_object(d)
    with pytest.raises(HekbExperienceRejected, match="zero records"):
        hv._encode_causal_log_blob([])
    with pytest.raises(HekbExperienceRejected, match="record 0 missing"):
        hv._encode_causal_log_blob([{"token_index": 0}])


def test_server_errors_map_to_structured_provider_errors(
    stub_blob_module: list[dict[str, Any]],
) -> None:
    with pytest.raises(HekbVNextUnavailable, match="PUT /v3/blobs"):
        _store(_Transport(put_status=500)).put_object(_data())
    with pytest.raises(HekbVNextUnavailable, match="space-id failed"):
        _store(_Transport(space_status=400)).put_object(_data())
    with pytest.raises(HekbExperienceRejected, match="trajectories rejected"):
        _store(_Transport(post_status=422)).put_object(_data())


def test_get_object_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        _store(_Transport()).get_object("x")


def test_blob_encoder_import_failure_is_hekb_vnext_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "storage.trajectory_blob", None)  # forces ImportError
    monkeypatch.setattr(sys, "path", list(sys.path))  # the root insertion is undone afterwards
    with pytest.raises(HekbVNextUnavailable, match="could not import storage.trajectory_blob"):
        _store(_Transport(), tmp_path).put_object(_data())


class _Resp(io.BytesIO):
    status = 201

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> None:
        self.close()


def test_default_http_helpers_return_status_and_text_also_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []

    def ok(req: Any, timeout: float) -> _Resp:
        seen.append((req.get_method(), req.full_url))
        return _Resp(b'{"ok":1}')

    monkeypatch.setattr(hv.urllib.request, "urlopen", ok)
    assert hv._default_http_put("http://h/a", headers={}, data=b"x") == (201, '{"ok":1}')
    assert hv._default_http_post("http://h/b", headers={}, data=b"x") == (201, '{"ok":1}')
    assert seen == [("PUT", "http://h/a"), ("POST", "http://h/b")]

    def err(req: Any, timeout: float) -> _Resp:
        raise urllib.error.HTTPError(req.full_url, 409, "conflict", {}, io.BytesIO(b"nope"))  # type: ignore[arg-type]

    monkeypatch.setattr(hv.urllib.request, "urlopen", err)
    assert hv._default_http_put("http://h/a", headers={}, data=b"x") == (409, "nope")
    assert hv._default_http_post("http://h/b", headers={}, data=b"x") == (409, "nope")
