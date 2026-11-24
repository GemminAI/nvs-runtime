"""``ObjectStore`` adapter for hekb-vnext's real ``/v3/*`` HTTP API, so ``HekbProvider`` can be
composed as ``HekbProvider(store=HekbVNextObjectStore(...))`` instead of the default
in-memory ``MemoryObjectStore`` -- the provider class itself is unchanged; only the
store it is given differs.

Maps one ``put_object(data, labels=...)`` call onto hekb-vnext's real write sequence:
``POST /v3/space-id`` (read-only, gets the canonical ``space_id``) -> ``PUT /v3/blobs/
{sha256}`` for the quantized board-cell matrix (and, when given, the causal log) ->
``POST /v3/trajectories`` (signed, builds ``payload.ext.hekbv3``). This is the live
equivalent of ``hekb-vnext/scripts/ingest_boardrecords.py``'s own batch path, driven one
session's accumulated tokens at a time instead of from pre-written shard files.

``causal_log_records`` is REQUIRED here (unlike hekb-vnext's own
``_require_hekbv3_trajectory_ext``, which treats it as optional for older ingests) --
this slice exists specifically to carry the 38-port causal log the design requires, so a
missing or incomplete one is rejected (``HekbExperienceRejected``), never silently
dropped or half-written.

The quantized-board-cell blob needs hekb-vnext's exact binary format
(``storage/trajectory_blob.py``: 128-byte header + zstd payload + 32-byte tail hash) to
stay decodable by that repo's own readers later -- reimplementing zstd framing here would
duplicate a format this module does not own, so it is imported directly from the
hekb-vnext checkout at ``hekb_vnext_root`` (the same cross-repo import
``ingest_boardrecords.py`` itself uses for the same reason). This is the one place this
adapter departs from "HTTP only" at the provider boundary; everything else goes through
the API. The causal-log blob's JSON canonicalization (``_encode_causal_log_blob``,
matching ``storage/causal_log_blob.py``'s own ``sort_keys``/no-whitespace scheme) is
small enough to duplicate safely -- ``store_service.py`` itself never decodes a blob's
content at write time (confirmed: it does not import either blob module), so exact byte
equality with hekb-vnext's own encoder is a courtesy to later readers, not something the
write path enforces.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from nvs_runtime.core.errors import ProviderError

REQUIRED_CAUSAL_LOG_FIELDS = ("token_index", "seq", "ports", "cycle_raw")
REQUIRED_DATA_FIELDS = (
    "session_id", "port_ids", "quant_step_log2", "board_format", "rows", "space",
)  # fmt: skip

#: injectable HTTP calls: (url, *, headers, data) -> (status_code, text)
HttpCall = Callable[..., tuple[int, str]]


class HekbExperienceRejected(ProviderError):
    """The accumulated session data does not meet this slice's own bar for a
    hekb-vnext experience (by design a causal log missing its 38-port readings or raw
    cycle must be visible as a rejection, never stored as if it were complete)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="HEKB_EXPERIENCE_REJECTED")


class HekbVNextUnavailable(ProviderError):
    """hekb-vnext itself (or its blob wire-format module) could not be reached/imported."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="HEKB_VNEXT_UNAVAILABLE")


def _canonical_bytes(value: Any) -> bytes:
    """Matches hekb-vnext's ``vendor/hekb_vnext/core/object.py::canonical_bytes`` exactly
    (sort_keys, no whitespace, ``ensure_ascii=True``) -- this is the ``object_id``/
    signature preimage, so any difference would make every signature fail to verify."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _encode_causal_log_blob(records: Sequence[Mapping[str, Any]]) -> tuple[bytes, str]:
    """Local duplicate of ``storage/causal_log_blob.py::encode_causal_log_blob``'s
    canonicalization (safe to duplicate -- see module docstring). Raises
    ``HekbExperienceRejected`` instead of a bare ``ValueError`` so it surfaces through
    ``HekbProvider.invoke`` as a structured, visible error (a causal log must be complete)."""
    if not records:
        raise HekbExperienceRejected("causal log has zero records")
    for i, rec in enumerate(records):
        missing = [k for k in REQUIRED_CAUSAL_LOG_FIELDS if k not in rec]
        if missing:
            raise HekbExperienceRejected(f"causal log record {i} missing field(s): {missing}")
    data = json.dumps(list(records), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return data, hashlib.sha256(data).hexdigest()


def _load_trajectory_blob_encoder(hekb_vnext_root: Path) -> Any:
    """Lazily imports hekb-vnext's real binary blob encoder (see module docstring for
    why this one function is imported rather than duplicated)."""
    root = str(Path(hekb_vnext_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from storage.trajectory_blob import DTYPE_INT64, encode_trajectory_blob
    except ImportError as exc:
        raise HekbVNextUnavailable(
            f"could not import storage.trajectory_blob from "
            f"hekb_vnext_root={hekb_vnext_root}: {exc}"
        ) from exc
    return DTYPE_INT64, encode_trajectory_blob


def _default_http_put(url: str, *, headers: dict[str, str], data: bytes) -> tuple[int, str]:
    request = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _default_http_post(url: str, *, headers: dict[str, str], data: bytes) -> tuple[int, str]:
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


class HekbVNextObjectStore:
    """``ObjectStore`` protocol (``put_object``/``get_object``) backed by a real
    hekb-vnext ``store_service`` instance. ``get_object`` is not implemented (this slice
    is write-only; recall is a separate concern) -- callers needing it should use
    ``MemoryObjectStore`` or a future read adapter instead."""

    def __init__(
        self,
        *,
        base_url: str,
        bearer_token: str,
        sign: Callable[[bytes], bytes],
        hekb_vnext_root: Path,
        http_put: HttpCall = _default_http_put,
        http_post: HttpCall = _default_http_post,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._bearer_token = bearer_token
        self._sign = sign
        self._hekb_vnext_root = Path(hekb_vnext_root)
        self._http_put = http_put
        self._http_post = http_post

    def _headers(self, *, signature: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "Content-Type": "application/json",
        }
        if signature is not None:
            headers["X-Audit-Signature"] = signature
        return headers

    def _put_blob(self, blob: bytes) -> str:
        sha256 = hashlib.sha256(blob).hexdigest()
        status, text = self._http_put(
            f"{self._base_url}/v3/blobs/{sha256}",
            headers={
                "Authorization": f"Bearer {self._bearer_token}",
                "Content-Type": "application/octet-stream",
            },
            data=blob,
        )
        if status != 201:
            raise HekbVNextUnavailable(f"PUT /v3/blobs/{sha256} failed: {status} {text}")
        return sha256

    def _space_id(self, space: Mapping[str, Any]) -> str:
        status, text = self._http_post(
            f"{self._base_url}/v3/space-id",
            headers=self._headers(),
            data=json.dumps(space).encode("utf-8"),
        )
        if status != 200:
            raise HekbVNextUnavailable(f"POST /v3/space-id failed: {status} {text}")
        return str(json.loads(text)["space_id"])

    def put_object(self, data: dict[str, Any], *, labels: dict[str, str] | None = None) -> str:
        missing = [k for k in REQUIRED_DATA_FIELDS if k not in data]
        if missing:
            raise HekbExperienceRejected(f"missing field(s): {missing}")
        raw_rows = data["rows"]
        port_ids: list[str] = data["port_ids"]
        if not isinstance(raw_rows, list) or not raw_rows:
            raise HekbExperienceRejected("rows has zero tokens")
        rows: list[list[int]] = []
        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, list) or len(raw_row) != len(port_ids):
                raise HekbExperienceRejected(
                    f"row {index} does not have {len(port_ids)} integer values"
                )
            row: list[int] = []
            for value in raw_row:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise HekbExperienceRejected(
                        f"row {index} has a non-integer value {value!r}"
                    )
                row.append(value)
            rows.append(row)
        raw_excluded = data.get("excluded_ports", [])
        if not isinstance(raw_excluded, list):
            raise HekbExperienceRejected("excluded_ports must be a list of port ids")
        excluded_ports: list[str] = []
        for port in raw_excluded:
            if not isinstance(port, str):
                raise HekbExperienceRejected("excluded_ports must be a list of port ids")
            excluded_ports.append(port)
        causal_log_records = data.get("causal_log_records")
        if not causal_log_records:
            raise HekbExperienceRejected(
                "causal_log_records is required for this slice (the design requires it): "
                "a board record with no causal log is not a complete experience here"
            )
        if len(causal_log_records) != len(rows):
            raise HekbExperienceRejected(
                f"causal_log_records has {len(causal_log_records)} entries, rows has {len(rows)}"
            )

        dtype_int64, encode_trajectory_blob = _load_trajectory_blob_encoder(self._hekb_vnext_root)
        # created_at_ms=0 (fixed): the blob's bytes -- and therefore its sha256 and the
        # resulting object_id -- must be identical on re-sending the same session
        # (idempotent put_object, K8), matching ingest_boardrecords.py's own reasoning.
        ports_bytes, _header = encode_trajectory_blob(
            rows, dimensions=len(port_ids), dtype=dtype_int64, created_at_ms=0
        )
        ports_sha256 = self._put_blob(ports_bytes)

        causal_bytes, _causal_sha_unused = _encode_causal_log_blob(causal_log_records)
        causal_sha256 = self._put_blob(causal_bytes)

        space = dict(data["space"])
        space.setdefault("board_format", data["board_format"])
        space.setdefault("quant_step_log2", data["quant_step_log2"])
        # ingest_boardrecords.py:413-415 sorts the space string. The ports list below
        # keeps `excluded` in cell order, matching lines 457-458 (not the sorted string).
        if excluded_ports:
            space["excluded_columns"] = ",".join(
                f"{port}=valueless" for port in sorted(excluded_ports)
            )
        space_id = self._space_id(space)

        ports: dict[str, Any] = {
            "blob_sha256": ports_sha256,
            "n_steps": len(rows),
            "dim": len(port_ids),
            "port_ids": port_ids,
            "quant_step_log2": data["quant_step_log2"],
        }
        if excluded_ports:
            ports["excluded_columns"] = [
                {"port_id": port, "reason": "valueless"} for port in excluded_ports
            ]
        ext_hekbv3 = {
            "kind": "trajectory",
            "origin": "experience",
            "space": {**space, "space_id": space_id},
            "ports": ports,
            "causal_log": {
                "blob_sha256": causal_sha256,
                "n_steps": len(causal_log_records),
                "schema": "hekbv3.causal_log/1",
            },
            "labels": dict(labels or {}),
            "outcome": {"result": "n/a"},
        }
        payload = {
            "schema": "hekb.experience/2",
            "raw_input": f"edge-hekb-slice session={data['session_id']}",
            "resolved_meaning": None,
            "action": None,
            "title": data.get("title") or data["session_id"],
            "tags": list(data.get("tags") or []),
            "ext": {"hekbv3": ext_hekbv3},
        }
        signature = base64.b64encode(self._sign(_canonical_bytes(payload))).decode("ascii")
        status, text = self._http_post(
            f"{self._base_url}/v3/trajectories",
            headers=self._headers(signature=signature),
            data=json.dumps({"payload": payload}).encode("utf-8"),
        )
        if status != 201:
            raise HekbExperienceRejected(f"POST /v3/trajectories rejected: {status} {text}")
        return str(json.loads(text)["object_id"])

    def get_object(self, object_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("HekbVNextObjectStore is write-only in this slice")


__all__ = [
    "HekbExperienceRejected",
    "HekbVNextObjectStore",
    "HekbVNextUnavailable",
]
