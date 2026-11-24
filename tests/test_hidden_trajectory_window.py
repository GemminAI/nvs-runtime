"""The window adapter: assemble msr.trajectory-window 0.10.0 from an observation.

The product module does not import MSR. The optional end-to-end test does, and
only when ``SENSOS_WINDOW_E2E_DIR`` names a copy of a capture.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nvs_runtime.hidden_trajectory.payload import token_index_bytes
from nvs_runtime.hidden_trajectory.window import (
    WindowArtifact,
    WindowInputError,
    build_window_artifact,
)

LENGTH = 6
DIMENSION = 8
_HEX = "ab" * 32


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rows(length: int = LENGTH, dimension: int = DIMENSION, *, nan: bool = False) -> bytes:
    count = length * dimension
    values = [float(index) for index in range(count)]
    if nan:
        values[0] = float("nan")
    return struct.pack(f"<{count}f", *values)


def _observation(
    points: bytes,
    *,
    length: int = LENGTH,
    dimension: int = DIMENSION,
    shape: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    trajectory_id = _sha(points)
    tokens = token_index_bytes(length)
    token_sha = _sha(tokens)
    payload = {
        "source": "T-hid",
        "position": "final_norm",
        "point": "final_norm",
        "kind": "final_norm",
        "layer": "final_norm",
        "length": length,
        "dimension": dimension,
        "points": {
            "unit": "activation",
            "dtype": "float32",
            "shape": [length, dimension] if shape is None else shape,
        },
        "token_index": {"sha256": token_sha, "unit": "token_position"},
        "raw_ref": {"trajectory_id": trajectory_id, "copied": False},
        "row_rule": "llama31-mlx-final-norm/1",
        "model_build_id": "model-a",
        "compute_environment": "test-env",
    }
    envelope = {
        "payload_schema": {"id": "nvs.hidden-trajectory", "version": "0.1.0"},
        "payload": payload,
        "artifact_id": "art:" + _HEX,
        "content_id": "cnt:" + _HEX,
        "session_id": "ses:" + _HEX,
        "observation_id": "obs:" + _HEX,
        "seq": 1,
        "produced_at": "2026-11-20T00:00:00Z",
    }
    return envelope, {trajectory_id: points, token_sha: tokens}


def _seal(envelope: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(
        json.dumps({"envelope": envelope, "lookup": lookup}, sort_keys=True).encode()
    ).hexdigest()
    envelope["content_id"] = "cnt:" + digest
    envelope["artifact_id"] = "art:" + digest
    return envelope


def _space_id(materials: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(materials, sort_keys=True).encode()
    ).hexdigest()
    return "hid:" + digest


def _build(
    envelope: dict[str, Any],
    blobs: dict[str, bytes],
    **kwargs: Any,
) -> WindowArtifact:
    def reader(digest: str) -> bytes:
        if digest not in blobs:
            raise FileNotFoundError(digest)
        return blobs[digest]

    arguments: dict[str, Any] = {
        "producer_build": "build-1",
        "space_id_of": _space_id,
        "seal": _seal,
    }
    arguments.update(kwargs)
    return build_window_artifact(envelope, reader, **arguments)


def _canonical(envelope: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_a_small_window_is_stable_and_names_its_parent() -> None:
    envelope, blobs = _observation(_rows())
    first = _build(envelope, blobs)
    second = _build(envelope, blobs)
    assert _canonical(first.envelope) == _canonical(second.envelope)
    parents = first.envelope["parents"]
    assert parents == [
        {
            "role": "source_trajectory",
            "ordered": False,
            "artifact_ids": [envelope["artifact_id"]],
        }
    ]
    assert type(first.envelope["payload"]["dimension"]) is type(envelope["payload"]["dimension"])
    assert first.envelope["payload"]["dimension"] == DIMENSION
    assert first.envelope["contract"] == {"id": "IC-00", "version": "1.1.0"}
    assert first.envelope["producer"]["name"] == "nvs-runtime"
    assert first.envelope["kind"] == "observation"
    assert first.origin["trajectory_id"] == envelope["payload"]["raw_ref"]["trajectory_id"]
    assert first.stored_parent[envelope["artifact_id"]]["envelope"] is envelope


def test_the_full_length_is_the_default_window() -> None:
    envelope, blobs = _observation(_rows())
    built = _build(envelope, blobs)
    window = built.envelope["payload"]["window"]
    assert window == {
        "rule": "rows",
        "start": 0,
        "length": LENGTH,
        "content_sha256": envelope["payload"]["raw_ref"]["trajectory_id"],
    }


def test_a_short_window_slices_the_rows() -> None:
    envelope, blobs = _observation(_rows())
    built = _build(envelope, blobs, start=2, length=3)
    payload = built.envelope["payload"]
    assert payload["window"]["start"] == 2
    assert payload["window"]["length"] == 3
    assert payload["points"]["shape"] == [3, DIMENSION]
    assert payload["dimension"] == DIMENSION
    point_bytes = built.blobs[payload["points"]["uri"]]
    assert len(point_bytes) == 3 * DIMENSION * 4
    assert _sha(point_bytes) == payload["window"]["content_sha256"]


def test_a_blob_of_the_wrong_size_is_refused() -> None:
    points = _rows()
    envelope, blobs = _observation(points)
    blobs[envelope["payload"]["raw_ref"]["trajectory_id"]] = points + b"\x00\x00"
    with pytest.raises(WindowInputError) as caught:
        _build(envelope, blobs)
    assert caught.value.field == "points"


def test_a_blob_with_a_different_sha256_is_refused() -> None:
    points = _rows()
    envelope, blobs = _observation(points)
    flipped = bytearray(points)
    flipped[0] ^= 0x01
    blobs[envelope["payload"]["raw_ref"]["trajectory_id"]] = bytes(flipped)
    with pytest.raises(WindowInputError) as caught:
        _build(envelope, blobs)
    assert caught.value.field == "sha256"


def test_a_nan_is_refused() -> None:
    points = _rows(nan=True)
    envelope, blobs = _observation(points)
    with pytest.raises(WindowInputError) as caught:
        _build(envelope, blobs)
    assert caught.value.field == "points"
    assert "NaN" in caught.value.detail


def test_a_declared_dimension_that_disagrees_with_the_shape_is_refused() -> None:
    points = _rows()
    envelope, blobs = _observation(points, shape=[LENGTH, DIMENSION - 1])
    with pytest.raises(WindowInputError) as caught:
        _build(envelope, blobs)
    assert caught.value.field == "dimension"


def test_a_window_past_the_origin_is_refused() -> None:
    envelope, blobs = _observation(_rows())
    with pytest.raises(WindowInputError) as caught:
        _build(envelope, blobs, start=4, length=3)
    assert caught.value.field == "window"


def test_the_module_does_not_import_msr_or_meaning_mapper_or_sensos_inference() -> None:
    source_path = Path(
        __import__(
            "nvs_runtime.hidden_trajectory.window", fromlist=["window"]
        ).__file__
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    banned = {"msr", "meaning_mapper", "sensos_inference"}
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    assert banned.isdisjoint(found)
    text = source_path.read_text(encoding="utf-8")
    for name in banned:
        assert f"import {name}" not in text


def _gate_client():
    if not os.environ.get("GATE_PY") or not os.environ.get("SENSOS_TOOLS"):
        pytest.skip("GATE_PY or SENSOS_TOOLS is unset; the Gate child cannot be started")
    if not Path(os.environ["GATE_PY"]).is_file():
        pytest.skip("GATE_PY is not a file; the Gate child cannot be started")
    try:
        from nvs_runtime.hidden_trajectory.gate_client import GateClient
    except Exception as error:
        pytest.skip(f"GateClient could not be imported: {error}")
    try:
        return GateClient.from_environment()
    except Exception as error:
        pytest.skip(f"GateClient.from_environment failed: {error}")


def _producer_build() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def test_the_gate_passes_the_window_and_fails_a_different_producer() -> None:
    client = _gate_client()
    envelope, blobs = _observation(_rows(length=4, dimension=2), length=4, dimension=2)

    def space_id_of(materials):
        return client.process.call({"op": "hid_space_id", "materials": dict(materials)})[
            "space_id"
        ]

    def seal(built, lookup):
        return client.process.call(
            {"op": "seal", "envelope": built, "lookup": lookup}
        )["envelope"]

    artifact = _build(
        envelope,
        blobs,
        producer_build=_producer_build(),
        space_id_of=space_id_of,
        seal=seal,
    )
    verdict = client.verdict(
        artifact.envelope,
        lookup=artifact.lookup,
        stored=artifact.stored_parent,
        blobs=artifact.blobs,
    )
    assert verdict["result"] == "PASS", verdict["violations"]
    forged = json.loads(json.dumps(artifact.envelope))
    forged["producer"]["name"] = "sensos-inference"
    resealed = seal(forged, artifact.lookup)
    refused = client.verdict(
        resealed,
        lookup=artifact.lookup,
        stored=artifact.stored_parent,
        blobs=artifact.blobs,
    )
    assert refused["result"] == "FAIL"
    assert any(item["code"] == "window_producer" for item in refused["violations"])


def _capture_observation(data_dir: Path) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    path = data_dir / "graph" / "ic00_envelopes.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)["record"]
        held = record.get("envelope")
        schema = held.get("payload_schema") if isinstance(held, dict) else None
        if isinstance(schema, dict) and schema.get("id") == "nvs.hidden-trajectory":
            found.append(held)
    if len(found) != 1:
        raise AssertionError(f"expected one observation, found {len(found)}")
    return found[0]


def _store_reader(data_dir: Path):
    def reader(digest: str) -> bytes:
        path = data_dir / "blobs" / digest[:2] / digest[2:4] / digest
        return path.read_bytes()

    return reader


def test_geometry_is_measured_only_when_msr_imports() -> None:
    raw = os.environ.get("SENSOS_WINDOW_E2E_DIR")
    if not raw:
        pytest.skip("SENSOS_WINDOW_E2E_DIR is unset; no capture copy was given")
    msr_src = Path.home() / "GemminAI" / "meaning-space-runtime-wt-contract-ver" / "src"
    if not (msr_src / "msr" / "geometry" / "__init__.py").is_file():
        pytest.skip(f"MSR src is not at {msr_src}")
    sys.path.insert(0, str(msr_src))
    try:
        import msr.geometry as geometry
    except Exception as error:
        pytest.skip(f"msr could not be imported: {error}")
    data_dir = Path(raw)
    client = _gate_client()
    observation = _capture_observation(data_dir)

    def space_id_of(materials):
        return client.process.call({"op": "hid_space_id", "materials": dict(materials)})[
            "space_id"
        ]

    def seal(built, lookup):
        return client.process.call(
            {"op": "seal", "envelope": built, "lookup": lookup}
        )["envelope"]

    artifact = build_window_artifact(
        observation,
        _store_reader(data_dir),
        producer_build=_producer_build(),
        space_id_of=space_id_of,
        seal=seal,
    )
    window = geometry.window_from_envelope(
        artifact.envelope,
        artifact.blobs,
        parent_contents={
            artifact.envelope["parents"][0]["artifact_ids"][0]: observation["content_id"]
        },
    )
    context = geometry.EmitContext(
        observation["session_id"],
        observation["observation_id"],
        artifact.envelope["seq"] + 1,
        producer_build=geometry.envelope.default_build(),
        produced_at=observation["produced_at"],
    )
    measured = geometry.emit_single(
        window.trajectory,
        context,
        trajectory_ref=window.ref,
        window=window,
        contract_version="1.1.0",
    )
    assert measured.envelope["contract"] == {"id": "IC-00", "version": "1.1.0"}
    assert measured.envelope["payload_schema"]["id"] == "msr.geometry-measurement"
