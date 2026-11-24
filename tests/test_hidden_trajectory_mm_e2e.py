"""A meaning-measurement envelope from a copied capture, through the Gate.

``SENSOS_MM_E2E_DIR`` is a further copy of a capture. These tests only read it.
``meaning_mapper`` is optional for the rest of this suite.

theta, sigma, and density come from ``measure_window``. The frame below is a
test fixture (D = 4096, d = 64). It is not a recommended frame.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("meaning_mapper")

import numpy as np
from meaning_mapper.envelope import build_frame_artifact, build_measurement_artifact
from meaning_mapper.envelope.build import _jcs
from meaning_mapper.window import WindowFrame, frame_content_id, measure_window, window_rows

from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.token_temporal import build_token_temporal_artifact
from nvs_runtime.hidden_trajectory.window import build_window_artifact

_ENV = "SENSOS_MM_E2E_DIR"
_TREE = "41982676774eccf9de03117126a44714db57ee2810492baa6f19a19435dcd4de"
_D = 64
_OTHER = "art:" + "9" * 64


def _tree(root: Path) -> tuple[int, str]:
    running = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        digest = hashlib.sha256(path.read_bytes()).hexdigest().encode()
        running.update(rel + b"\0" + digest + b"\n")
    return len(files), running.hexdigest()


@pytest.fixture
def capture_dir() -> Any:
    raw = os.environ.get(_ENV)
    if not raw:
        pytest.skip(f"{_ENV} is unset; no capture copy was given")
    root = Path(raw)
    if not (root / "graph" / "ic00_envelopes.jsonl").is_file():
        pytest.skip(f"{_ENV} has no observation log: {root}")
    before = _tree(root)
    assert before == (12, _TREE)
    yield root
    assert _tree(root) == before


def _gate_client() -> Any:
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


def _seal(client: Any) -> Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]]:
    def seal(built: dict[str, Any], lookup: Mapping[str, Any]) -> dict[str, Any]:
        answer = client.process.call({"op": "seal", "envelope": built, "lookup": dict(lookup)})
        envelope: dict[str, Any] = answer["envelope"]
        return envelope

    return seal


def _observation(data_dir: Path) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    path = data_dir / "graph" / "ic00_envelopes.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        held = json.loads(line)["record"].get("envelope")
        schema = held.get("payload_schema") if isinstance(held, dict) else None
        if isinstance(schema, dict) and schema.get("id") == "nvs.hidden-trajectory":
            found.append(held)
    if len(found) != 1:
        raise AssertionError(f"expected one observation, found {len(found)}")
    return found[0]


def _blob_reader(data_dir: Path) -> Callable[[str], bytes]:
    def read(digest: str) -> bytes:
        path = data_dir / "blobs" / digest[:2] / digest[2:4] / digest
        if not path.is_file():
            raise FileNotFoundError(digest)
        return path.read_bytes()

    return read


def _space_id(materials: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(materials), sort_keys=True, separators=(",", ":")).encode()
    return "spc:" + hashlib.sha256(raw).hexdigest()


def _test_frame(rows: np.ndarray[Any, np.dtype[np.float64]]) -> WindowFrame:
    """Fixture only. Not a recommended frame. ``d`` is 64; ``D`` is the row width."""
    width = int(rows.shape[1])
    basis = np.random.default_rng(0).standard_normal((_D, width)) / math.sqrt(width)
    return WindowFrame(mu=rows.mean(axis=0), basis=basis, sigma0=1e-3 * np.eye(_D))


def _density_numpy(theta: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """RFC-MM003 ``D_s`` with ``c = 1``, written here with numpy, not Meaning Mapper."""
    dimension = int(theta.shape[0])
    magnitudes = np.abs(theta)
    total = float(np.sum(magnitudes))
    if dimension < 2 or total == 0.0:
        return 0.0
    probabilities = magnitudes / total
    nonzero = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    density = 1.0 - entropy / math.log(dimension)
    return float(min(max(density, 0.0), 1.0))


def _independent(
    rows: np.ndarray[Any, np.dtype[np.float64]],
    points: list[dict[str, int]],
    frame: WindowFrame,
) -> tuple[Any, Any, Any]:
    mu = np.array(frame.mu, dtype=np.float64, copy=True)
    basis = np.array(frame.basis, dtype=np.float64, copy=True)
    sigma0 = np.array(frame.sigma0, dtype=np.float64, copy=True)
    theta = np.empty((len(points), basis.shape[0]), dtype=np.float64)
    sigma = np.empty((len(points), basis.shape[0], basis.shape[0]), dtype=np.float64)
    density = np.empty(len(points), dtype=np.float64)
    for index, point in enumerate(points):
        start, length = point["start"], point["length"]
        mean = np.mean(rows[start : start + length], axis=0, dtype=np.float64)
        theta[index] = basis @ (mean - mu)
        sigma[index] = sigma0 / length
        density[index] = _density_numpy(theta[index])
    return theta, sigma, density


def _points(length: int, per_token: bool) -> list[dict[str, int]]:
    if not per_token:
        return [{"start": 0, "length": length}]
    return [{"start": index, "length": 1} for index in range(length)]


def _measurement_id(payload: Mapping[str, Any]) -> str:
    body = {
        "domain": "nvs.meaning-measurement/1",
        "frame_id": payload["frame_id"],
        "input": payload["input"],
        "points": payload["points"],
        "theta.sha256": payload["theta"]["sha256"],
        "sigma.sha256": payload["sigma"]["sha256"],
        "density.sha256": payload["density"]["sha256"],
    }
    return hashlib.sha256(_jcs(body)).hexdigest()


def _world(
    data_dir: Path,
    client: Any,
    *,
    per_token: bool,
    with_axis: bool,
    keep_observation: bool = True,
) -> tuple[Any, dict[str, Any]]:
    observation = _observation(data_dir)
    seal = _seal(client)
    window = build_window_artifact(
        observation,
        _blob_reader(data_dir),
        producer_build="mm-e2e",
        space_id_of=_space_id,
        seal=seal,
    )
    axis = None
    if with_axis:
        axis = build_token_temporal_artifact(
            observation, CaptureArchive(data_dir).read_ids, seal=seal
        )
    rows = window_rows(window.envelope, window.blobs)
    frame = _test_frame(rows)
    assert frame.frame_id == frame_content_id(frame.mu, frame.basis, frame.sigma0)
    length = int(window.envelope["payload"]["window"]["length"])
    points = _points(length, per_token)
    measured = measure_window(rows, points, frame)
    theta, sigma, density = _independent(rows, points, frame)
    assert measured.theta.tobytes() == theta.tobytes()
    assert measured.sigma.tobytes() == sigma.tobytes()
    assert measured.density.tobytes() == density.tobytes()
    artifact = build_measurement_artifact(
        window.envelope,
        None if axis is None else axis.envelope,
        str(frame.frame_id),
        points,
        measured.theta,
        measured.sigma,
        measured.density,
        "mm-e2e",
        seal,
        observation if with_axis else None,
    )
    stored = dict(artifact.stored_parents)
    if keep_observation:
        stored.update(window.stored_parent)
    else:
        stored.pop(observation["artifact_id"], None)
    return artifact, stored


def _verdict(
    client: Any,
    artifact: Any,
    stored: Mapping[str, Any],
    blobs: Mapping[str, bytes],
) -> dict[str, Any]:
    result: dict[str, Any] = client.verdict(
        artifact.envelope,
        lookup=artifact.lookup,
        stored=stored,
        blobs=blobs,
    )
    return result


def _assert_frame_not_supplied(verdict: Mapping[str, Any]) -> None:
    """A measurement with no stored nvs.meaning-frame is PASS_UNVERIFIED."""
    assert verdict["result"] == "PASS_UNVERIFIED", verdict.get("violations")
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["layers"]["L2"]["kind"] == "meaning_measurement"
    assert verdict["layers"]["L2"]["status"] == "PASS_UNVERIFIED"
    assert verdict["violations"] == []
    assert verdict["unverifiable"] == [
        {"layer": "L2", "check": "frame_binding", "reason": "frame_not_supplied"}
    ]


def test_a_window_only_measurement_passes(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=False)
    _assert_frame_not_supplied(_verdict(client, artifact, stored, artifact.blobs))


def test_a_window_only_measurement_without_the_observation_passes(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(
        capture_dir, client, per_token=False, with_axis=False, keep_observation=False
    )
    assert _observation(capture_dir)["artifact_id"] not in stored
    _assert_frame_not_supplied(_verdict(client, artifact, stored, artifact.blobs))


def test_a_window_and_token_temporal_measurement_passes(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=True)
    _assert_frame_not_supplied(_verdict(client, artifact, stored, artifact.blobs))


def test_one_point_per_token_passes(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=True, with_axis=True)
    assert len(artifact.envelope["payload"]["points"]) == 48
    assert artifact.envelope["payload"]["theta"]["shape"] == [48, 64]
    _assert_frame_not_supplied(_verdict(client, artifact, stored, artifact.blobs))


def test_rows_past_the_window_fail(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=True)
    envelope = copy.deepcopy(artifact.envelope)
    envelope["payload"]["points"][0]["rows"]["length"] += 1
    envelope["payload"]["measurement_id"] = _measurement_id(envelope["payload"])
    envelope = _seal(client)(envelope, artifact.lookup)
    verdict = client.verdict(envelope, lookup=artifact.lookup, stored=stored, blobs=artifact.blobs)
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["codes"] == ["meaning_measurement_rows"]


def test_a_missing_theta_blob_fails(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=True)
    blobs = dict(artifact.blobs)
    del blobs[artifact.envelope["payload"]["theta"]["uri"]]
    verdict = client.verdict(artifact.envelope, lookup=artifact.lookup, stored=stored, blobs=blobs)
    assert verdict["result"] == "FAIL"
    assert verdict["codes"] == ["meaning_measurement_blob_not_supplied"]


def test_a_non_positive_definite_sigma_fails(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=True)
    envelope = copy.deepcopy(artifact.envelope)
    ref = envelope["payload"]["sigma"]
    count = int(np.prod(ref["shape"]))
    values = np.zeros(count, dtype="<f8")
    values[0] = -1.0
    raw = values.tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    blobs = dict(artifact.blobs)
    blobs.pop(ref["uri"], None)
    ref["sha256"] = digest
    ref["uri"] = "blob:sha256:" + digest
    blobs[ref["uri"]] = raw
    envelope["payload"]["measurement_id"] = _measurement_id(envelope["payload"])
    envelope = _seal(client)(envelope, artifact.lookup)
    verdict = client.verdict(envelope, lookup=artifact.lookup, stored=stored, blobs=blobs)
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["codes"] == ["meaning_measurement_sigma"]


def test_a_window_from_another_observation_fails(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=True)
    stored = copy.deepcopy(stored)
    window_id = artifact.envelope["payload"]["input"]["window_artifact_id"]
    stored[window_id]["envelope"]["parents"][0]["artifact_ids"] = [_OTHER]
    verdict = client.verdict(
        artifact.envelope, lookup=artifact.lookup, stored=stored, blobs=artifact.blobs
    )
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["codes"] == ["meaning_measurement_lineage"]


def test_an_nvs_runtime_token_axis_is_not_the_temporal_schema(capture_dir: Path) -> None:
    """The stored token_axis envelope is another nvs-runtime envelope.

    L1 still sees the measurement (meaning-mapper, interpretation). The payload
    check sees a parent that is not nvs.token-temporal 0.1.0.
    """
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=True)
    stored = copy.deepcopy(stored)
    axis_id = artifact.envelope["payload"]["input"]["token_axis_artifact_id"]
    parent = stored[axis_id]["envelope"]
    parent["producer"]["name"] = "nvs-runtime"
    parent["payload_schema"] = {"id": "msr.trajectory-window", "version": "0.10.0"}
    verdict = client.verdict(
        artifact.envelope, lookup=artifact.lookup, stored=stored, blobs=artifact.blobs
    )
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["layers"]["L2"]["kind"] == "meaning_measurement"
    assert verdict["layers"]["L2"]["status"] == "FAIL"
    assert verdict["codes"] == ["meaning_measurement_axis"]


def test_a_missing_observation_fails_when_a_token_axis_is_present(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(
        capture_dir, client, per_token=False, with_axis=True, keep_observation=False
    )
    verdict = client.verdict(
        artifact.envelope, lookup=artifact.lookup, stored=stored, blobs=artifact.blobs
    )
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["layers"]["L2"]["kind"] == "meaning_measurement"
    assert verdict["layers"]["L2"]["status"] == "FAIL"
    assert verdict["codes"] == ["meaning_measurement_observation_not_supplied"]


def test_a_free_frame_id_fails_the_schema(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=False)
    envelope = copy.deepcopy(artifact.envelope)
    envelope["payload"]["frame_id"] = "f"
    envelope["payload"]["measurement_id"] = _measurement_id(envelope["payload"])
    envelope = _seal(client)(envelope, artifact.lookup)
    verdict = client.verdict(envelope, lookup=artifact.lookup, stored=stored, blobs=artifact.blobs)
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["codes"] == ["schema"]


def test_a_changed_mu_is_another_frame_id(capture_dir: Path) -> None:
    client = _gate_client()
    window = build_window_artifact(
        _observation(capture_dir),
        _blob_reader(capture_dir),
        producer_build="mm-e2e",
        space_id_of=_space_id,
        seal=_seal(client),
    )
    rows = window_rows(window.envelope, window.blobs)
    frame = _test_frame(rows)
    shifted = np.array(frame.mu, dtype=np.float64, copy=True)
    shifted[0] += 1.0
    other = WindowFrame(
        mu=shifted,
        basis=np.array(frame.basis, dtype=np.float64, copy=True),
        sigma0=np.array(frame.sigma0, dtype=np.float64, copy=True),
    )
    assert frame.frame_id.startswith("frame:sha256:")
    assert other.frame_id != frame.frame_id


def test_renaming_the_token_axis_producer_fails(capture_dir: Path) -> None:
    """The stored token_axis producer is nvs-runtime. The schema stays 0.1.0.

    L1 of the measurement passes. L2 reports meaning_measurement_axis.
    """
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=True)
    stored = copy.deepcopy(stored)
    axis_id = artifact.envelope["payload"]["input"]["token_axis_artifact_id"]
    stored[axis_id]["envelope"]["producer"]["name"] = "nvs-runtime"
    verdict = client.verdict(
        artifact.envelope, lookup=artifact.lookup, stored=stored, blobs=artifact.blobs
    )
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["layers"]["L2"]["kind"] == "meaning_measurement"
    assert verdict["layers"]["L2"]["status"] == "FAIL"
    assert verdict["codes"] == ["meaning_measurement_axis"]


def _scene(capture_dir: Path, client: Any) -> tuple[Any, Any, Any, WindowFrame, Any]:
    observation = _observation(capture_dir)
    seal = _seal(client)
    window = build_window_artifact(
        observation,
        _blob_reader(capture_dir),
        producer_build="mm-e2e",
        space_id_of=_space_id,
        seal=seal,
    )
    rows = window_rows(window.envelope, window.blobs)
    return observation, window, rows, _test_frame(rows), seal


def _store_frame(
    stored: Mapping[str, Any],
    measurement_blobs: Mapping[str, bytes],
    frame_envelope: Mapping[str, Any],
    frame_blobs: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    held = dict(stored)
    held[frame_envelope["artifact_id"]] = {"envelope": frame_envelope}
    blobs = dict(measurement_blobs)
    blobs.update(frame_blobs)
    return held, blobs


def _asymmetric_sigma(
    frame: WindowFrame,
    frame_envelope: Mapping[str, Any],
    frame_blobs: Mapping[str, bytes],
    seal: Any,
) -> tuple[dict[str, Any], dict[str, bytes], str]:
    """A frame whose sigma0 is not symmetric. The frame_id is the new bytes."""
    envelope = copy.deepcopy(dict(frame_envelope))
    blobs = dict(frame_blobs)
    ref = envelope["payload"]["sigma0"]
    raw = bytearray(blobs.pop(ref["uri"]))
    rank = int(envelope["payload"]["d"])
    struct.pack_into("<d", raw, (0 * rank + 1) * 8, 1.0)
    struct.pack_into("<d", raw, (1 * rank + 0) * 8, 2.0)
    sigma0 = np.frombuffer(bytes(raw), dtype="<f8").reshape(rank, rank).copy()
    new_id = frame_content_id(frame.mu, frame.basis, sigma0)
    digest = hashlib.sha256(raw).hexdigest()
    ref["sha256"] = digest
    ref["uri"] = "blob:sha256:" + digest
    blobs[ref["uri"]] = bytes(raw)
    envelope["payload"]["frame_id"] = new_id
    sealed = seal(envelope, {})
    return sealed, blobs, new_id


def test_the_frame_artifact_passes(capture_dir: Path) -> None:
    client = _gate_client()
    observation, _window, _rows, frame, seal = _scene(capture_dir, client)
    artifact = build_frame_artifact(frame, observation["space_id"], "mm-e2e", seal, observation)
    verdict = client.verdict(artifact.envelope, lookup=artifact.lookup, blobs=artifact.blobs)
    assert verdict["result"] == "PASS", verdict.get("violations")
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["layers"]["L2"]["kind"] == "meaning_frame"
    assert verdict["layers"]["L2"]["status"] == "PASS"
    assert verdict["violations"] == []
    assert verdict["unverifiable"] == []


def test_a_measurement_with_its_frame_passes(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=False)
    observation, _window, _rows, frame, seal = _scene(capture_dir, client)
    frame_artifact = build_frame_artifact(
        frame, observation["space_id"], "mm-e2e", seal, observation
    )
    held, blobs = _store_frame(
        stored, artifact.blobs, frame_artifact.envelope, frame_artifact.blobs
    )
    verdict = client.verdict(artifact.envelope, lookup=artifact.lookup, stored=held, blobs=blobs)
    assert verdict["result"] == "PASS", verdict.get("violations")
    assert verdict["layers"]["L2"]["kind"] == "meaning_measurement"
    assert verdict["layers"]["L2"]["status"] == "PASS"
    assert verdict["violations"] == []
    assert verdict["unverifiable"] == []


def test_the_frame_makes_the_measurement_unverified_pass_or_fail(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=False)
    _assert_frame_not_supplied(_verdict(client, artifact, stored, artifact.blobs))

    observation, window, rows, frame, seal = _scene(capture_dir, client)
    frame_artifact = build_frame_artifact(
        frame, observation["space_id"], "mm-e2e", seal, observation
    )
    held, blobs = _store_frame(
        stored, artifact.blobs, frame_artifact.envelope, frame_artifact.blobs
    )
    passed = client.verdict(artifact.envelope, lookup=artifact.lookup, stored=held, blobs=blobs)
    assert passed["result"] == "PASS", passed.get("violations")
    assert passed["layers"]["L2"]["kind"] == "meaning_measurement"
    assert passed["unverifiable"] == []

    broken, broken_blobs, broken_id = _asymmetric_sigma(
        frame, frame_artifact.envelope, frame_artifact.blobs, seal
    )
    length = int(window.envelope["payload"]["window"]["length"])
    points = _points(length, False)
    measured = measure_window(rows, points, frame)
    measurement = build_measurement_artifact(
        window.envelope,
        None,
        broken_id,
        points,
        measured.theta,
        measured.sigma,
        measured.density,
        "mm-e2e",
        seal,
        None,
    )
    failed_stored, failed_blobs = _store_frame(
        measurement.stored_parents, measurement.blobs, broken, broken_blobs
    )
    failed_stored.update(window.stored_parent)
    failed = client.verdict(
        measurement.envelope,
        lookup=measurement.lookup,
        stored=failed_stored,
        blobs=failed_blobs,
    )
    assert failed["result"] == "FAIL"
    assert failed["layers"]["L2"]["kind"] == "meaning_measurement"
    assert failed["layers"]["L2"]["status"] == "FAIL"
    assert failed["codes"] == ["meaning_measurement_frame_invalid"]


def test_a_frame_for_another_space_fails_the_measurement(capture_dir: Path) -> None:
    client = _gate_client()
    artifact, stored = _world(capture_dir, client, per_token=False, with_axis=False)
    observation, _window, _rows, frame, seal = _scene(capture_dir, client)
    other = "hid:" + "ab" * 32
    assert other != observation["space_id"]
    frame_artifact = build_frame_artifact(frame, other, "mm-e2e", seal, observation)
    held, blobs = _store_frame(
        stored, artifact.blobs, frame_artifact.envelope, frame_artifact.blobs
    )
    verdict = client.verdict(artifact.envelope, lookup=artifact.lookup, stored=held, blobs=blobs)
    assert verdict["result"] == "FAIL"
    assert verdict["layers"]["L2"]["kind"] == "meaning_measurement"
    assert verdict["layers"]["L2"]["status"] == "FAIL"
    assert verdict["codes"] == ["meaning_measurement_frame_space"]
