"""Token temporal from a copied capture, through the builder and the Gate.

``SENSOS_TOKEN_TEMPORAL_E2E_DIR`` is a copy of a capture. These tests only read
it. A missing or replaced id list is tried on a temporary copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from nvs_runtime.hidden_trajectory.capture_archive import ArchiveError, CaptureArchive
from nvs_runtime.hidden_trajectory.token_temporal import (
    TokenTemporalInputError,
    build_token_temporal_artifact,
)

_ENV = "SENSOS_TOKEN_TEMPORAL_E2E_DIR"
_OTHER_SHA = "ab" * 32
_OTHER_TOKENIZER = "cd" * 32


def _snapshot(root: Path) -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        found.append((path.relative_to(root).as_posix(), digest))
    return tuple(found)


@pytest.fixture
def capture_dir() -> Any:
    raw = os.environ.get(_ENV)
    if not raw:
        pytest.skip(f"{_ENV} is unset; no capture copy was given")
    root = Path(raw)
    if not (root / "graph" / "ic00_envelopes.jsonl").is_file():
        pytest.skip(f"{_ENV} has no observation log: {root}")
    if not (root / "capture" / "records.jsonl").is_file():
        pytest.skip(f"{_ENV} has no capture records: {root}")
    before = _snapshot(root)
    yield root
    assert _snapshot(root) == before


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


def _seal(client: Any) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    def seal(built: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any]:
        answer = client.process.call({"op": "seal", "envelope": built, "lookup": lookup})
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


def _stored(data_dir: Path) -> dict[str, Any]:
    stored: dict[str, Any] = {}
    path = data_dir / "graph" / "ic00_envelopes.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        held = json.loads(line)["record"].get("envelope")
        if isinstance(held, dict) and isinstance(held.get("artifact_id"), str):
            stored[held["artifact_id"]] = {"envelope": held}
    return stored


def _record(data_dir: Path) -> tuple[int, dict[str, Any]]:
    path = data_dir / "capture" / "records.jsonl"
    rows: list[tuple[int, dict[str, Any]]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line:
            rows.append((number, json.loads(line)["record"]))
    if len(rows) != 1:
        raise AssertionError(f"expected one capture record, found {len(rows)}")
    via_archive = list(CaptureArchive(data_dir).records())
    if via_archive != [rows[0][1]]:
        raise AssertionError("CaptureArchive.records does not match records.jsonl")
    return rows[0]


def _artifact(data_dir: Path, client: Any) -> Any:
    return build_token_temporal_artifact(
        _observation(data_dir),
        CaptureArchive(data_dir).read_ids,
        seal=_seal(client),
    )


def test_the_archive_ids_match_the_record_and_the_observation(capture_dir: Path) -> None:
    observation = _observation(capture_dir)
    line_number, record = _record(capture_dir)
    binding = observation["payload"]["binding"]
    provenance = observation["payload"]["capture_provenance"]
    capture = record["capture"]
    assert line_number == 1
    assert capture["n_in"] == binding["n_in"]
    assert capture["n_gen"] == binding["n_gen"]
    assert capture["model_build_id"] == observation["payload"]["model_build_id"]
    assert record["observation_id"] == observation["observation_id"]
    assert record["session_id"] == observation["session_id"]
    archive = CaptureArchive(capture_dir)
    pairs = (("prompt_ids_sha256", "n_in"), ("gen_token_ids_sha256", "n_gen"))
    for hash_key, count_key in pairs:
        digest = provenance[hash_key]
        assert digest == capture[hash_key]
        data = archive.read_ids(digest)
        assert data is not None
        assert hashlib.sha256(data).hexdigest() == digest
        assert len(data) == 8 * binding[count_key]


def test_the_artifact_names_the_observation_and_repeats(capture_dir: Path) -> None:
    client = _gate_client()
    observation = _observation(capture_dir)
    reader = CaptureArchive(capture_dir).read_ids
    seal = _seal(client)
    first = build_token_temporal_artifact(observation, reader, seal=seal)
    second = build_token_temporal_artifact(observation, reader, seal=seal)
    assert first.envelope == second.envelope
    assert first.envelope["parents"] == [
        {
            "role": "source_observation",
            "ordered": False,
            "artifact_ids": [observation["artifact_id"]],
        }
    ]
    assert first.envelope["observation_id"] == observation["observation_id"]
    assert first.envelope["session_id"] == observation["session_id"]
    assert first.envelope["producer"] == {
        "name": observation["producer"]["name"],
        "version": observation["producer"]["version"],
        "build": observation["producer"]["build"],
    }


def test_the_gate_passes_the_artifact_from_the_capture(capture_dir: Path) -> None:
    client = _gate_client()
    observation = _observation(capture_dir)
    stored = _stored(capture_dir)
    artifact = _artifact(capture_dir, client)
    assert observation["artifact_id"] in stored
    verdict = client.verdict(
        artifact.envelope,
        lookup=artifact.lookup,
        stored=stored,
        blobs=artifact.blobs,
    )
    assert verdict["result"] == "PASS", verdict["violations"]
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["layers"]["L2"]["kind"] == "ic02"
    assert verdict["violations"] == []


@pytest.mark.parametrize(
    ("case", "codes"),
    [
        ("byte", ["S24"]),
        ("missing_blob", ["token_temporal_blob_not_supplied"]),
        ("n", ["token_temporal_n", "token_temporal_segment_id"]),
        ("parent_hash", ["token_temporal_hash"]),
        ("tokenizer", ["token_temporal_identity", "token_temporal_segment_id"]),
        # L1 rejects the producer (S4) and does not reach token_temporal_producer.
        ("producer", ["S4"]),
    ],
)
def test_the_gate_fails_one_change_to_the_capture_artifact(
    capture_dir: Path, case: str, codes: list[str]
) -> None:
    client = _gate_client()
    observation = _observation(capture_dir)
    stored = _stored(capture_dir)
    artifact = _artifact(capture_dir, client)
    envelope = json.loads(json.dumps(artifact.envelope))
    blobs = dict(artifact.blobs)
    uri = envelope["payload"]["segments"][0]["ids"]["uri"]
    reseal = False
    if case == "byte":
        flipped = bytearray(blobs[uri])
        flipped[0] ^= 1
        blobs[uri] = bytes(flipped)
    elif case == "missing_blob":
        del blobs[uri]
    elif case == "n":
        envelope["payload"]["segments"][0]["n"] += 1
        reseal = True
    elif case == "parent_hash":
        stored = json.loads(json.dumps(stored))
        parent = stored[observation["artifact_id"]]["envelope"]
        parent["payload"]["capture_provenance"]["gen_token_ids_sha256"] = _OTHER_SHA
    elif case == "tokenizer":
        envelope["payload"]["tokenizer_sha256"] = _OTHER_TOKENIZER
        reseal = True
    else:
        envelope["producer"]["name"] = "other-producer"
        reseal = True
    if reseal:
        envelope = _seal(client)(envelope, artifact.lookup)
    verdict = client.verdict(
        envelope,
        lookup=artifact.lookup,
        stored=stored,
        blobs=blobs,
    )
    assert verdict["result"] == "FAIL", verdict
    assert verdict["codes"] == codes
    if case == "producer":
        assert verdict["layers"]["L1"]["status"] == "FAIL"
        assert verdict["layers"]["L2"]["status"] == "not_run"
    else:
        assert verdict["layers"]["L1"]["status"] == "PASS"
        assert verdict["layers"]["L2"]["kind"] == "ic02"


def _file_reader(data_dir: Path) -> Callable[[str], bytes | None]:
    """The bytes of ``capture/ids``, or ``None`` when the file is absent.

    Unlike ``CaptureArchive.read_ids``, a file whose bytes do not match its name
    is returned, so the builder sees the mismatch itself.
    """

    def ids_reader(digest: str) -> bytes | None:
        path = data_dir / "capture" / "ids" / digest[:2] / digest[2:4] / digest
        if not path.is_file():
            return None
        return path.read_bytes()

    return ids_reader


def test_a_missing_or_replaced_id_list_builds_nothing(
    capture_dir: Path, tmp_path: Path
) -> None:
    client = _gate_client()
    observation = _observation(capture_dir)
    before = client.verdict(observation)
    prompt = observation["payload"]["capture_provenance"]["prompt_ids_sha256"]

    def attempt(name: str, mutate: Callable[[Path], None]) -> str:
        dest = tmp_path / name
        shutil.copytree(capture_dir, dest)
        mutate(dest)
        with pytest.raises(TokenTemporalInputError) as caught:
            build_token_temporal_artifact(
                observation, _file_reader(dest), seal=_seal(client)
            )
        return str(caught.value)

    def drop_directory(dest: Path) -> None:
        shutil.rmtree(dest / "capture" / "ids")

    def drop_one(dest: Path) -> None:
        (dest / "capture" / "ids" / prompt[:2] / prompt[2:4] / prompt).unlink()

    def replace_bytes(dest: Path) -> None:
        path = dest / "capture" / "ids" / prompt[:2] / prompt[2:4] / prompt
        path.write_bytes(b"\x00" * path.stat().st_size)

    absent = "prompt_ids: the archive has no bytes for this sha256"
    assert attempt("directory", drop_directory) == absent
    assert attempt("one_file", drop_one) == absent
    assert attempt("replaced_bytes", replace_bytes) == (
        "prompt_ids: sha256 of the bytes is not the key"
    )
    after = client.verdict(observation)
    assert after["result"] == before["result"] == "PASS_UNVERIFIED"
    assert after["codes"] == before["codes"] == []
    assert [(item["check"], item["reason"]) for item in after["unverifiable"]] == [
        ("token_index_bytes", "blob_not_supplied"),
        ("points_bytes", "blob_not_supplied"),
        ("contract_binding", "contract_not_supplied"),
    ]
    assert after["layers"]["L2"]["kind"] == "hidden_trajectory"


def test_the_gate_rejects_a_window_producer_on_the_token_temporal(
    capture_dir: Path,
) -> None:
    client = _gate_client()
    _observation(capture_dir)
    stored = _stored(capture_dir)
    artifact = _artifact(capture_dir, client)
    envelope = json.loads(json.dumps(artifact.envelope))
    envelope["producer"]["name"] = "nvs-runtime"
    envelope = _seal(client)(envelope, artifact.lookup)
    verdict = client.verdict(
        envelope,
        lookup=artifact.lookup,
        stored=stored,
        blobs=artifact.blobs,
    )
    assert verdict["result"] == "FAIL", verdict
    assert verdict["layers"]["L1"]["status"] == "PASS"
    assert verdict["layers"]["L2"]["kind"] == "ic02"
    assert verdict["codes"] == ["token_temporal_producer"]


def test_read_ids_wraps_an_archive_hash_mismatch(
    capture_dir: Path, tmp_path: Path
) -> None:
    client = _gate_client()
    observation = _observation(capture_dir)
    prompt = observation["payload"]["capture_provenance"]["prompt_ids_sha256"]
    dest = tmp_path / "archive"
    shutil.copytree(capture_dir, dest)
    path = dest / "capture" / "ids" / prompt[:2] / prompt[2:4] / prompt
    path.write_bytes(b"\x00" * path.stat().st_size)
    with pytest.raises(TokenTemporalInputError) as caught:
        build_token_temporal_artifact(
            observation, CaptureArchive(dest).read_ids, seal=_seal(client)
        )
    cause = caught.value.__cause__ or caught.value.__context__
    assert isinstance(cause, ArchiveError)
