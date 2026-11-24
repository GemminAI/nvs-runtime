"""hidden_trajectory.payload: the payload of one position of a contract v2 (design U6 v0.1,
sections 6 and 7.1; tests T-U6-E02..E06, E08, E09 and the payload side of T-U6-R04, R06, R07).

No test here needs nvs_edge or sensos: the contract is the stored U4 sample, and the module under
test imports the standard library and nvs_runtime.core.errors only (the last test checks that).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

import nvs_runtime.hidden_trajectory.payload as payload_module
from nvs_runtime.hidden_trajectory import (
    PAYLOAD_SCHEMA_ID,
    PAYLOAD_SCHEMA_VERSION,
    POLICY_ALL_ROWS,
    PayloadNotBuilt,
    blob_uri,
    build_payload,
    kind_of,
    legacy_position,
    position_obstacles,
    session_obstacles,
    token_index_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures" / "hidden_trajectory"
SAMPLE = json.loads((FIXTURES / "contract-v2-u4-base.json").read_text(encoding="utf-8"))
REF = {
    "name": "RawObservationContract",
    "version": 2,
    "contract_sha256": SAMPLE["contract_sha256"],
}
POINTS = ("block_out:3", "block_out:7", "mlp_out:7", "final_norm")
N_IN, N_GEN, DIMENSION = 5, 6, 16

#: The keys of the payload schema nvs.hidden-trajectory 0.1.0 (sensos contracts), all required.
PAYLOAD_KEYS = {
    "source",
    "position",
    "point",
    "kind",
    "layer",
    "length",
    "dimension",
    "points",
    "token_index",
    "raw_ref",
    "row_rule",
    "binding",
    "final_norm_applied",
    "state_dtype",
    "model_build_id",
    "compute_environment",
    "policy",
    "capture_provenance",
    "contract_ref",
    "unknowns",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE["document"])


def position(doc: dict[str, Any], point: str) -> dict[str, Any]:
    return next(p for p in doc["positions"] if p["point"] == point)


def rows(steps: int = N_GEN, dimension: int = DIMENSION, salt: float = 0.0) -> bytes:
    return struct.pack(f"<{steps * dimension}f", *[salt + i for i in range(steps * dimension)])


def reasons_of(doc: Any, point: str, **kwargs: Any) -> tuple[str, ...]:
    with pytest.raises(PayloadNotBuilt) as raised:
        build_payload(doc, point, contract_ref=REF, **kwargs)
    assert raised.value.code == "PAYLOAD_NOT_BUILT"
    assert raised.value.reasons == (
        *session_obstacles(doc),
        *position_obstacles(doc, point, **kwargs),
    )
    return raised.value.reasons


# ---- the sample contract --------------------------------------------------------------------


def test_schema_identity() -> None:
    assert (PAYLOAD_SCHEMA_ID, PAYLOAD_SCHEMA_VERSION) == ("nvs.hidden-trajectory", "0.1.0")


@pytest.mark.parametrize("point", POINTS)
def test_every_position_of_the_sample_gives_a_payload(point: str) -> None:
    doc = document()
    assert session_obstacles(doc) == ()
    assert position_obstacles(doc, point) == ()
    payload = build_payload(doc, point, contract_ref=REF)
    assert set(payload) == PAYLOAD_KEYS
    assert doc == SAMPLE["document"]
    json.dumps(payload)


def test_the_payload_of_one_position_in_full() -> None:
    doc = document()
    recorded = position(doc, "block_out:3")
    index = struct.pack("<6q", 0, 1, 2, 3, 4, 5)
    assert build_payload(doc, "block_out:3", contract_ref=REF) == {
        "source": "T-hid",
        "position": "L3.block_out",
        "point": "block_out:3",
        "kind": "block_out",
        "layer": 3,
        "length": 6,
        "dimension": 16,
        "points": {
            "uri": "blob:sha256:" + recorded["trajectory_id"],
            "sha256": recorded["trajectory_id"],
            "media_type": "application/vnd.sensos.ndarray-raw",
            "dtype": "float32",
            "shape": [6, 16],
            "byte_order": "little",
            "order": "C",
            "unit": "activation",
        },
        "token_index": {
            "uri": "blob:sha256:" + sha(index),
            "sha256": sha(index),
            "media_type": "application/vnd.sensos.ndarray-raw",
            "dtype": "int64",
            "shape": [6],
            "byte_order": "little",
            "order": "C",
            "quantization": None,
            "unit": "token_position",
        },
        "raw_ref": {"trajectory_id": recorded["trajectory_id"], "copied": False},
        "row_rule": "synthetic-llama-block-out/1",
        "binding": {
            "alignment": "decides-next/1",
            "n_in": 5,
            "n_gen": 6,
            "finish_reason": "length",
        },
        "final_norm_applied": False,
        "state_dtype": "bfloat16",
        "model_build_id": "synthetic-test-1",
        "compute_environment": "synthetic-env",
        "policy": {"name": "capture-all-rows/1", "sha256": sha(b"capture-all-rows/1")},
        "capture_provenance": {
            "weights_sha256": "d" * 64,
            "tokenizer_sha256": "a" * 64,
            "chat_template_sha256": "not_applicable",
            "prompt_ids_sha256": "b" * 64,
            "gen_token_ids_sha256": "c" * 64,
            "capture": {"package": "capture-pkg", "version": "1", "adapter": "adapter-1"},
            "runtime": {"backend": "mlx", "mlx": "0.1.0", "mlx_lm": "0.2.0", "device": "gpu"},
            "model_source": None,
        },
        "contract_ref": REF,
        "unknowns": [],
    }


# ---- T-U6-E02: point -> position, kind, layer -----------------------------------------------


@pytest.mark.parametrize(
    ("point", "legacy", "kind"),
    [
        ("block_out:15", "L15.block_out", "block_out"),
        ("mlp_out:0", "L0.mlp_out", "mlp_out"),
        ("final_norm", "final_norm", "final_norm"),
    ],
)
def test_e02_names_of_a_point(point: str, legacy: str, kind: str) -> None:
    assert (legacy_position(point), kind_of(point)) == (legacy, kind)


@pytest.mark.parametrize(
    "name", ["L15.block_out", "block_out:015", "block_out", "final_norm:1", "embed:0", "", 15, None]
)
def test_e02_a_name_that_is_not_a_point_is_refused(name: Any) -> None:
    with pytest.raises(ValueError, match="canonical point name"):
        legacy_position(name)
    with pytest.raises(ValueError, match="canonical point name"):
        kind_of(name)


def test_e02_layer_keeps_its_type() -> None:
    doc = document()
    block = build_payload(doc, "block_out:7", contract_ref=REF)
    mlp = build_payload(doc, "mlp_out:7", contract_ref=REF)
    final = build_payload(doc, "final_norm", contract_ref=REF)
    assert (block["position"], block["kind"], block["layer"]) == ("L7.block_out", "block_out", 7)
    assert (mlp["position"], mlp["kind"], mlp["layer"]) == ("L7.mlp_out", "mlp_out", 7)
    assert type(block["layer"]) is int
    assert (final["position"], final["kind"], final["layer"]) == (
        "final_norm",
        "final_norm",
        "final_norm",
    )
    assert final["final_norm_applied"] is True and mlp["final_norm_applied"] is False


# ---- T-U6-E03: points and token_index -------------------------------------------------------


@pytest.mark.parametrize("point", POINTS)
def test_e03_points_and_token_index(point: str) -> None:
    payload = build_payload(document(), point, contract_ref=REF)
    points, index = payload["points"], payload["token_index"]
    assert points["sha256"] == payload["raw_ref"]["trajectory_id"]
    assert points["uri"] == blob_uri(points["sha256"])
    assert points["shape"] == [payload["length"], payload["dimension"]]
    assert index["shape"] == [payload["length"]]
    data = token_index_bytes(payload["length"])
    assert index["sha256"] == sha(data) and index["uri"] == blob_uri(sha(data))
    assert list(struct.unpack(f"<{payload['length']}q", data)) == list(range(payload["length"]))
    assert payload["length"] == payload["binding"]["n_gen"]


def test_e03_token_index_bytes() -> None:
    assert token_index_bytes(0) == b""
    assert token_index_bytes(3) == struct.pack("<3q", 0, 1, 2)
    for bad in (-1, True, 1.0, "3"):
        with pytest.raises(ValueError, match="integer >= 0"):
            token_index_bytes(bad)  # type: ignore[arg-type]


def test_e03_copied_is_written_as_given() -> None:
    payload = build_payload(document(), "final_norm", contract_ref=REF, copied=True)
    assert payload["raw_ref"]["copied"] is True


# ---- T-U6-E04, E05, E07 (payload side), E08 -------------------------------------------------


def test_e04_policy_sha256_is_of_its_name() -> None:
    doc = document()
    payload = build_payload(doc, "final_norm", contract_ref=REF)
    assert payload["policy"] == {"name": POLICY_ALL_ROWS, "sha256": sha(POLICY_ALL_ROWS.encode())}
    assert payload["policy"]["sha256"] != sha(payload["row_rule"].encode())
    named = build_payload(doc, "final_norm", contract_ref=REF, policy_name="rows-0+128/1")
    assert named["policy"] == {"name": "rows-0+128/1", "sha256": sha(b"rows-0+128/1")}


def test_e05_capture_has_three_keys() -> None:
    doc = document()
    assert set(doc["origin"]["capture"]) == {"package", "version", "adapter", "alignment"}
    payload = build_payload(doc, "final_norm", contract_ref=REF)
    assert set(payload["capture_provenance"]["capture"]) == {"package", "version", "adapter"}
    assert payload["capture_provenance"]["model_source"] is None


def test_e07_contract_ref_is_written_as_given() -> None:
    doc = document()
    assert build_payload(doc, "final_norm", contract_ref=None)["contract_ref"] is None
    ref = dict(REF)
    payload = build_payload(doc, "final_norm", contract_ref=ref)
    assert payload["contract_ref"] == REF
    payload["contract_ref"]["version"] = 3
    assert ref == REF


def test_e08_no_draft_fields() -> None:
    payload = build_payload(document(), "final_norm", contract_ref=REF)
    for absent in ("native_ids", "dimensions", "space_id", "session_id"):
        assert absent not in payload
    assert "alignment" not in payload["capture_provenance"]["capture"]


def test_the_payload_shares_nothing_with_the_document() -> None:
    doc = document()
    payload = build_payload(doc, "final_norm", contract_ref=REF)
    payload["binding"]["n_in"] = 99
    payload["capture_provenance"]["runtime"]["backend"] = "changed"
    payload["capture_provenance"]["capture"]["package"] = "changed"
    assert doc == SAMPLE["document"]


# ---- T-U6-E06: unknowns ---------------------------------------------------------------------


def test_e06_unknowns_list_exactly_the_unknown_places() -> None:
    doc = document()
    doc["origin"]["weights_sha256"] = "unknown"
    doc["raw"]["state_dtype"] = "unknown"
    doc["binding"]["tokenizer_sha256"] = "unknown"
    doc["origin"]["runtime"]["mlx_lm"] = "unknown"
    doc["origin"]["capture"]["version"] = "unknown"
    doc["origin"]["hardware"] = "unknown"  # not a field of the payload
    doc["unknowns"] = [
        {"field": "binding.tokenizer_sha256", "source": "t", "reason": "failed"},
        {"field": "origin.capture.version", "source": "t", "reason": "unavailable"},
        {"field": "origin.hardware", "source": "t", "reason": "not_recorded"},
        {"field": "origin.runtime.mlx_lm", "source": "t", "reason": "not_recorded"},
        {"field": "origin.weights_sha256", "source": "t", "reason": "not_recorded"},
        # raw.state_dtype is left out on purpose: no entry in the contract -> reason "unknown"
    ]
    payload = build_payload(doc, "final_norm", contract_ref=REF)
    assert payload["unknowns"] == [
        {"field": "/capture_provenance/capture/version", "reason": "unavailable"},
        {"field": "/capture_provenance/runtime/mlx_lm", "reason": "not_recorded"},
        {"field": "/capture_provenance/tokenizer_sha256", "reason": "failed"},
        {"field": "/capture_provenance/weights_sha256", "reason": "not_recorded"},
        {"field": "/state_dtype", "reason": "unknown"},
    ]
    fields = [entry["field"] for entry in payload["unknowns"]]
    assert fields == sorted(fields, key=lambda field: field.encode("utf-8"))
    assert _unknown_places(payload) == sorted(fields)


def test_e06_not_applicable_is_not_listed() -> None:
    payload = build_payload(document(), "final_norm", contract_ref=REF)
    assert payload["capture_provenance"]["chat_template_sha256"] == "not_applicable"
    assert payload["unknowns"] == []


def test_e06_a_reason_the_payload_cannot_say_becomes_unknown() -> None:
    doc = document()
    doc["origin"]["weights_sha256"] = "unknown"
    doc["unknowns"] = [
        {"field": "origin.weights_sha256", "source": "t", "reason": "lost"},
        "not an entry",
    ]
    payload = build_payload(doc, "final_norm", contract_ref=REF)
    assert payload["unknowns"] == [
        {"field": "/capture_provenance/weights_sha256", "reason": "unknown"}
    ]
    doc["unknowns"] = "not a list"
    assert build_payload(doc, "final_norm", contract_ref=REF)["unknowns"] == payload["unknowns"]


def test_e06_a_runtime_key_is_escaped_as_a_json_pointer() -> None:
    doc = document()
    doc["origin"]["runtime"] = {"backend": "mlx", "a/b~c": "unknown"}
    payload = build_payload(doc, "final_norm", contract_ref=REF)
    assert payload["unknowns"] == [
        {"field": "/capture_provenance/runtime/a~1b~0c", "reason": "unknown"}
    ]


def _unknown_places(node: Any, path: str = "") -> list[str]:
    """Every place of a payload (but ``unknowns``) whose value is "unknown", as the Gate reads."""
    if node == "unknown" and isinstance(node, str):
        return [path]
    if isinstance(node, dict):
        return sorted(
            place
            for key, value in node.items()
            if not (path == "" and key == "unknowns")
            for place in _unknown_places(value, f"{path}/{key}")
        )
    return []


# ---- T-U6-R04 (payload side): what no position can be written with --------------------------


@pytest.mark.parametrize(
    ("section", "key", "value", "reason"),
    [
        ("binding", "n_in", "unknown", "binding.n_in"),
        ("binding", "n_in", 0, "binding.n_in"),
        ("binding", "n_in", True, "binding.n_in"),
        ("binding", "n_gen", "unknown", "binding.n_gen"),
        ("binding", "n_gen", 2**53, "binding.n_gen"),
        ("binding", "finish_reason", "unknown", "binding.finish_reason"),
        ("binding", "finish_reason", "stop", "binding.finish_reason"),
        ("binding", "alignment", "other/1", "binding.alignment"),
        ("binding", "tokenizer_sha256", None, "binding.tokenizer_sha256"),
        ("origin", "model_build_id", "unknown", "origin.model_build_id"),
        ("origin", "compute_environment", "unknown", "origin.compute_environment"),
        ("origin", "compute_environment", "", "origin.compute_environment"),
        ("origin", "weights_sha256", 5, "origin.weights_sha256"),
        ("origin", "capture", "unknown", "origin.capture"),
        ("origin", "runtime", "unknown", "origin.runtime"),
        ("origin", "runtime", {"mlx": "0.1.0"}, "origin.runtime"),
        ("origin", "runtime", {"backend": "mlx", "threads": 8}, "origin.runtime.threads"),
        ("raw", "state_dtype", None, "raw.state_dtype"),
    ],
)
def test_r04_values_no_payload_can_hold(section: str, key: str, value: Any, reason: str) -> None:
    doc = document()
    doc[section][key] = value
    expected = (f"payload_cannot_express:{reason}",)
    assert session_obstacles(doc) == expected
    assert reasons_of(doc, "final_norm") == expected


def test_r04_capture_entries_must_be_text() -> None:
    doc = document()
    doc["origin"]["capture"] = {"package": "p", "version": 1, "alignment": "a"}
    assert session_obstacles(doc) == (
        "payload_cannot_express:origin.capture.version",
        "payload_cannot_express:origin.capture.adapter",
    )


def test_r04_every_reason_is_reported_at_once() -> None:
    doc = document()
    doc["binding"]["n_in"] = "unknown"
    doc["binding"]["finish_reason"] = "unknown"
    doc["origin"]["runtime"] = "unknown"
    position(doc, "block_out:3")["final_norm_applied"] = "unknown"
    assert reasons_of(doc, "block_out:3") == (
        "payload_cannot_express:binding.n_in",
        "payload_cannot_express:binding.finish_reason",
        "payload_cannot_express:origin.runtime",
        "payload_cannot_express:final_norm_applied",
    )


@pytest.mark.parametrize("not_a_contract", [None, [], "text", {}, {"binding": 3, "origin": []}])
def test_r04_a_document_that_is_not_a_contract_gives_reasons_not_errors(
    not_a_contract: Any,
) -> None:
    assert len(session_obstacles(not_a_contract)) == 14
    assert position_obstacles(not_a_contract, "final_norm") == ("point_not_in_contract",)
    assert reasons_of(not_a_contract, "final_norm")[-1] == "point_not_in_contract"


# ---- T-U6-R06, R07 (payload side): one position ---------------------------------------------


def test_r06_unknown_final_norm_applied_stops_that_position_only() -> None:
    doc = document()
    position(doc, "block_out:3")["final_norm_applied"] = "unknown"
    assert reasons_of(doc, "block_out:3") == ("payload_cannot_express:final_norm_applied",)
    assert position_obstacles(doc, "block_out:7") == ()
    assert build_payload(doc, "block_out:7", contract_ref=REF)["final_norm_applied"] is False


def test_r07_unknown_trajectory_id_without_rows() -> None:
    doc = document()
    position(doc, "final_norm")["trajectory_id"] = "unknown"
    assert reasons_of(doc, "final_norm") == ("trajectory_id_unknown",)


def test_r07_unknown_trajectory_id_and_steps_are_taken_from_the_rows() -> None:
    doc = document()
    data = rows()
    position(doc, "final_norm")["trajectory_id"] = "unknown"
    position(doc, "final_norm")["stored_range"]["steps"] = "unknown"
    assert reasons_of(doc, "final_norm") == ("trajectory_id_unknown", "length_unknown")
    payload = build_payload(doc, "final_norm", contract_ref=REF, rows_bytes=data)
    assert payload["raw_ref"]["trajectory_id"] == sha(data) == payload["points"]["sha256"]
    assert payload["length"] == N_GEN and payload["points"]["shape"] == [N_GEN, DIMENSION]


def test_r07_rows_that_are_not_the_contracts_are_not_used() -> None:
    doc = document()
    assert reasons_of(doc, "final_norm", rows_bytes=rows(salt=1.0)) == ("rows_bytes:trajectory_id",)
    assert reasons_of(doc, "final_norm", rows_bytes=rows(steps=N_GEN - 1)) == (
        "rows_bytes:trajectory_id",
        "rows_bytes:steps",
    )
    for broken in (b"", rows()[:-1]):
        assert reasons_of(doc, "final_norm", rows_bytes=broken) == ("rows_bytes:length",)


def test_r07_rows_that_agree_with_the_contract_change_nothing() -> None:
    doc = document()
    data = rows()
    position(doc, "final_norm")["trajectory_id"] = sha(data)
    without = build_payload(doc, "final_norm", contract_ref=REF)
    assert build_payload(doc, "final_norm", contract_ref=REF, rows_bytes=data) == without


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"dimension": 0}, "payload_cannot_express:dimension"),
        ({"dimension": 65536}, "payload_cannot_express:dimension"),
        ({"dimension": 16.0}, "payload_cannot_express:dimension"),
        ({"row_rule": None}, "payload_cannot_express:row_rule"),
        ({"space": {"layer": "15"}}, "payload_cannot_express:space.layer"),
        ({"space": {"layer": True}}, "payload_cannot_express:space.layer"),
        ({"space": "unknown"}, "payload_cannot_express:space.layer"),
        ({"trajectory_id": "ABC"}, "payload_cannot_express:trajectory_id"),
        ({"stored_range": "unknown"}, "payload_cannot_express:stored_range.steps"),
        ({"stored_range": {"steps": 0}}, "payload_cannot_express:stored_range.steps"),
    ],
)
def test_position_values_the_payload_cannot_hold(change: dict[str, Any], reason: str) -> None:
    doc = document()
    position(doc, "final_norm").update(change)
    assert reasons_of(doc, "final_norm") == (reason,)
    assert reasons_of(doc, "final_norm", rows_bytes=rows())[0] == reason


def test_a_point_that_is_not_in_the_contract() -> None:
    doc = document()
    assert reasons_of(doc, "block_out:99") == ("point_not_in_contract",)
    assert position_obstacles(doc, 3) == ("point_not_in_contract",)  # type: ignore[arg-type]
    doc["positions"] = "not a list"
    assert position_obstacles(doc, "final_norm") == ("point_not_in_contract",)


def test_a_position_whose_point_is_not_a_canonical_name() -> None:
    doc = document()
    position(doc, "final_norm")["point"] = "L3.block_out"
    assert reasons_of(doc, "L3.block_out") == ("payload_cannot_express:point",)


# ---- T-U6-E09: imports ----------------------------------------------------------------------


def test_e09_payload_module_imports_stdlib_and_core_errors_only() -> None:
    """Standard library and nvs_runtime.core.errors only: no nvs_edge, no sensos contracts."""
    tree = ast.parse(Path(payload_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported == {
        "__future__",
        "collections.abc",
        "hashlib",
        "re",
        "struct",
        "typing",
        "nvs_runtime.core.errors",
    }
