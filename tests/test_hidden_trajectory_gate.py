"""The join: contract v2 -> contract_facts -> envelopes -> the real sensos Gate
(design U6 v0.1, section 8; tests T-U6-G01..G07).

Every test here needs three things that a plain CI runner does not have, and is skipped without
any of them (the reason says which):

* ``SENSOS_TOOLS`` -- the ``tools/`` directory of a sensos checkout with ``contracts/gate_nvs.py``;
* ``GATE_PY`` -- an interpreter with ``jsonschema``, in which the Gate runs;
* nvs-kernel's ``nvs_edge`` with contract v2, and numpy (to make RAW files).

The contracts use the registered row rules (Qwen3.5, Llama), so the Gate's own registry is used
as it is: nothing is added to it.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from nvs_observation.contract_check import (
    VERIFIED,
    ContractCheck,
    check_contract_v2,
    edge_contract_available,
    rows_from_raw,
)

from nvs_runtime.hidden_trajectory import (
    BuildResult,
    EnvelopeIds,
    build_envelope,
    build_envelopes,
    build_payload,
    contract_ref_of,
    token_index_bytes,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "hidden_trajectory"
FIXTURES = Path(__file__).parent / "fixtures" / "hidden_trajectory"
SAMPLE = json.loads((FIXTURES / "contract-v2-u4-base.json").read_text(encoding="utf-8"))


def _skip_reason() -> str | None:
    missing = [name for name in ("SENSOS_TOOLS", "GATE_PY") if not os.environ.get(name)]
    if missing:
        return f"environment variables not set: {', '.join(missing)} (examples/hidden_trajectory)"
    tools = Path(os.environ["SENSOS_TOOLS"])
    if not (tools / "contracts" / "gate_nvs.py").is_file():
        return f"SENSOS_TOOLS has no contracts/gate_nvs.py (sensos without U5): {tools}"
    if not Path(os.environ["GATE_PY"]).is_file():
        return f"GATE_PY is not a file: {os.environ['GATE_PY']}"
    if not edge_contract_available():
        return "nvs_edge contract v2 is not installed"
    if importlib.util.find_spec("numpy") is None:
        return "numpy is not installed (needed to make RAW files)"
    return None


_REASON = _skip_reason()
pytestmark = pytest.mark.skipif(_REASON is not None, reason=_REASON or "")

PRODUCED_AT = "2026-11-15T00:00:00Z"
DIMENSION = 16
QWEN = {
    "block_out": "qwen35-mlx-block-out/1",
    "mlp_out": "qwen35-mlx-mlp-out/1",
    "final_norm": "qwen35-mlx-final-norm/1",
}
POINTS = ["block_out:3", "block_out:7", "mlp_out:7", "final_norm"]
WINDOW = 128


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def door() -> Any:
    sys.path.insert(0, str(EXAMPLE))
    try:
        from sensos_door import SensosDoor
    finally:
        sys.path.remove(str(EXAMPLE))
    return SensosDoor.from_environment()


class Session:
    """One capture: a contract v2 with registered row rules, its RAW files, ids and a counter."""

    def __init__(
        self, door: Any, *, item: str, n_gen: int = 6, rules: dict[str, str] | None = None
    ) -> None:
        import numpy as np
        from nvs_edge.raw_store import encode_raw

        self.door = door
        rules = QWEN if rules is None else rules
        doc = copy.deepcopy(SAMPLE["document"])
        doc["session_id"] = f"u6-join-{item}"
        doc["binding"]["n_gen"] = n_gen
        doc["positions"] = [
            p for p in doc["positions"] if p["point"].partition(":")[0] in rules
        ]
        self.raw: dict[str, bytes] = {}
        for index, position in enumerate(doc["positions"]):
            rule = rules[position["point"].partition(":")[0]]
            position["row_rule"] = position["space"]["row_rule"] = rule
            position["stored_range"]["steps"] = n_gen
            states = (np.arange(n_gen * DIMENSION, dtype=np.float32) / 7 + index).reshape(
                n_gen, DIMENSION
            )
            self.raw[position["point"]], header = encode_raw(states, dimension=DIMENSION)
            position["trajectory_id"] = header.trajectory_id
        self.document = doc
        self.session_id = door.session_id(
            spec=sha(b"U6-JOIN"), item_id=item, condition="join", replicate=0, rng_seed=0
        )
        self.observation_id = door.observation_id(self.session_id, 0)
        self._seq = 0

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def ids(self) -> EnvelopeIds:
        return EnvelopeIds(self.session_id, self.observation_id, PRODUCED_AT, self.next_seq)

    def check(self, **kwargs: Any) -> ContractCheck:
        return check_contract_v2(self.document, raw_bytes=self.raw, **kwargs)

    def build(self, check: ContractCheck | None = None, *, with_rows: bool = True) -> BuildResult:
        return build_envelopes(
            self.document,
            self.check() if check is None else check,
            ids=self.ids,
            space_id_of=self.door.space_id_of,
            seal=self.door.seal,
            rows=rows_from_raw(self.raw) if with_rows else None,
        )


def verdicts(door: Any, result: BuildResult, **kwargs: Any) -> dict[str, dict[str, Any]]:
    return {
        point: door.check(envelope, blobs=result.blobs, **kwargs)
        for point, envelope in result.envelopes.items()
    }


def unverifiable(verdict: dict[str, Any]) -> list[tuple[str, str]]:
    return sorted((u["check"], u["reason"]) for u in verdict["payload"]["unverifiable"])


# ---- T-U6-G01: the four positions pass the real Gate ----------------------------------------


def test_g01_four_envelopes_pass_with_blobs_and_facts(door: Any) -> None:
    session = Session(door, item="g01")
    check = session.check()
    assert check.status == VERIFIED and check.facts is not None
    result = session.build(check)
    assert list(result.envelopes) == POINTS and result.record["outcome"] == "enveloped"
    for point, verdict in verdicts(door, result, contract_facts=check.facts).items():
        assert verdict["l1_codes"] == [], (point, verdict["l1_messages"])
        assert verdict["payload"] == {
            "kind": "hidden_trajectory",
            "status": "PASS",
            "violations": [],
            "unverifiable": [],
        }, point
        assert verdict["window"] is None


def test_g01_the_ids_are_those_of_the_sensos_rules(door: Any) -> None:
    session = Session(door, item="g01-ids")
    result = session.build()
    for position in session.document["positions"]:
        envelope = result.envelopes[position["point"]]
        assert envelope["space_id"] == door.space_id_of(position["space"])
        assert envelope["space_id"].startswith("hid:") and len(envelope["space_id"]) == 68
        resealed = door.seal(copy.deepcopy(envelope), {})
        assert (resealed["content_id"], resealed["artifact_id"]) == (
            envelope["content_id"],
            envelope["artifact_id"],
        )
    again = Session(door, item="g01-ids").build()
    assert {p: e["artifact_id"] for p, e in again.envelopes.items()} == {
        p: e["artifact_id"] for p, e in result.envelopes.items()
    }


# ---- T-U6-G02, G03: what the Gate was not given is unverifiable, not a failure --------------


def test_g02_without_blobs_the_bytes_are_unverifiable(door: Any) -> None:
    session = Session(door, item="g02")
    check = session.check()
    result = session.build(check, with_rows=False)
    for envelope in result.envelopes.values():
        verdict = door.check(envelope, contract_facts=check.facts)
        assert verdict["l1_codes"] == []
        assert verdict["payload"]["status"] == "PASS_UNVERIFIED"
        assert verdict["payload"]["violations"] == []
        assert unverifiable(verdict) == [
            ("points_bytes", "blob_not_supplied"),
            ("token_index_bytes", "blob_not_supplied"),
        ]


def test_g03_without_contract_facts_the_binding_is_unverifiable(door: Any) -> None:
    """What the Gate sees when nvs_edge is not there: no facts. Not a failure."""
    session = Session(door, item="g03")
    result = session.build()
    for verdict in verdicts(door, result, contract_facts=None).values():
        assert verdict["l1_codes"] == [] and verdict["payload"]["violations"] == []
        assert verdict["payload"]["status"] == "PASS_UNVERIFIED"
        assert unverifiable(verdict) == [("contract_binding", "contract_not_supplied")]


def test_g03_a_contract_with_unknown_values_is_unverifiable_in_those_fields(door: Any) -> None:
    session = Session(door, item="g03-unknown")
    doc = session.document
    doc["origin"]["weights_sha256"] = "unknown"
    doc["raw"]["state_dtype"] = "unknown"
    doc["binding"]["tokenizer_sha256"] = "unknown"
    doc["unknowns"] = [
        {"field": "binding.tokenizer_sha256", "source": "test", "reason": "failed"},
        {"field": "origin.weights_sha256", "source": "test", "reason": "not_recorded"},
        {"field": "raw.state_dtype", "source": "test", "reason": "unavailable"},
    ]
    check = session.check()
    assert check.status == VERIFIED
    result = session.build(check)
    assert list(result.envelopes) == POINTS
    for verdict in verdicts(door, result, contract_facts=check.facts).values():
        assert verdict["l1_codes"] == [] and verdict["payload"]["violations"] == []
        assert verdict["payload"]["status"] == "PASS_UNVERIFIED"
        assert unverifiable(verdict) == [
            ("contract_binding.tokenizer_sha256", "contract_field_unknown"),
            ("contract_binding.weights_sha256", "contract_field_unknown"),
        ]
    payload = result.envelopes["final_norm"]["payload"]
    assert payload["unknowns"] == [
        {"field": "/capture_provenance/tokenizer_sha256", "reason": "failed"},
        {"field": "/capture_provenance/weights_sha256", "reason": "not_recorded"},
        {"field": "/state_dtype", "reason": "unavailable"},
    ]


def test_g03_a_position_whose_trajectory_id_the_contract_does_not_know(door: Any) -> None:
    session = Session(door, item="g03-traj")
    doc = session.document
    doc["positions"][0]["trajectory_id"] = "unknown"
    doc["unknowns"] = [{"field": "positions[0].trajectory_id", "source": "t", "reason": "failed"}]
    check = session.check()
    result = session.build(check)
    assert list(result.envelopes) == POINTS
    verdict = door.check(
        result.envelopes[POINTS[0]], blobs=result.blobs, contract_facts=check.facts
    )
    assert verdict["payload"]["violations"] == []
    assert unverifiable(verdict) == [("contract_binding.trajectory_id", "contract_field_unknown")]


# ---- T-U6-G04: the Gate's own refusal of synthetic data -------------------------------------


def test_g04_a_synthetic_envelope_made_around_u6_fails_at_the_gate(door: Any) -> None:
    session = Session(door, item="g04")
    doc = session.document
    doc["origin"]["synthetic"] = True
    check = session.check()
    assert check.status == VERIFIED and check.facts is not None and check.facts["synthetic"] is True
    refused = session.build(check)
    assert refused.envelopes == {} and refused.record["refusals"] == ["synthetic"]
    # around build_envelopes, with the lower-level functions that do not decide anything
    point = "final_norm"
    payload = build_payload(doc, point, contract_ref=contract_ref_of(check))
    position = next(p for p in doc["positions"] if p["point"] == point)
    envelope = build_envelope(
        doc,
        payload,
        ids=session.ids,
        space_id=door.space_id_of(position["space"]),
        seal=door.seal,
    )
    verdict = door.check(envelope, contract_facts=check.facts)
    assert verdict["l1_codes"] == []
    assert verdict["payload"]["status"] == "FAIL"
    assert [v["code"] for v in verdict["payload"]["violations"]] == ["E_TRAJ_SYNTHETIC"]


# ---- T-U6-G05: every registered row rule ----------------------------------------------------


def test_g05_the_llama_final_norm_rule_passes_the_registry(door: Any) -> None:
    session = Session(door, item="g05", rules={"final_norm": "llama31-mlx-final-norm/1"})
    check = session.check()
    result = session.build(check)
    assert list(result.envelopes) == ["final_norm"]
    verdict = door.check(
        result.envelopes["final_norm"], blobs=result.blobs, contract_facts=check.facts
    )
    assert verdict["l1_codes"] == [] and verdict["payload"]["status"] == "PASS"
    assert result.envelopes["final_norm"]["payload"]["row_rule"] == "llama31-mlx-final-norm/1"


def test_g05_the_three_qwen_rules_pass_the_registry(door: Any) -> None:
    session = Session(door, item="g05-qwen")
    check = session.check()
    result = session.build(check)
    rules = {e["payload"]["row_rule"] for e in result.envelopes.values()}
    assert rules == set(QWEN.values())
    for verdict in verdicts(door, result, contract_facts=check.facts).values():
        assert verdict["payload"]["status"] == "PASS"


def test_g05_a_row_rule_that_is_not_registered_fails(door: Any) -> None:
    session = Session(
        door,
        item="g05-unregistered",
        rules={"final_norm": "synthetic-llama-final-norm/1"},
    )
    check = session.check()
    result = session.build(check)
    verdict = door.check(
        result.envelopes["final_norm"], blobs=result.blobs, contract_facts=check.facts
    )
    assert "E_TRAJ_ROW_RULE_UNKNOWN" in [v["code"] for v in verdict["payload"]["violations"]]


# ---- T-U6-G06: an envelope changed after it was built ---------------------------------------


def _changed(door: Any, envelope: dict[str, Any], change: Any) -> dict[str, Any]:
    altered = copy.deepcopy(envelope)
    change(altered)
    return door.seal(altered, {})


def _set_n_gen(envelope: dict[str, Any]) -> None:
    envelope["payload"]["binding"]["n_gen"] += 1


def _set_dimension(envelope: dict[str, Any]) -> None:
    envelope["payload"]["dimension"] += 1


def _set_contract_hash(envelope: dict[str, Any]) -> None:
    envelope["payload"]["contract_ref"]["contract_sha256"] = "0" * 64


def _set_space(envelope: dict[str, Any]) -> None:
    envelope["space_id"] = "hid:" + "0" * 64


def _set_row_rule(envelope: dict[str, Any]) -> None:
    envelope["payload"]["row_rule"] = QWEN["block_out"]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (_set_n_gen, "E_TRAJ_ROWS"),
        (_set_dimension, "E_TRAJ_SHAPE"),
        (_set_contract_hash, "E_TRAJ_CONTRACT"),
        (_set_space, "E_TRAJ_SPACE"),
        (_set_row_rule, "E_TRAJ_ROW_RULE_KIND"),
    ],
)
def test_g06_one_changed_field_is_caught(door: Any, change: Any, code: str) -> None:
    session = Session(door, item="g06")
    check = session.check()
    result = session.build(check)
    altered = _changed(door, result.envelopes["final_norm"], change)
    verdict = door.check(altered, blobs=result.blobs, contract_facts=check.facts)
    assert verdict["l1_codes"] == [], verdict["l1_messages"]
    assert verdict["payload"]["status"] == "FAIL"
    assert code in [v["code"] for v in verdict["payload"]["violations"]]


def test_g06_changed_rows_are_caught(door: Any) -> None:
    session = Session(door, item="g06-rows")
    check = session.check()
    result = session.build(check)
    envelope = result.envelopes["final_norm"]
    uri = envelope["payload"]["points"]["uri"]
    blobs = dict(result.blobs)
    blobs[uri] = bytes([blobs[uri][0] ^ 1]) + blobs[uri][1:]
    verdict = door.check(envelope, blobs=blobs, contract_facts=check.facts)
    assert verdict["payload"]["status"] == "FAIL"


def test_g06_an_envelope_that_fails_the_envelope_rule_is_not_looked_at_further(door: Any) -> None:
    session = Session(door, item="g06-l1")
    result = session.build()
    envelope = copy.deepcopy(result.envelopes["final_norm"])
    envelope["payload"]["binding"]["n_gen"] += 1  # not sealed again: the content id is now wrong
    verdict = door.check(envelope, blobs=result.blobs)
    assert verdict["l1_codes"] != []
    assert verdict["payload"] == "not_run" and verdict["window"] is None


# ---- T-U6-G07: facts of the wrong shape -----------------------------------------------------


def test_g07_malformed_contract_facts_are_an_error_of_the_caller(door: Any) -> None:
    session = Session(door, item="g07")
    check = session.check()
    result = session.build(check)
    envelope = result.envelopes["final_norm"]
    assert check.facts is not None
    for damage in ("drop", "extra", "version"):
        facts = copy.deepcopy(check.facts)
        if damage == "drop":
            del facts["synthetic"]
        elif damage == "extra":
            facts["positions"]["final_norm"]["space"]["kind"] = "final_norm"
        else:
            facts["contract"]["version"] = 3
        with pytest.raises(Exception) as raised:  # noqa: PT011 -- GateInternalError of the door
            door.check(envelope, contract_facts=facts)
        assert type(raised.value).__name__ == "GateInternalError"
        assert raised.value.code == "gate_internal_error"  # type: ignore[attr-defined]


def test_g07_an_unknown_op_is_an_error_too(door: Any) -> None:
    with pytest.raises(Exception) as raised:  # noqa: PT011
        door.call({"op": "store"})
    assert type(raised.value).__name__ == "GateInternalError"


# ---- a window of a hidden trajectory: check_window_space too --------------------------------


def _window(door: Any, session: Session, parent: dict[str, Any], rows: bytes, start: int) -> Any:
    """A 128-row window of ``parent`` as its own artifact (IC-02 0.10.0, rule ``rows``), written
    here as ``examples/pb_path/stages.py`` writes it; nvs-runtime has no window builder in src."""
    fp = parent["payload"]
    width = 4 * fp["dimension"]
    points = rows[start * width : (start + WINDOW) * width]
    index = token_index_bytes(start + WINDOW)[8 * start :]
    policy = f"rows-{start}+{WINDOW}/1"
    media = "application/vnd.sensos.ndarray-raw"
    blob = {"media_type": media, "byte_order": "little", "order": "C"}
    payload = {
        "window": {"rule": "rows", "start": start, "length": WINDOW, "content_sha256": sha(points)},
        "origin": {
            "trajectory_id": fp["raw_ref"]["trajectory_id"],
            "position": fp["position"],
            "length": fp["length"],
        },
        "source": "T-hid",
        "layer": fp["layer"],
        "dimension": fp["dimension"],
        "model_build_id": fp["model_build_id"],
        "policy": {"name": policy, "sha256": sha(policy.encode())},
        "points": {
            **blob,
            "uri": "blob:sha256:" + sha(points),
            "sha256": sha(points),
            "dtype": "float32",
            "shape": [WINDOW, fp["dimension"]],
            "unit": "activation",
        },
        "token_index": {
            **blob,
            "uri": "blob:sha256:" + sha(index),
            "sha256": sha(index),
            "dtype": "int64",
            "shape": [WINDOW],
            "quantization": None,
            "unit": "token_position",
        },
    }
    materials = {
        "model_build_id": fp["model_build_id"],
        "layer": fp["layer"],
        "row_rule": f"{fp['row_rule']}+rows{start}+{WINDOW}/1",
        "compute_environment": fp["compute_environment"],
        "dimension": fp["dimension"],
        "space_rule": "nvs.hidden-trajectory.space/1",
    }
    lookup = {
        parent["artifact_id"]: {
            "content_id": parent["content_id"],
            "seq": parent["seq"],
            "observation_id": parent["observation_id"],
        }
    }
    envelope = {
        "contract": {"id": "IC-00", "version": "1.1.0"},
        "content_id": "cnt:" + "0" * 64,
        "artifact_id": "art:" + "0" * 64,
        "kind": "observation",
        "payload_schema": {"id": "msr.trajectory-window", "version": "0.10.0"},
        "payload": payload,
        "producer": {"name": "nvs-runtime", "version": "0", "build": "u6-join-test"},
        "session_id": session.session_id,
        "observation_id": session.observation_id,
        "event_span": None,
        "space_id": door.space_id_of(materials),
        "parents": [
            {"role": "source_trajectory", "ordered": False, "artifact_ids": [parent["artifact_id"]]}
        ],
        "policy_refs": [],
        "calibration_ref": None,
        "rule_refs": [],
        "observed_at": None,
        "seq": session.next_seq(),
        "produced_at": PRODUCED_AT,
        "extensions": {},
    }
    sealed = door.seal(envelope, lookup)
    blobs = {payload["points"]["uri"]: points, payload["token_index"]["uri"]: index}
    return sealed, lookup, blobs


@pytest.mark.parametrize("start", [0, 64])
def test_a_window_of_a_u6_envelope_passes_ic02_and_the_window_space_check(
    door: Any, start: int
) -> None:
    session = Session(door, item=f"window-{start}", n_gen=WINDOW + 64)
    check = session.check()
    result = session.build(check)
    parent = result.envelopes["block_out:3"]
    parent_verdict = door.check(parent, blobs=result.blobs, contract_facts=check.facts)
    assert parent_verdict["payload"]["status"] == "PASS"
    rows = rows_from_raw(session.raw)["block_out:3"]
    window, lookup, blobs = _window(door, session, parent, rows, start)
    stored = {parent["artifact_id"]: {"envelope": parent}}
    verdict = door.check(window, lookup=lookup, stored=stored, blobs=blobs)
    assert verdict["l1_codes"] == [], verdict["l1_messages"]
    assert verdict["payload"]["kind"] == "ic02"
    assert verdict["payload"]["violations"] == [] and verdict["payload"]["status"] == "PASS"
    assert verdict["window"] == {
        "kind": "window_space",
        "status": "PASS",
        "violations": [],
        "unverifiable": [],
    }


def test_a_window_whose_space_is_the_parents_fails_the_window_space_check(door: Any) -> None:
    session = Session(door, item="window-space", n_gen=WINDOW + 64)
    result = session.build()
    parent = result.envelopes["block_out:3"]
    rows = rows_from_raw(session.raw)["block_out:3"]
    window, lookup, blobs = _window(door, session, parent, rows, 0)
    window["space_id"] = parent["space_id"]
    window = door.seal(window, lookup)
    stored = {parent["artifact_id"]: {"envelope": parent}}
    verdict = door.check(window, lookup=lookup, stored=stored, blobs=blobs)
    assert verdict["l1_codes"] == [], verdict["l1_messages"]
    assert verdict["window"]["status"] == "FAIL"
    assert [v["code"] for v in verdict["window"]["violations"]] == ["E_WIN_SPACE"]
