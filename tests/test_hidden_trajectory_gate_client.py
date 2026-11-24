"""``GateClient``: the formal Gate that builds a ``GateVerdict`` (design U6T-1 v0.1; T-T1-02..13).

Two kinds of test:

* **with a fake child** (``FakeProcess``): the layers, the ``result`` rule, the catalogue, the
  lookup, the errors. No sensos, no interpreter -- they run everywhere;
* **with the real child** (``GATE_PY`` and ``SENSOS_TOOLS``, as ``test_hidden_trajectory_gate.py``):
  the verdict of real envelopes, its determinism, the bundle, the schema of sensos. Skipped,
  with the reason, without them. The test of the schema needs ``SENSOS_TOOLS`` of a sensos with
  ``nvs/gate-verdict.schema.json``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_hidden_trajectory_gate import (
    EXAMPLE,
    POINTS,
    WINDOW,
    Session,
    _changed,
    _set_n_gen,
    _set_row_rule,
    _skip_reason,
    _window,
    sha,
)

from nvs_runtime.hidden_trajectory import gate_client as gc
from nvs_runtime.hidden_trajectory.gate_client import (
    CHILD_SCRIPT,
    CatalogMismatch,
    GateClient,
    GateInternalError,
    GateProcess,
    GateUnavailable,
    LookupMissing,
    missing_lookup,
    required_lookup_ids,
)

REAL_REASON = _skip_reason()
needs_real = pytest.mark.skipif(REAL_REASON is not None, reason=REAL_REASON or "")

ART = "art:" + "1" * 64
PARENT = "art:" + "2" * 64
CALIBRATION = "art:" + "3" * 64
CNT = "cnt:" + "4" * 64
OBS = "obs:" + "5" * 64
HEX = "6" * 64

CATALOG_ROWS = [
    ["msr.trajectory-window", "window_rows", "raw_not_in_artifact"],
    ["msr.trajectory-window", "window_space_derivation", "rule_not_derivable"],
    ["nvs.hidden-trajectory", "contract_binding", "contract_absent"],
    ["nvs.hidden-trajectory", "contract_binding", "contract_not_supplied"],
    ["nvs.hidden-trajectory", "contract_binding.<field>", "contract_field_unknown"],
    ["nvs.hidden-trajectory", "points_bytes", "blob_not_supplied"],
    ["nvs.hidden-trajectory", "token_index_bytes", "blob_not_supplied"],
]


# ---- a fake child ---------------------------------------------------------------------------


def _passing_answer() -> dict[str, Any]:
    return {
        "l1": [],
        "payload": {
            "kind": "hidden_trajectory",
            "status": "PASS",
            "violations": [],
            "unverifiable": [],
        },
        "window": None,
        "envelope_sha256": HEX,
        "lookup_sha256": HEX,
        "contract_facts_sha256": None,
    }


class FakeProcess:
    """What ``GateProcess`` is to ``GateClient``: ``call(request)``. ``answer`` is the child's
    answer to ``gate``."""

    def __init__(self, tmp: Path, answer: dict[str, Any] | None = None) -> None:
        self.gate_py = Path("/nonexistent/python")
        self.sensos_tools = tmp / "tools"
        self.answer = _passing_answer() if answer is None else answer
        self.calls: list[str] = []

    def call(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request["op"]
        self.calls.append(op)
        if op == "checker_files":
            bundle = b"BUNDLE:" + b"".join(
                sorted(f["b64"].encode() for f in request["extra_files"])
            )
            return {
                "bundle_b64": base64.b64encode(bundle).decode(),
                "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
                "files": [{"repo": "sensos", "path": "tools/contracts/gate.py", "sha256": HEX}],
                "runtime": {
                    "python": "3.14.7",
                    "implementation": "CPython",
                    "packages": {"jsonschema": "4.26.0", "referencing": "0.37.0"},
                    "platform": {"system": "Darwin", "machine": "arm64"},
                },
            }
        if op == "catalog":
            return {"rows": CATALOG_ROWS, "sha256": HEX}
        if op == "gate":
            return copy.deepcopy(self.answer)
        if op == "jcs_sha256":
            text = json.dumps(request["value"], sort_keys=True).encode()
            return {"sha256": hashlib.sha256(text).hexdigest()}
        raise AssertionError(op)


def envelope_of(**changes: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "contract": {"id": "IC-00", "version": "1.1.0"},
        "artifact_id": ART,
        "content_id": CNT,
        "payload_schema": {"id": "nvs.hidden-trajectory", "version": "0.1.0"},
        "parents": [],
        "calibration_ref": None,
    }
    envelope.update(changes)
    return envelope


def fake_client(tmp_path: Path, answer: dict[str, Any] | None = None) -> GateClient:
    client = GateClient(Path("/nonexistent/python"), tmp_path / "tools")
    client.process = FakeProcess(tmp_path, answer)  # type: ignore[assignment]
    return client


def with_unverifiable(
    rows: list[dict[str, str]], status: str = "PASS_UNVERIFIED"
) -> dict[str, Any]:
    answer = _passing_answer()
    answer["payload"]["status"] = status
    answer["payload"]["unverifiable"] = rows
    return answer


# ---- the lookup (DC-T1-3) -------------------------------------------------------------------


def test_the_ids_a_lookup_must_have_are_the_parents_and_the_calibration() -> None:
    envelope = envelope_of(
        parents=[
            {"role": "generated_text", "ordered": False, "artifact_ids": [PARENT, ART]},
            {"role": "x", "ordered": False, "artifact_ids": [PARENT]},
        ],
        calibration_ref={"artifact_id": CALIBRATION, "content_id": CNT},
    )
    assert required_lookup_ids(envelope) == tuple(sorted({PARENT, ART, CALIBRATION}))
    assert required_lookup_ids(envelope_of()) == ()
    assert required_lookup_ids({}) == () and required_lookup_ids({"parents": "x"}) == ()
    assert missing_lookup(envelope, {PARENT: {}}) == tuple(sorted({ART, CALIBRATION}))
    assert missing_lookup(envelope, None) == required_lookup_ids(envelope)


def test_t_t1_02_an_envelope_with_parents_and_no_lookup_is_stopped_before_the_gate(
    tmp_path: Path,
) -> None:
    client = fake_client(tmp_path)
    envelope = envelope_of(
        parents=[{"role": "generated_text", "ordered": False, "artifact_ids": [PARENT]}]
    )
    for lookup in (None, {}, {ART: {}}):
        with pytest.raises(LookupMissing) as raised:
            client.verdict(envelope, lookup=lookup)
        assert raised.value.missing == (PARENT,)
    assert client.process.calls == []  # type: ignore[attr-defined]  # nothing was run
    assert issubclass(LookupMissing, gc.GateError) and not issubclass(
        LookupMissing, GateUnavailable
    )


# ---- the layers and the result (T-T1-05, 06) ------------------------------------------------


def test_t_t1_05_pass(tmp_path: Path) -> None:
    verdict = fake_client(tmp_path).verdict(envelope_of())
    assert verdict["result"] == "PASS" and verdict["codes"] == []
    assert verdict["layers"] == {
        "L1": {"status": "PASS", "codes": []},
        "L2": {"status": "PASS", "kind": "hidden_trajectory", "codes": []},
        "L3": {"status": "not_applicable", "codes": []},
    }
    assert verdict["waived"] == [] and verdict["verdict_version"] == "gate-verdict/1"


def test_t_t1_05_pass_unverified_when_only_something_is_unverifiable(tmp_path: Path) -> None:
    rows = [
        {"check": "points_bytes", "reason": "blob_not_supplied"},
        {"check": "contract_binding", "reason": "contract_not_supplied"},
    ]
    verdict = fake_client(tmp_path, with_unverifiable(rows)).verdict(envelope_of())
    assert verdict["result"] == "PASS_UNVERIFIED" and verdict["violations"] == []
    assert verdict["layers"]["L2"]["status"] == "PASS_UNVERIFIED"
    assert verdict["unverifiable"] == [dict(r, layer="L2") for r in rows]


def test_t_t1_05_fail_wins_over_unverifiable(tmp_path: Path) -> None:
    answer = with_unverifiable([{"check": "points_bytes", "reason": "blob_not_supplied"}], "FAIL")
    answer["payload"]["violations"] = [
        {"code": "E_TRAJ_ROWS", "message": "m1"},
        {"code": "E_TRAJ_CONTRACT", "message": "m2"},
        {"code": "E_TRAJ_ROWS", "message": "m3"},
    ]
    verdict = fake_client(tmp_path, answer).verdict(envelope_of())
    assert verdict["result"] == "FAIL"
    assert verdict["codes"] == ["E_TRAJ_CONTRACT", "E_TRAJ_ROWS"]  # once each, in order
    assert verdict["layers"]["L2"]["codes"] == ["E_TRAJ_CONTRACT", "E_TRAJ_ROWS"]
    assert [v["message"] for v in verdict["violations"]] == ["m1", "m2", "m3"]
    assert {v["layer"] for v in verdict["violations"]} == {"L2"}
    assert len(verdict["unverifiable"]) == 1  # kept: what could not be checked is still said


def test_t_t1_06_a_violation_of_l1_leaves_l2_and_l3_not_run(tmp_path: Path) -> None:
    answer = {
        "l1": [{"code": "S8", "message": "m"}, {"code": "parent_unknown", "message": "n"}],
        "payload": "not_run",
        "window": None,
        "envelope_sha256": HEX,
        "lookup_sha256": HEX,
        "contract_facts_sha256": None,
    }
    verdict = fake_client(tmp_path, answer).verdict(envelope_of())
    assert verdict["result"] == "FAIL" and verdict["codes"] == ["S8", "parent_unknown"]
    assert verdict["layers"] == {
        "L1": {"status": "FAIL", "codes": ["S8", "parent_unknown"]},
        "L2": {"status": "not_run", "kind": None, "codes": []},
        "L3": {"status": "not_run", "codes": []},
    }
    assert [v["layer"] for v in verdict["violations"]] == ["L1", "L1"]
    assert verdict["unverifiable"] == []


def test_the_window_is_the_third_layer_of_a_window_payload_only(tmp_path: Path) -> None:
    answer = _passing_answer()
    answer["payload"] = {"kind": "ic02", "status": "PASS", "violations": [], "unverifiable": []}
    answer["window"] = {
        "kind": "window_space",
        "status": "PASS_UNVERIFIED",
        "violations": [],
        "unverifiable": [{"check": "window_space_derivation", "reason": "rule_not_derivable"}],
    }
    window = envelope_of(payload_schema={"id": "msr.trajectory-window", "version": "0.10.0"})
    verdict = fake_client(tmp_path, answer).verdict(window)
    assert verdict["result"] == "PASS_UNVERIFIED"
    assert verdict["layers"]["L2"] == {"status": "PASS", "kind": "ic02", "codes": []}
    assert verdict["layers"]["L3"] == {"status": "PASS_UNVERIFIED", "codes": []}
    assert verdict["unverifiable"] == [
        {"layer": "L3", "check": "window_space_derivation", "reason": "rule_not_derivable"}
    ]
    assert verdict["rules"]["payload_schema"] == {
        "id": "msr.trajectory-window",
        "version": "0.10.0",
    }


def test_a_failed_window_space_is_a_violation_of_l3(tmp_path: Path) -> None:
    answer = _passing_answer()
    answer["payload"] = {"kind": "ic02", "status": "PASS", "violations": [], "unverifiable": []}
    answer["window"] = {
        "kind": "window_space",
        "status": "FAIL",
        "violations": [{"code": "window_space_equals_parent", "message": "m"}],
        "unverifiable": [],
    }
    verdict = fake_client(tmp_path, answer).verdict(
        envelope_of(payload_schema={"id": "msr.trajectory-window", "version": "0.10.0"})
    )
    assert verdict["result"] == "FAIL" and verdict["layers"]["L3"]["status"] == "FAIL"
    assert verdict["layers"]["L3"]["codes"] == ["window_space_equals_parent"]
    assert verdict["layers"]["L2"]["status"] == "PASS"


# ---- the catalogue (T-T1-07, 08, DC-T1-4) ---------------------------------------------------


def test_t_t1_07_an_unverifiable_row_the_catalogue_lacks_makes_no_verdict(tmp_path: Path) -> None:
    for row in (
        {"check": "made_up", "reason": "blob_not_supplied"},
        {"check": "points_bytes", "reason": "made_up"},
        {
            "check": "window_rows",
            "reason": "raw_not_in_artifact",
        },  # a window row, not this schema's
    ):
        client = fake_client(tmp_path, with_unverifiable([row]))
        with pytest.raises(CatalogMismatch):
            client.verdict(envelope_of())


def test_t_t1_08_a_typed_row_matches_any_field_name(tmp_path: Path) -> None:
    for field in ("trajectory_id", "a", "positions.0.space", "x.y.z"):
        row = {"check": f"contract_binding.{field}", "reason": "contract_field_unknown"}
        verdict = fake_client(tmp_path, with_unverifiable([row])).verdict(envelope_of())
        assert verdict["result"] == "PASS_UNVERIFIED"


def test_t_t1_08_a_typed_row_does_not_match_the_wrong_prefix_reason_or_an_empty_field(
    tmp_path: Path,
) -> None:
    for row in (
        {"check": "contract_binding.", "reason": "contract_field_unknown"},
        {"check": "contract_bindingx", "reason": "contract_field_unknown"},
        {"check": "other.field", "reason": "contract_field_unknown"},
        {"check": "contract_binding.field", "reason": "contract_absent"},
    ):
        with pytest.raises(CatalogMismatch):
            fake_client(tmp_path, with_unverifiable([row])).verdict(envelope_of())


def test_the_window_rows_of_the_catalogue_are_for_a_window_envelope(tmp_path: Path) -> None:
    answer = _passing_answer()
    answer["payload"] = {"kind": "ic02", "status": "PASS", "violations": [], "unverifiable": []}
    answer["window"] = {
        "kind": "window_space",
        "status": "PASS_UNVERIFIED",
        "violations": [],
        "unverifiable": [{"check": "window_rows", "reason": "raw_not_in_artifact"}],
    }
    window = envelope_of(payload_schema={"id": "msr.trajectory-window", "version": "0.10.0"})
    assert fake_client(tmp_path, answer).verdict(window)["result"] == "PASS_UNVERIFIED"


# ---- the document ---------------------------------------------------------------------------


def test_the_inputs_are_recorded_and_the_catalogue_and_bundle_are_named(tmp_path: Path) -> None:
    answer = _passing_answer()
    answer["contract_facts_sha256"] = "7" * 64
    lookup = {PARENT: {"content_id": CNT, "seq": 3, "observation_id": OBS}}
    envelope = envelope_of(
        parents=[{"role": "generated_text", "ordered": False, "artifact_ids": [PARENT]}]
    )
    client = fake_client(tmp_path, answer)
    blobs = {"blob:sha256:aa": b"two", "blob:sha256:bb": b"one", "blob:sha256:cc": b"two"}
    verdict = client.verdict(
        envelope,
        lookup=lookup,
        stored={ART: {"envelope": {}}, PARENT: {"envelope": {}}},
        blobs=blobs,
        contract_facts={"x": 1},
    )
    inputs = verdict["inputs"]
    assert inputs["lookup"] == lookup and inputs["lookup_sha256"] == HEX
    assert inputs["stored"] == sorted([ART, PARENT])
    assert inputs["blobs_checked"] == sorted(
        {hashlib.sha256(b"one").hexdigest(), hashlib.sha256(b"two").hexdigest()}
    )
    assert inputs["contract_facts_sha256"] == "7" * 64
    assert verdict["subject"] == {"artifact_id": ART, "content_id": CNT, "envelope_sha256": HEX}
    gate = verdict["gate"]
    assert gate["checker"]["bundle_sha256"] == hashlib.sha256(client.checker_bundle()).hexdigest()
    assert gate["unverifiable_catalog"]["sha256"] == HEX
    assert gate["runtime"]["packages"] == {"jsonschema": "4.26.0", "referencing": "0.37.0"}
    assert verdict["produced_by"]["name"] == "nvs-runtime"
    assert set(verdict) == {
        "verdict_version",
        "subject",
        "result",
        "layers",
        "codes",
        "violations",
        "unverifiable",
        "waived",
        "rules",
        "inputs",
        "gate",
        "produced_by",
        "verdict_sha256",
    }  # no eligibility, no admission, no time


def test_the_bundle_is_made_once_per_client(tmp_path: Path) -> None:
    client = fake_client(tmp_path)
    for _ in range(3):
        client.verdict(envelope_of())
    calls = client.process.calls  # type: ignore[attr-defined]
    assert calls.count("checker_files") == 1 and calls.count("catalog") == 1
    assert calls.count("gate") == 3 and calls.count("jcs_sha256") == 3
    assert client.checker_bundle() is client.checker_bundle()


def test_the_bundle_gets_the_files_of_nvs_runtime_itself(tmp_path: Path) -> None:
    client = fake_client(tmp_path)
    client.verdict(envelope_of())
    package_root = CHILD_SCRIPT.parents[2]
    for relative in (
        "nvs_runtime/hidden_trajectory/_gate_child.py",
        "nvs_runtime/hidden_trajectory/gate_client.py",
    ):
        b64 = base64.b64encode((package_root / relative).read_bytes()).decode()
        assert b64.encode() in client.checker_bundle()


def test_verdict_sha256_is_the_hash_of_everything_else(tmp_path: Path) -> None:
    verdict = fake_client(tmp_path).verdict(envelope_of())
    body = {k: v for k, v in verdict.items() if k != "verdict_sha256"}
    assert (
        verdict["verdict_sha256"]
        == hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    )


def test_the_same_inputs_give_the_same_verdict_with_a_fake_child(tmp_path: Path) -> None:
    first = fake_client(tmp_path).verdict(envelope_of())
    second = fake_client(tmp_path).verdict(envelope_of())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.parametrize(
    "envelope",
    [
        {},
        envelope_of(artifact_id=None),
        envelope_of(content_id=""),
        envelope_of(contract={"version": 1}),
        envelope_of(payload_schema={"id": "x"}),
        {"contract": "x"},
    ],
)
def test_an_envelope_that_cannot_say_what_it_is_makes_no_verdict(
    tmp_path: Path, envelope: dict[str, Any]
) -> None:
    with pytest.raises(GateUnavailable):
        fake_client(tmp_path).verdict(envelope)


def test_the_parent_process_does_not_import_the_child_or_sensos() -> None:
    assert "nvs_runtime.hidden_trajectory._gate_child" not in sys.modules
    assert "contracts" not in sys.modules


# ---- the real child --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def door() -> Any:
    sys.path.insert(0, str(EXAMPLE))
    try:
        from sensos_door import SensosDoor
    finally:
        sys.path.remove(str(EXAMPLE))
    return SensosDoor.from_environment()


@pytest.fixture(scope="module")
def client() -> GateClient:
    return GateClient.from_environment()


def _schema_file() -> Path:
    return (
        Path(os.environ.get("SENSOS_TOOLS", "/nonexistent")).parent
        / "docs/architecture/contracts/nvs/gate-verdict.schema.json"
    )


def schema_errors(verdict: Any) -> list[str]:
    """The errors of the sensos JSON Schema, found in ``GATE_PY`` (where ``jsonschema`` is)."""
    script = (
        "import json, sys\n"
        "from jsonschema import Draft202012Validator as V\n"
        "schema = json.load(open(sys.argv[1]))\n"
        "doc = json.load(sys.stdin)\n"
        "print(json.dumps([e.message for e in V(schema).iter_errors(doc)]))\n"
    )
    done = subprocess.run(
        [os.environ["GATE_PY"], "-c", script, str(_schema_file())],
        input=json.dumps(verdict),
        capture_output=True,
        text=True,
        check=True,
    )
    errors: list[str] = json.loads(done.stdout)
    return errors


needs_schema = pytest.mark.skipif(
    REAL_REASON is None and not _schema_file().is_file(),
    reason="SENSOS_TOOLS has no nvs/gate-verdict.schema.json",
)


def lookup_of(envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        envelope["artifact_id"]: {
            "content_id": envelope["content_id"],
            "seq": envelope["seq"],
            "observation_id": envelope["observation_id"],
        }
    }


@needs_real
def test_a_real_pass_and_a_real_pass_unverified(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-pass")
    check = session.check()
    result = session.build(check)
    envelope = result.envelopes["final_norm"]
    passed = client.verdict(envelope, blobs=result.blobs, contract_facts=check.facts)
    assert passed["result"] == "PASS", passed["violations"]
    assert passed["layers"]["L1"] == {"status": "PASS", "codes": []}
    assert passed["layers"]["L2"] == {"status": "PASS", "kind": "hidden_trajectory", "codes": []}
    unverified = client.verdict(envelope)
    assert unverified["result"] == "PASS_UNVERIFIED" and unverified["violations"] == []
    assert sorted((u["check"], u["reason"]) for u in unverified["unverifiable"]) == [
        ("contract_binding", "contract_not_supplied"),
        ("points_bytes", "blob_not_supplied"),
        ("token_index_bytes", "blob_not_supplied"),
    ]
    assert {u["layer"] for u in unverified["unverifiable"]} == {"L2"}
    assert passed["subject"]["artifact_id"] == envelope["artifact_id"]


@needs_real
def test_a_real_violation_of_l2_and_of_l1(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-fail")
    check = session.check()
    result = session.build(check)
    altered = _changed(door, result.envelopes["final_norm"], _set_row_rule)
    failed = client.verdict(altered, blobs=result.blobs, contract_facts=check.facts)
    assert failed["result"] == "FAIL" and failed["layers"]["L1"]["status"] == "PASS"
    assert failed["layers"]["L2"]["status"] == "FAIL"
    assert "E_TRAJ_ROW_RULE_KIND" in failed["codes"]
    broken = copy.deepcopy(result.envelopes["final_norm"])
    _set_n_gen(broken)  # not sealed again
    l1 = client.verdict(broken, blobs=result.blobs)
    assert l1["result"] == "FAIL" and l1["layers"]["L1"]["status"] == "FAIL"
    assert l1["layers"]["L2"] == {"status": "not_run", "kind": None, "codes": []}
    assert l1["layers"]["L3"] == {"status": "not_run", "codes": []}
    assert {v["layer"] for v in l1["violations"]} == {"L1"}


@needs_real
def test_t_t1_01_the_lookup_decides_whether_a_parent_is_known(
    door: Any, client: GateClient
) -> None:
    session = Session(door, item="gc-parent")
    check = session.check()
    parent = session.build(check).envelopes["block_out:3"]
    # an envelope with a parent, sealed with its lookup
    from nvs_observation.contract_check import rows_from_raw

    from nvs_runtime.hidden_trajectory import ParentRef, build_envelopes

    ref = ParentRef(
        parent["artifact_id"], parent["content_id"], parent["seq"], parent["observation_id"]
    )
    child = build_envelopes(
        session.document,
        check,
        ids=session.ids,
        space_id_of=door.space_id_of,
        seal=door.seal,
        rows=rows_from_raw(session.raw),
        parent=ref,
    )
    envelope = child.envelopes["final_norm"]
    assert envelope["parents"]
    with pytest.raises(LookupMissing):
        client.verdict(envelope, blobs=child.blobs, contract_facts=check.facts)
    stored = {parent["artifact_id"]: {"envelope": parent}}
    # the lookup is what L1 needs: with it L1 passes. L2 (HT-19) reads the parent's envelope from
    # ``stored`` -- which the design said stays empty for a hidden trajectory; it does not.
    without = client.verdict(
        envelope, lookup=lookup_of(parent), blobs=child.blobs, contract_facts=check.facts
    )
    assert without["layers"]["L1"] == {"status": "PASS", "codes": []}
    assert without["layers"]["L2"]["codes"] == ["parent_unknown"] and without["result"] == "FAIL"
    good = client.verdict(
        envelope,
        lookup=lookup_of(parent),
        stored=stored,
        blobs=child.blobs,
        contract_facts=check.facts,
    )
    assert good["layers"]["L1"] == {"status": "PASS", "codes": []}
    assert good["result"] == "PASS", (good["violations"], good["unverifiable"])
    assert good["inputs"]["lookup"] == lookup_of(parent)
    assert good["inputs"]["stored"] == [parent["artifact_id"]]
    wrong = lookup_of(parent)
    wrong[parent["artifact_id"]]["content_id"] = "cnt:" + "0" * 64
    bad = client.verdict(envelope, lookup=wrong, blobs=child.blobs, contract_facts=check.facts)
    assert bad["result"] == "FAIL" and bad["layers"]["L1"]["status"] == "FAIL"


@needs_real
def test_a_real_window_has_the_third_layer(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-window", n_gen=WINDOW + 64)
    check = session.check()
    result = session.build(check)
    parent = result.envelopes["block_out:3"]
    window, lookup, blobs = _window(door, session, parent, session_rows(session), 0)
    stored = {parent["artifact_id"]: {"envelope": parent}}
    verdict = client.verdict(window, lookup=lookup, stored=stored, blobs=blobs)
    assert verdict["layers"]["L2"]["kind"] == "ic02"
    assert verdict["layers"]["L3"]["status"] in ("PASS", "PASS_UNVERIFIED")
    assert verdict["inputs"]["stored"] == [parent["artifact_id"]]
    assert verdict["rules"]["payload_schema"] == {
        "id": "msr.trajectory-window",
        "version": "0.10.0",
    }
    assert verdict["result"] in ("PASS", "PASS_UNVERIFIED"), verdict["violations"]


def session_rows(session: Session) -> bytes:
    from nvs_observation.contract_check import rows_from_raw

    rows: bytes = rows_from_raw(session.raw)["block_out:3"]
    return rows


@needs_real
def test_t_t1_03_the_same_input_twice_is_the_same_bytes(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-twice")
    check = session.check()
    result = session.build(check)
    envelope = result.envelopes["final_norm"]
    first = client.verdict(envelope, blobs=result.blobs, contract_facts=check.facts)
    second = client.verdict(envelope, blobs=result.blobs, contract_facts=check.facts)
    other = GateClient.from_environment().verdict(
        envelope, blobs=result.blobs, contract_facts=check.facts
    )
    dump = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False)  # noqa: E731
    assert dump(first) == dump(second) == dump(other)
    assert first["verdict_sha256"] == other["verdict_sha256"]
    assert "produced_at" not in json.dumps(first) and "created_at" not in json.dumps(first)


@needs_real
@needs_schema
def test_t_t1_04_real_verdicts_fit_the_schema_of_sensos(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-schema", n_gen=WINDOW + 64)
    check = session.check()
    result = session.build(check)
    envelope = result.envelopes["final_norm"]
    verdicts = [
        client.verdict(envelope, blobs=result.blobs, contract_facts=check.facts),
        client.verdict(envelope),
        client.verdict(_changed(door, envelope, _set_row_rule), blobs=result.blobs),
    ]
    broken = copy.deepcopy(envelope)
    _set_n_gen(broken)
    verdicts.append(client.verdict(broken))
    parent = result.envelopes["block_out:3"]
    window, lookup, blobs = _window(door, session, parent, session_rows(session), 0)
    verdicts.append(
        client.verdict(
            window, lookup=lookup, stored={parent["artifact_id"]: {"envelope": parent}}, blobs=blobs
        )
    )
    assert {v["result"] for v in verdicts} == {"PASS", "PASS_UNVERIFIED", "FAIL"} or len(
        verdicts
    ) == 5
    for verdict in verdicts:
        assert schema_errors(verdict) == [], verdict["result"]
    extra = copy.deepcopy(verdicts[0])
    extra["extra"] = 1
    assert schema_errors(extra) != []
    waived = copy.deepcopy(verdicts[0])
    waived["result"] = "WAIVED"
    assert schema_errors(waived) != []


@needs_real
def test_t_t1_09_the_bundle_is_the_same_for_the_same_files_and_not_for_another(
    client: GateClient,
) -> None:
    def bundle(extra: bytes) -> dict[str, Any]:
        files = [{"repo": "nvs-runtime", "path": "x.py", "b64": base64.b64encode(extra).decode()}]
        return client.process.call({"op": "checker_files", "extra_files": files})

    first, second, changed = bundle(b"abc"), bundle(b"abc"), bundle(b"abd")
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["bundle_b64"] == second["bundle_b64"]
    assert changed["bundle_sha256"] != first["bundle_sha256"]
    # the order of the files is (repo, path), and what is hashed is what is held
    document = json.loads(base64.b64decode(first["bundle_b64"]))
    assert document["version"] == "gate-checker-bundle/1"
    keys = [(f["repo"], f["path"]) for f in document["files"]]
    assert keys == sorted(keys) and len(keys) == len(set(keys))
    for entry in document["files"]:
        assert entry["sha256"] == hashlib.sha256(base64.b64decode(entry["bytes_b64"])).hexdigest()
    assert (
        hashlib.sha256(base64.b64decode(first["bundle_b64"])).hexdigest() == first["bundle_sha256"]
    )


@needs_real
def test_the_bundle_of_the_client_names_the_files_and_is_the_same_twice(client: GateClient) -> None:
    other = GateClient.from_environment()
    assert client.checker_bundle() == other.checker_bundle()
    assert client.bundle_sha256() == other.bundle_sha256()
    assert hashlib.sha256(client.checker_bundle()).hexdigest() == client.bundle_sha256()
    document = json.loads(client.checker_bundle())
    paths = {(f["repo"], f["path"]) for f in document["files"]}
    assert ("nvs-runtime", "nvs_runtime/hidden_trajectory/_gate_child.py") in paths
    assert ("nvs-runtime", "nvs_runtime/hidden_trajectory/gate_client.py") in paths
    assert ("sensos", "tools/contracts/gate_nvs.py") in paths
    assert ("sensos", "tools/contracts/hid_row_rules.json") in paths
    assert ("sensos", "docs/architecture/contracts/nvs/hidden-trajectory.schema.json") in paths
    assert ("sensos", "docs/architecture/contracts/IC-00/1.1.0/artifact.schema.json") in paths
    sensos = [p for repo, p in paths if repo == "sensos"]
    assert len(sensos) >= 30
    assert all(not p.endswith(".pyc") and "__pycache__" not in p for p in sensos)
    assert not any(p.startswith("tests/") or "/vectors/" in p for p in sensos)


@needs_real
def test_t_t1_10_every_file_the_checker_reads_is_in_the_bundle(
    door: Any, client: GateClient
) -> None:
    """A real check of several kinds of envelope, traced in the child: every file of sensos that
    it imported or opened is in the bundle."""
    session = Session(door, item="gc-trace", n_gen=WINDOW + 64)
    check = session.check()
    result = session.build(check)
    parent = result.envelopes["block_out:3"]
    window, lookup, blobs = _window(door, session, parent, session_rows(session), 0)
    requests = [
        {"envelope": result.envelopes["final_norm"], "blobs": result.blobs, "facts": check.facts},
        {
            "envelope": _changed(door, result.envelopes["final_norm"], _set_row_rule),
            "blobs": result.blobs,
            "facts": check.facts,
        },
        {
            "envelope": window,
            "lookup": lookup,
            "stored": {parent["artifact_id"]: {"envelope": parent}},
            "blobs": blobs,
        },
    ]
    in_bundle = {
        p
        for repo, p in (
            (f["repo"], f["path"]) for f in json.loads(client.checker_bundle())["files"]
        )
        if repo == "sensos"
    }
    for request in requests:
        answer = client.process.call(
            {
                "op": "gate",
                "trace": True,
                "envelope": request["envelope"],
                "lookup": request.get("lookup") or {},
                "stored": request.get("stored") or {},
                "blobs": {u: base64.b64encode(d).decode() for u, d in request["blobs"].items()},
                "contract_facts": request.get("facts"),
            }
        )
        trace = answer["_trace"]
        assert trace["modules"] and trace["opened"]
        assert set(trace["modules"]) <= in_bundle, set(trace["modules"]) - in_bundle
        assert set(trace["opened"]) <= in_bundle, set(trace["opened"]) - in_bundle


@needs_real
def test_t_t1_11_verdict_sha256_is_the_jcs_hash_of_the_rest(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-hash")
    result = session.build()
    verdict = client.verdict(result.envelopes["final_norm"], blobs=result.blobs)
    body = {k: v for k, v in verdict.items() if k != "verdict_sha256"}
    assert (
        verdict["verdict_sha256"]
        == client.process.call({"op": "jcs_sha256", "value": body})["sha256"]
    )
    body["result"] = "FAIL"
    assert (
        client.process.call({"op": "jcs_sha256", "value": body})["sha256"]
        != verdict["verdict_sha256"]
    )


def _plain_jcs(value: Any) -> bytes:
    """RFC 8785 for integers, strings, booleans, null, lists and objects whose keys are ASCII."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@needs_real
def test_t_t1_12_envelope_sha256_is_the_hash_of_the_jcs_of_the_envelope(
    door: Any, client: GateClient
) -> None:
    session = Session(door, item="gc-env")
    result = session.build()
    envelope = result.envelopes["final_norm"]
    verdict = client.verdict(envelope, blobs=result.blobs)
    assert verdict["subject"]["envelope_sha256"] == hashlib.sha256(_plain_jcs(envelope)).hexdigest()
    assert verdict["inputs"]["lookup_sha256"] == hashlib.sha256(_plain_jcs({})).hexdigest()


@needs_real
def test_a_wrong_contract_facts_is_an_error_of_the_caller(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-facts")
    check = session.check()
    result = session.build(check)
    facts = copy.deepcopy(check.facts)
    assert facts is not None
    del facts["synthetic"]
    with pytest.raises(GateInternalError) as raised:
        client.verdict(result.envelopes["final_norm"], contract_facts=facts)
    assert raised.value.code == "gate_internal_error" and isinstance(raised.value, GateUnavailable)


@needs_real
def test_t_t1_13_a_failed_child_makes_no_verdict(door: Any) -> None:
    session = Session(door, item="gc-down")
    envelope = session.build().envelopes["final_norm"]
    for python in ("/usr/bin/false", "/usr/bin/true", "/bin/echo", "/nonexistent/python"):
        client = GateClient(Path(python), Path(os.environ["SENSOS_TOOLS"]))
        with pytest.raises(GateUnavailable):
            client.verdict(envelope)
    with pytest.raises(GateUnavailable):
        GateProcess(Path(os.environ["GATE_PY"]), Path(os.environ["SENSOS_TOOLS"])).call(
            {"op": "no_such_op"}
        )


@needs_real
def test_the_points_all_make_a_verdict(door: Any, client: GateClient) -> None:
    session = Session(door, item="gc-all")
    check = session.check()
    result = session.build(check)
    assert list(result.envelopes) == POINTS
    for envelope in result.envelopes.values():
        verdict = client.verdict(envelope, blobs=result.blobs, contract_facts=check.facts)
        assert verdict["result"] == "PASS", verdict["violations"]
    assert sha(b"") == hashlib.sha256(b"").hexdigest()
