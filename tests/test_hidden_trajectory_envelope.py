"""hidden_trajectory.envelope and .record: whether an envelope is built, the envelope, and the
capture record (design U6 v0.1, sections 6 and 7; tests T-U6-E01, E07, E09, R01..R09).

The space id and the seal are sensos's and are stood in for here by two counting functions; the
join with the real Gate is U6-4. Tests that need the real nvs_edge (RAW files) are skipped
without it.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from nvs_observation.contract_check import (
    HASH_NONE,
    HASH_RECOMPUTED_MATCHES_RECORDED,
    HASH_RECOMPUTED_NO_RECORDED,
    HASH_RECORDED_ONLY,
    NOT_VERIFIED,
    REJECTED,
    VERIFIED,
    ContractCheck,
    check_contract_v2,
    edge_contract_available,
    rows_from_raw,
)

import nvs_runtime.hidden_trajectory.envelope as envelope_module
import nvs_runtime.hidden_trajectory.record as record_module
from nvs_runtime.hidden_trajectory import (
    RECORD_VERSION,
    BuildResult,
    EnvelopeIds,
    ParentRef,
    build_envelopes,
    capture_block,
    contract_ref_of,
    empty_generation_record,
    session_refusals,
    token_index_bytes,
    with_gate_result,
)
from nvs_runtime.hidden_trajectory.record import build_record, outcome_of, position_entry

FIXTURES = Path(__file__).parent / "fixtures" / "hidden_trajectory"
SAMPLE = json.loads((FIXTURES / "contract-v2-u4-base.json").read_text(encoding="utf-8"))
RECORDED = SAMPLE["contract_sha256"]
POINTS = ["block_out:3", "block_out:7", "mlp_out:7", "final_norm"]
N_GEN, DIMENSION = 6, 16
SESSION, OBSERVATION, PRODUCED_AT = "ses:" + "1" * 64, "obs:" + "2" * 64, "2026-11-15T00:00:00Z"

needs_real_contract_v2 = pytest.mark.skipif(
    not edge_contract_available(),
    reason="nvs_edge contract v2 is not installed",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document() -> dict[str, Any]:
    return copy.deepcopy(SAMPLE["document"])


def position(doc: dict[str, Any], point: str) -> dict[str, Any]:
    return next(p for p in doc["positions"] if p["point"] == point)


def rows(salt: float = 0.0, steps: int = N_GEN) -> bytes:
    return struct.pack(f"<{steps * DIMENSION}f", *[salt + i for i in range(steps * DIMENSION)])


def check_of(
    doc: Any,
    status: str = VERIFIED,
    *,
    sha256: str | None = RECORDED,
    hash_source: str = HASH_RECOMPUTED_MATCHES_RECORDED,
    reasons: tuple[str, ...] = (),
    report: dict[str, Any] | None = None,
) -> ContractCheck:
    """A ContractCheck written by hand, so that these tests do not need nvs_edge."""
    origin = doc.get("origin") if isinstance(doc, Mapping) else None
    synthetic = origin.get("synthetic") if isinstance(origin, Mapping) else None
    return ContractCheck(
        status=status,
        contract_sha256=sha256,
        hash_source=hash_source,
        facts={"contract": {"contract_sha256": sha256}} if status == VERIFIED else None,
        synthetic=synthetic if isinstance(synthetic, bool) else None,
        reasons=reasons,
        report=report,
    )


class Ports:
    """Stand-ins for the two sensos functions; they count their calls."""

    def __init__(self) -> None:
        self.spaces: list[dict[str, Any]] = []
        self.sealed: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self._seq = 0

    def space_id_of(self, materials: Mapping[str, Any]) -> str:
        self.spaces.append(dict(materials))
        return "hid:" + sha(json.dumps(materials, sort_keys=True).encode())

    def seal(self, envelope: dict[str, Any], lookup: Mapping[str, Any]) -> dict[str, Any]:
        self.sealed.append((copy.deepcopy(envelope), dict(lookup)))
        body = {k: v for k, v in envelope.items() if k not in ("content_id", "artifact_id")}
        digest = sha(json.dumps(body, sort_keys=True).encode())
        return {**envelope, "content_id": "cnt:" + digest, "artifact_id": "art:" + digest}

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def ids(self) -> EnvelopeIds:
        return EnvelopeIds(SESSION, OBSERVATION, PRODUCED_AT, self.next_seq)

    def build(self, doc: Any, check: ContractCheck | None = None, **kwargs: Any) -> BuildResult:
        return build_envelopes(
            doc,
            check_of(doc) if check is None else check,
            ids=self.ids,
            space_id_of=self.space_id_of,
            seal=self.seal,
            **kwargs,
        )

    @property
    def untouched(self) -> bool:
        return not self.spaces and not self.sealed and self._seq == 0


def assert_refused(result: BuildResult, ports: Ports, refusals: tuple[str, ...]) -> None:
    assert result.envelopes == {} and result.blobs == {}
    assert result.record["outcome"] == "refused"
    assert result.record["refusals"] == list(refusals)
    assert ports.untouched
    for entry in result.record["positions"].values():
        assert entry == {
            "status": "not_built",
            "reasons": ["session_refused"],
            "artifact_id": None,
            "gate": {"status": "not_called", "violations": [], "unverifiable": []},
        }


# ---- T-U6-E01: one envelope per position ----------------------------------------------------


def test_e01_four_positions_give_four_envelopes() -> None:
    ports, doc = Ports(), document()
    result = ports.build(doc)
    assert list(result.envelopes) == POINTS
    envelopes = list(result.envelopes.values())
    assert len({e["space_id"] for e in envelopes}) == 4
    assert len({e["artifact_id"] for e in envelopes}) == 4
    for same in ("session_id", "observation_id", "produced_at", "producer", "contract", "kind"):
        assert len({json.dumps(e[same], sort_keys=True) for e in envelopes}) == 1
    for field in ("binding", "contract_ref"):
        assert len({json.dumps(e["payload"][field], sort_keys=True) for e in envelopes}) == 1
    assert [e["seq"] for e in envelopes] == [1, 2, 3, 4]
    assert [e["payload"]["point"] for e in envelopes] == POINTS
    assert ports.spaces == [p["space"] for p in doc["positions"]]
    assert doc == SAMPLE["document"]
    json.dumps(result.envelopes)


def test_e01_the_envelope_in_full() -> None:
    ports, doc = Ports(), document()
    envelope = ports.build(doc).envelopes["final_norm"]
    payload = envelope.pop("payload")
    assert payload["point"] == "final_norm" and payload["contract_ref"] == {
        "name": "RawObservationContract",
        "version": 2,
        "contract_sha256": RECORDED,
    }
    content_id, artifact_id = envelope.pop("content_id"), envelope.pop("artifact_id")
    assert content_id.startswith("cnt:") and artifact_id.startswith("art:")
    assert content_id != "cnt:" + "0" * 64
    assert envelope == {
        "contract": {"id": "IC-00", "version": "1.1.0"},
        "kind": "observation",
        "payload_schema": {"id": "nvs.hidden-trajectory", "version": "0.1.0"},
        "producer": {"name": "sensos-inference", "version": "1", "build": "adapter-1"},
        "session_id": SESSION,
        "observation_id": OBSERVATION,
        "event_span": None,
        "space_id": ports.space_id_of(position(doc, "final_norm")["space"]),
        "parents": [],
        "policy_refs": [],
        "calibration_ref": None,
        "rule_refs": [],
        "observed_at": None,
        "seq": 4,
        "produced_at": PRODUCED_AT,
        "extensions": {
            "x-contract-session-id": "synthetic-session-1",
            "x-assembled-by": envelope_module.ASSEMBLED_BY,
        },
    }
    assert envelope_module.ASSEMBLED_BY.startswith("nvs-runtime/")


def test_e01_the_seal_is_given_placeholder_ids_and_the_lookup() -> None:
    ports = Ports()
    parent = ParentRef("art:" + "a" * 64, "cnt:" + "b" * 64, 7, OBSERVATION)
    result = ports.build(document(), parent=parent)
    for unsealed, lookup in ports.sealed:
        assert unsealed["content_id"] == "cnt:" + "0" * 64
        assert unsealed["artifact_id"] == "art:" + "0" * 64
        assert lookup == {
            parent.artifact_id: {
                "content_id": parent.content_id,
                "seq": 7,
                "observation_id": OBSERVATION,
            }
        }
    for envelope in result.envelopes.values():
        assert envelope["parents"] == [
            {"role": "generated_text", "ordered": False, "artifact_ids": [parent.artifact_id]}
        ]


def test_e01_producer_build_and_payload_options() -> None:
    ports = Ports()
    result = ports.build(
        document(), producer_build="build-9", copied=True, policy_name="rows-0+128/1"
    )
    for envelope in result.envelopes.values():
        assert envelope["producer"] == {
            "name": "sensos-inference",
            "version": "1",
            "build": "build-9",
        }
        assert envelope["payload"]["raw_ref"]["copied"] is True
        assert envelope["payload"]["policy"]["name"] == "rows-0+128/1"


def test_e01_extensions_per_session_and_per_point() -> None:
    ports = Ports()
    result = ports.build(
        document(),
        session_extensions={"x-prompt-id": "PB07", "x-seed": 3},
        extensions={"final_norm": {"x-raw-location": "/data/final.trj", "x-raw-format": "TRJ_V2"}},
    )
    final, block = result.envelopes["final_norm"], result.envelopes["block_out:3"]
    assert final["extensions"]["x-raw-location"] == "/data/final.trj"
    assert final["extensions"]["x-prompt-id"] == "PB07" and final["extensions"]["x-seed"] == 3
    assert "x-raw-location" not in block["extensions"]
    assert block["extensions"]["x-prompt-id"] == "PB07"


@pytest.mark.parametrize(
    ("given", "message"),
    [
        ({"raw-location": "x"}, "starts with 'x-'"),
        ({3: "x"}, "starts with 'x-'"),
        ({"x-assembled-by": "me"}, "written by nvs-runtime"),
        ({"x-contract-session-id": "s"}, "written by nvs-runtime"),
    ],
)
def test_e01_extension_keys_are_checked(given: dict[Any, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Ports().build(document(), session_extensions=given)
    with pytest.raises(ValueError, match=message):
        Ports().build(document(), extensions={"block_out:3": given})


def test_e01_blobs_token_index_always_points_where_rows_were_given() -> None:
    ports, doc = Ports(), document()
    data = rows()
    position(doc, "final_norm")["trajectory_id"] = sha(data)
    result = ports.build(doc, rows={"final_norm": data})
    index = token_index_bytes(N_GEN)
    assert result.blobs == {
        "blob:sha256:" + sha(index): index,
        "blob:sha256:" + sha(data): data,
    }
    for envelope in result.envelopes.values():
        assert envelope["payload"]["token_index"]["uri"] in result.blobs
    assert result.envelopes["final_norm"]["payload"]["points"]["uri"] in result.blobs
    assert result.envelopes["block_out:3"]["payload"]["points"]["uri"] not in result.blobs


def test_e01_record_of_a_fully_enveloped_capture() -> None:
    ports, doc = Ports(), document()
    result = ports.build(doc)
    assert result.record == {
        "record_version": "nvs.capture-record/1",
        "outcome": "enveloped",
        "session_id": SESSION,
        "observation_id": OBSERVATION,
        "contract_session_id": "synthetic-session-1",
        "capture": {
            "n_in": 5,
            "n_gen": 6,
            "finish_reason": "length",
            "points": POINTS,
            "model_build_id": "synthetic-test-1",
            "compute_environment": "synthetic-env",
            "adapter": "adapter-1",
            "prompt_ids_sha256": "b" * 64,
            "gen_token_ids_sha256": "c" * 64,
        },
        "contract": {
            "status": "VERIFIED",
            "contract_sha256": RECORDED,
            "hash_source": "recomputed_matches_recorded",
        },
        "refusals": [],
        "positions": {
            point: {
                "status": "enveloped",
                "reasons": [],
                "artifact_id": result.envelopes[point]["artifact_id"],
                "gate": {"status": "not_called", "violations": [], "unverifiable": []},
            }
            for point in POINTS
        },
    }
    assert RECORD_VERSION == "nvs.capture-record/1"


# ---- T-U6-E07 and DC-U6-4: contract_ref and where the hash came from ------------------------


def test_e07_not_verified_with_a_recorded_hash_refers_to_it() -> None:
    doc = document()
    check = check_of(doc, NOT_VERIFIED, hash_source=HASH_RECORDED_ONLY, reasons=("nvs_edge",))
    result = Ports().build(doc, check)
    assert list(result.envelopes) == POINTS
    for envelope in result.envelopes.values():
        assert envelope["payload"]["contract_ref"]["contract_sha256"] == RECORDED
    assert result.record["contract"] == {
        "status": "NOT_VERIFIED",
        "contract_sha256": RECORDED,
        "hash_source": "recorded_only",
    }


def test_e07_not_verified_without_a_recorded_hash_has_no_contract_ref() -> None:
    doc = document()
    check = check_of(doc, NOT_VERIFIED, sha256=None, hash_source=HASH_NONE)
    result = Ports().build(doc, check)
    assert all(e["payload"]["contract_ref"] is None for e in result.envelopes.values())
    assert result.record["contract"]["hash_source"] == "none"


def test_e07_contract_ref_of() -> None:
    doc = document()
    assert contract_ref_of(check_of(doc)) == {
        "name": "RawObservationContract",
        "version": 2,
        "contract_sha256": RECORDED,
    }
    assert contract_ref_of(check_of(doc, NOT_VERIFIED, sha256=None)) is None
    assert contract_ref_of(check_of(doc, REJECTED)) is None


def test_dc_u6_4_hash_source_is_in_the_record_only() -> None:
    doc = document()
    check = check_of(doc, hash_source=HASH_RECOMPUTED_NO_RECORDED)
    result = Ports().build(doc, check)
    assert list(result.envelopes) == POINTS
    assert result.record["contract"] == {
        "status": "VERIFIED",
        "contract_sha256": RECORDED,
        "hash_source": "recomputed_no_recorded",
    }
    text = json.dumps(result.envelopes)
    assert "hash_source" not in text and "recomputed_no_recorded" not in text


# ---- T-U6-R01, R02 (DC-U6-5): synthetic -----------------------------------------------------


def test_r01_a_synthetic_contract_gives_no_envelope() -> None:
    ports, doc = Ports(), document()
    doc["origin"]["synthetic"] = True
    assert_refused(ports.build(doc, rows={"final_norm": rows()}), ports, ("synthetic",))


@pytest.mark.parametrize("stated", ["false", 0, None, "unknown", 1])
@pytest.mark.parametrize("status", [VERIFIED, NOT_VERIFIED])
def test_r02_synthetic_not_stated_gives_no_envelope(stated: Any, status: str) -> None:
    ports, doc = Ports(), document()
    doc["origin"]["synthetic"] = stated
    assert_refused(ports.build(doc, check_of(doc, status)), ports, ("synthetic_not_stated",))


def test_r02_synthetic_absent_gives_no_envelope() -> None:
    ports, doc = Ports(), document()
    del doc["origin"]["synthetic"]
    assert_refused(ports.build(doc), ports, ("synthetic_not_stated",))


def test_r02_the_document_decides_not_the_check() -> None:
    """A check that says "not synthetic" does not open the way for a synthetic document."""
    ports, doc = Ports(), document()
    check = check_of(doc)
    assert check.synthetic is False
    doc["origin"]["synthetic"] = True
    assert_refused(ports.build(doc, check), ports, ("synthetic",))


# ---- T-U6-R03: a contract that failed verification ------------------------------------------


@pytest.mark.parametrize(
    "reasons", [("contract_structure",), ("contract_sha256_mismatch",), ("contract_structure", "x")]
)
def test_r03_a_rejected_contract_gives_no_envelope(reasons: tuple[str, ...]) -> None:
    ports, doc = Ports(), document()
    check = check_of(doc, REJECTED, sha256=None, hash_source=HASH_NONE, reasons=reasons)
    result = ports.build(doc, check)
    assert_refused(result, ports, reasons)
    assert result.record["contract"] == {
        "status": "REJECTED",
        "contract_sha256": None,
        "hash_source": "none",
    }


def test_r03_not_verified_and_not_a_contract_v2() -> None:
    ports, doc = Ports(), document()
    doc["contract"]["version"] = 1
    assert_refused(ports.build(doc, check_of(doc, NOT_VERIFIED)), ports, ("not_a_contract_v2",))


@pytest.mark.parametrize("not_a_contract", [None, [], "text", {}])
def test_r03_what_is_not_a_document_is_refused_with_reasons(not_a_contract: Any) -> None:
    ports = Ports()
    result = ports.build(not_a_contract, check_of(not_a_contract, NOT_VERIFIED, sha256=None))
    assert result.envelopes == {} and ports.untouched
    assert result.record["outcome"] == "refused"
    assert result.record["refusals"][:2] == ["synthetic_not_stated", "not_a_contract_v2"]
    assert result.record["refusals"][-1] == "no_positions"
    assert result.record["positions"] == {} and result.record["contract_session_id"] is None
    assert result.record["capture"]["points"] == []


def test_r03_a_contract_without_positions() -> None:
    ports, doc = Ports(), document()
    doc["positions"] = []
    assert_refused(ports.build(doc, check_of(doc, NOT_VERIFIED)), ports, ("no_positions",))


def test_an_unknown_contract_status_is_an_error() -> None:
    doc = document()
    with pytest.raises(ValueError, match="unknown contract status"):
        session_refusals(doc, check_of(doc, "PASSED"))


# ---- T-U6-R04: values no payload can hold ---------------------------------------------------


@pytest.mark.parametrize(
    ("section", "key", "reason"),
    [
        ("binding", "n_in", "payload_cannot_express:binding.n_in"),
        ("binding", "finish_reason", "payload_cannot_express:binding.finish_reason"),
        ("origin", "runtime", "payload_cannot_express:origin.runtime"),
    ],
)
def test_r04_unknown_where_the_payload_has_no_word_for_it(
    section: str, key: str, reason: str
) -> None:
    ports, doc = Ports(), document()
    doc[section][key] = "unknown"
    assert_refused(ports.build(doc), ports, (reason,))


def test_all_refusals_are_recorded_together() -> None:
    ports, doc = Ports(), document()
    doc["origin"]["synthetic"] = True
    doc["binding"]["n_in"] = "unknown"
    check = check_of(doc, REJECTED, sha256=None, reasons=("contract_sha256_mismatch",))
    assert_refused(
        ports.build(doc, check),
        ports,
        ("synthetic", "contract_sha256_mismatch", "payload_cannot_express:binding.n_in"),
    )


# ---- T-U6-R05, R06, R07: one position -------------------------------------------------------


def g1_report(**positions: dict[str, str]) -> dict[str, Any]:
    return {"G1": {"positions": {p: {"checks": c} for p, c in positions.items()}}}


def test_r05_one_position_whose_raw_failed_g1() -> None:
    ports, doc = Ports(), document()
    report = g1_report(
        **{
            "block_out:7": {"raw_file": "FAIL"},
            "mlp_out:7": {"raw_file": "PASS", "trajectory_id": "PASS", "dimension": "PASS"},
            "final_norm": {"raw_file": "UNVERIFIABLE"},
        }
    )
    result = ports.build(doc, check_of(doc, report=report), rows={"block_out:7": rows()})
    assert list(result.envelopes) == ["block_out:3", "mlp_out:7", "final_norm"]
    assert result.record["outcome"] == "partly_enveloped"
    assert result.record["refusals"] == []
    assert result.record["positions"]["block_out:7"] == {
        "status": "not_built",
        "reasons": ["g1:raw_file"],
        "artifact_id": None,
        "gate": {"status": "not_called", "violations": [], "unverifiable": []},
    }
    assert [e["seq"] for e in result.envelopes.values()] == [1, 2, 3]
    assert len(ports.sealed) == 3


def test_r05_every_failed_check_of_the_position_is_named() -> None:
    ports, doc = Ports(), document()
    checks = {"raw_file": "PASS", "trajectory_id": "FAIL", "stored_range": "FAIL"}
    result = ports.build(doc, check_of(doc, report=g1_report(final_norm=checks)))
    assert result.record["positions"]["final_norm"]["reasons"] == [
        "g1:trajectory_id",
        "g1:stored_range",
    ]


def test_r06_unknown_final_norm_applied_stops_that_position_only() -> None:
    ports, doc = Ports(), document()
    position(doc, "block_out:3")["final_norm_applied"] = "unknown"
    result = ports.build(doc)
    assert list(result.envelopes) == ["block_out:7", "mlp_out:7", "final_norm"]
    assert result.record["positions"]["block_out:3"]["reasons"] == [
        "payload_cannot_express:final_norm_applied"
    ]


def test_r07_unknown_trajectory_id_with_and_without_rows() -> None:
    doc = document()
    position(doc, "final_norm")["trajectory_id"] = "unknown"
    without = Ports().build(doc)
    assert "final_norm" not in without.envelopes
    assert without.record["positions"]["final_norm"]["reasons"] == ["trajectory_id_unknown"]
    assert without.record["outcome"] == "partly_enveloped"
    data = rows()
    with_rows = Ports().build(doc, rows={"final_norm": data})
    payload = with_rows.envelopes["final_norm"]["payload"]
    assert payload["raw_ref"]["trajectory_id"] == sha(data)
    assert with_rows.blobs[payload["points"]["uri"]] == data
    assert with_rows.record["outcome"] == "enveloped"


def test_no_position_can_be_built() -> None:
    ports, doc = Ports(), document()
    for item in doc["positions"]:
        item["trajectory_id"] = "unknown"
    result = ports.build(doc)
    assert result.envelopes == {} and ports.untouched
    assert result.record["outcome"] == "not_enveloped"
    assert result.record["refusals"] == []


def test_a_second_position_of_the_same_name_is_not_built_twice() -> None:
    """Only a document nvs_edge did not check can have one; the first position stands."""
    ports, doc = Ports(), document()
    doc["positions"].append(copy.deepcopy(position(doc, "final_norm")))
    result = ports.build(doc, check_of(doc, NOT_VERIFIED))
    assert list(result.envelopes) == POINTS and len(ports.sealed) == 4


# ---- T-U6-R08, R09: the capture record ------------------------------------------------------


def empty_capture(**changes: Any) -> dict[str, Any]:
    block = capture_block(
        n_in=18,
        n_gen=0,
        finish_reason="eos",
        points=["block_out:15", "final_norm"],
        model_build_id="qwen35-4b-mlx-4bit",
        compute_environment="mlx-4bit",
        adapter="qwen35-mlx-capture",
        prompt_ids_sha256="b" * 64,
        gen_token_ids_sha256=sha(b""),
    )
    block.update(changes)
    return block


def test_r08_the_record_of_an_empty_generation() -> None:
    record = empty_generation_record(
        session_id=SESSION, observation_id=OBSERVATION, capture=empty_capture()
    )
    not_built = {
        "status": "not_built",
        "reasons": ["empty_generation"],
        "artifact_id": None,
        "gate": {"status": "not_called", "violations": [], "unverifiable": []},
    }
    assert record == {
        "record_version": "nvs.capture-record/1",
        "outcome": "empty_generation",
        "session_id": SESSION,
        "observation_id": OBSERVATION,
        "contract_session_id": None,
        "capture": empty_capture(),
        "contract": {"status": "none", "contract_sha256": None, "hash_source": "none"},
        "refusals": [],
        "positions": {"block_out:15": not_built, "final_norm": not_built},
    }


@pytest.mark.parametrize("n_gen", [1, 6, None, "0", False, -1])
def test_r08_only_n_gen_zero_is_an_empty_generation(n_gen: Any) -> None:
    with pytest.raises(ValueError, match="n_gen == 0"):
        empty_generation_record(
            session_id=SESSION, observation_id=OBSERVATION, capture=empty_capture(n_gen=n_gen)
        )


def test_r08_the_record_module_does_not_reach_nvs_edge() -> None:
    """No contract is built for an empty generation: the module cannot even call nvs_edge."""
    assert imports_of(record_module) == {
        "__future__",
        "collections.abc",
        "copy",
        "typing",
        "nvs_observation.contracts.capture_record",
    }
    capture_record = __import__(
        "nvs_observation.contracts.capture_record", fromlist=["capture_block"]
    )
    roots = {name.split(".")[0] for name in imports_of(capture_record)}
    assert roots <= sys.stdlib_module_names
    assert "nvs_edge" not in roots


def test_r09_records_are_json_repeatable_and_hold_no_text() -> None:
    doc = document()
    first, second = Ports().build(doc).record, Ports().build(doc).record
    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert set(first) == {
        "record_version",
        "outcome",
        "session_id",
        "observation_id",
        "contract_session_id",
        "capture",
        "contract",
        "refusals",
        "positions",
    }
    assert set(first["capture"]) == set(record_module.CAPTURE_KEYS)
    for forbidden in ("prompt", "text", "token_ids", "key", "secret", "special_tokens_in"):
        assert forbidden not in {k for k in first["capture"] if not k.endswith("_sha256")}
    assert "special_tokens_in" not in json.dumps(first) and "eos_ids" not in json.dumps(first)


def test_r09_the_record_shares_nothing_with_its_inputs() -> None:
    capture = empty_capture()
    record = empty_generation_record(
        session_id=SESSION, observation_id=OBSERVATION, capture=capture
    )
    record["capture"]["points"].append("changed")
    assert capture == empty_capture()


def test_build_record_wants_exactly_the_capture_keys() -> None:
    capture = empty_capture()
    del capture["adapter"]
    with pytest.raises(ValueError, match="exactly the keys"):
        build_record(
            session_id=SESSION,
            observation_id=OBSERVATION,
            contract_session_id=None,
            capture=capture,
            contract_status="none",
            contract_sha256=None,
            hash_source="none",
            refusals=(),
            positions={},
        )


def test_position_entry_and_outcome() -> None:
    with pytest.raises(ValueError, match="unknown position status"):
        position_entry("built")
    with pytest.raises(ValueError, match="artifact_id"):
        position_entry("enveloped")
    with pytest.raises(ValueError, match="artifact_id"):
        position_entry("not_built", artifact_id="art:x")
    built, missing = position_entry("enveloped", artifact_id="art:x"), position_entry("not_built")
    assert outcome_of((), {"a": built, "b": built}) == "enveloped"
    assert outcome_of((), {"a": built, "b": missing}) == "partly_enveloped"
    assert outcome_of((), {"a": missing}) == "not_enveloped"
    assert outcome_of((), {}) == "not_enveloped"
    assert outcome_of(("synthetic",), {"a": missing}) == "refused"


def test_with_gate_result_writes_a_copy() -> None:
    record = Ports().build(document()).record
    before = copy.deepcopy(record)
    violations = [{"code": "E_TRAJ_ROWS", "message": "m"}]
    after = with_gate_result(
        record,
        "final_norm",
        status="FAIL",
        violations=violations,
        unverifiable=[{"check": "points_bytes", "reason": "blob_not_supplied"}],
    )
    assert record == before
    assert after["positions"]["final_norm"]["gate"] == {
        "status": "FAIL",
        "violations": violations,
        "unverifiable": [{"check": "points_bytes", "reason": "blob_not_supplied"}],
    }
    assert after["positions"]["block_out:3"]["gate"]["status"] == "not_called"
    violations[0]["code"] = "changed"
    assert after["positions"]["final_norm"]["gate"]["violations"][0]["code"] == "E_TRAJ_ROWS"


def test_with_gate_result_only_for_an_enveloped_position() -> None:
    doc = document()
    position(doc, "final_norm")["trajectory_id"] = "unknown"
    record = Ports().build(doc).record
    for point in ("final_norm", "block_out:99"):
        with pytest.raises(ValueError, match="not an enveloped position"):
            with_gate_result(record, point, status="PASS")


# ---- T-U6-E09: imports ----------------------------------------------------------------------


def imports_of(module: Any) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
    return found


def test_e09_envelope_module_imports_no_numpy_nvs_edge_or_sensos() -> None:
    """Standard library and this package only: no numpy, no nvs_edge, no sensos contracts."""
    assert imports_of(envelope_module) == {
        "__future__",
        "collections.abc",
        "dataclasses",
        "typing",
        "nvs_observation.contract_check",
        "nvs_runtime.__version__",
        "nvs_runtime.hidden_trajectory",
        "nvs_runtime.hidden_trajectory.payload",
    }


def test_e09_no_module_of_the_package_names_sensos_or_numpy() -> None:
    package = Path(envelope_module.__file__).parent
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                root = name.split(".")[0]
                assert root not in ("numpy", "contracts", "sensos_inference", "nvs_edge"), (
                    path.name,
                    name,
                )


# ---- with the real nvs_edge: RAW files ------------------------------------------------------


def _real_session() -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    """The sample contract with RAW files made by nvs_edge for each of its positions."""
    import numpy as np
    from nvs_edge.raw_store import encode_raw

    doc = document()
    raw: dict[str, bytes] = {}
    expected_rows: dict[str, bytes] = {}
    for index, item in enumerate(doc["positions"]):
        states = (np.arange(N_GEN * DIMENSION, dtype=np.float32) + index).reshape(N_GEN, DIMENSION)
        raw[item["point"]], header = encode_raw(states, dimension=DIMENSION)
        item["trajectory_id"] = header.trajectory_id
        expected_rows[item["point"]] = states.astype("<f4").tobytes()
    return doc, raw, expected_rows


@needs_real_contract_v2
def test_real_contract_and_raw_give_four_envelopes_with_their_rows() -> None:
    doc, raw, expected_rows = _real_session()
    check = check_contract_v2(doc, raw_bytes=raw)
    assert check.status == VERIFIED and check.hash_source == HASH_RECOMPUTED_NO_RECORDED
    assert check.report is not None and check.report["G1"]["result"] == "UNVERIFIABLE"
    decoded = rows_from_raw(raw)
    assert decoded == expected_rows
    result = Ports().build(doc, check, rows=decoded)
    assert list(result.envelopes) == POINTS
    assert result.record["outcome"] == "enveloped"
    assert result.record["contract"]["hash_source"] == "recomputed_no_recorded"
    for point, envelope in result.envelopes.items():
        payload = envelope["payload"]
        assert payload["contract_ref"]["contract_sha256"] == check.contract_sha256
        assert result.blobs[payload["points"]["uri"]] == expected_rows[point]
        assert payload["points"]["sha256"] == sha(expected_rows[point])


@needs_real_contract_v2
def test_r05_real_raw_of_one_position_is_damaged() -> None:
    doc, raw, _ = _real_session()
    raw["block_out:7"] = raw["block_out:7"][:-1] + bytes([raw["block_out:7"][-1] ^ 1])
    check = check_contract_v2(doc, raw_bytes=raw)
    assert check.status == VERIFIED
    decoded = rows_from_raw(raw)
    assert sorted(decoded) == sorted(p for p in POINTS if p != "block_out:7")
    result = Ports().build(doc, check, rows=decoded)
    assert list(result.envelopes) == ["block_out:3", "mlp_out:7", "final_norm"]
    assert result.record["outcome"] == "partly_enveloped"
    assert result.record["positions"]["block_out:7"]["reasons"] == ["g1:raw_file"]


@needs_real_contract_v2
def test_r05_real_raw_that_belongs_to_another_position() -> None:
    doc, raw, _ = _real_session()
    raw["final_norm"] = raw["block_out:3"]
    result = Ports().build(doc, check_contract_v2(doc, raw_bytes=raw), rows=rows_from_raw(raw))
    assert "final_norm" not in result.envelopes
    assert result.record["positions"]["final_norm"]["reasons"] == ["g1:trajectory_id"]


@needs_real_contract_v2
def test_r01_real_synthetic_contract_is_verified_and_still_refused() -> None:
    ports = Ports()
    doc, raw, _ = _real_session()
    doc["origin"]["synthetic"] = True
    check = check_contract_v2(doc, raw_bytes=raw)
    assert check.status == VERIFIED and check.synthetic is True
    assert check.facts is not None and check.facts["synthetic"] is True
    assert_refused(ports.build(doc, check, rows=rows_from_raw(raw)), ports, ("synthetic",))
