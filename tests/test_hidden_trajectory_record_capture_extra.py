"""``capture_extra`` in ``build_record`` and ``empty_generation_record``, the reasons of U6-5 and
the check of every record that ``record.py`` returns (U6-5a'; T-U65-10, T-U65-11, T-U65-17).

T-U65-17 (the existing U6a tests keep their count and their result) is the existing test files
themselves, which are not changed: they build their records through the functions tested here.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import nvs_observation.capture_check as capture_check
import nvs_observation.capture_edge as capture_edge
import nvs_observation.capture_map as capture_map
import pytest
from edge_env import requires_edge, requires_no_edge
from nvs_observation.capture_extra import build_capture_extra
from nvs_observation.contract_build import contract_from_capture
from test_hidden_trajectory_envelope import OBSERVATION, SESSION, empty_capture
from u65_support import make_capture, make_facts

import nvs_runtime.hidden_trajectory.record as record_module
from nvs_runtime.hidden_trajectory.record import (
    build_record,
    empty_generation_record,
    position_entry,
    with_gate_result,
)
from nvs_runtime.hidden_trajectory.record_schema import validate_record

HASH = "a" * 64


def extra() -> dict[str, Any]:
    result = build_capture_extra(
        {
            "sampler": {"temperature": 0.0},
            "max_tokens": 8,
            "recorded_at": "2026-11-16T09:00:00+09:00",
        }
    )
    assert result.extra is not None
    return result.extra


def call(**changes: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "session_id": SESSION,
        "observation_id": OBSERVATION,
        "contract_session_id": "synthetic-session-1",
        "capture": empty_capture(n_gen=6),
        "contract_status": "VERIFIED",
        "contract_sha256": HASH,
        "hash_source": "recomputed_no_recorded",
        "refusals": [],
        "positions": {"final_norm": position_entry("enveloped", artifact_id="art:1")},
    }
    args.update(changes)
    return build_record(**args)


# ---- T-U65-10 --------------------------------------------------------------------------------


def test_t_u65_10_without_capture_extra_the_record_is_what_it_was() -> None:
    expected = {
        "record_version": "nvs.capture-record/1",
        "outcome": "enveloped",
        "session_id": SESSION,
        "observation_id": OBSERVATION,
        "contract_session_id": "synthetic-session-1",
        "capture": empty_capture(n_gen=6),
        "contract": {
            "status": "VERIFIED",
            "contract_sha256": HASH,
            "hash_source": "recomputed_no_recorded",
        },
        "refusals": [],
        "positions": {
            "final_norm": {
                "status": "enveloped",
                "reasons": [],
                "artifact_id": "art:1",
                "gate": {"status": "not_called", "violations": [], "unverifiable": []},
            }
        },
    }
    for record in (call(), call(capture_extra=None)):
        assert record == expected and "capture_extra" not in record
        assert json.dumps(record, sort_keys=True) == json.dumps(expected, sort_keys=True)
    empty = empty_generation_record(
        session_id=SESSION, observation_id=OBSERVATION, capture=empty_capture()
    )
    assert "capture_extra" not in empty
    assert list(empty) == list(
        empty_generation_record(
            session_id=SESSION,
            observation_id=OBSERVATION,
            capture=empty_capture(),
            capture_extra=None,
        )
    )


def test_capture_extra_is_a_copy_at_the_top_level_and_the_capture_is_untouched() -> None:
    given = extra()
    before = copy.deepcopy(given)
    record = call(capture_extra=given)
    assert record["capture_extra"] == given and record["capture_extra"] is not given
    assert list(record)[-1] == "capture_extra"
    assert set(record["capture"]) == set(record_module.CAPTURE_KEYS)
    record["capture_extra"]["values"]["sampler"]["temperature"] = 9
    assert given == before
    assert record["record_version"] == "nvs.capture-record/1"


def test_a_capture_extra_that_does_not_fit_is_not_returned() -> None:
    for bad in (
        {**extra(), "surprise": 1},
        {**extra(), "version": "nvs.capture-extra/2"},
        {"version": "nvs.capture-extra/1", "values": {"odd": 1}, "unknowns": []},
        {"version": "nvs.capture-extra/1", "values": {"sampler": float("nan")}, "unknowns": []},
        {"version": "nvs.capture-extra/1", "values": {}},
    ):
        with pytest.raises(ValueError, match="does not fit nvs.capture-record/1"):
            call(capture_extra=bad)
    with pytest.raises(ValueError):
        call(capture_extra=[1, 2])  # type: ignore[arg-type]


def test_a_record_that_does_not_fit_the_schema_is_never_returned() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        call(contract_status="verified")
    with pytest.raises(ValueError, match="does not fit"):
        call(hash_source="elsewhere")
    with pytest.raises(ValueError, match="does not fit"):
        call(contract_sha256="short")
    with pytest.raises(ValueError, match="does not fit"):
        call(positions={"final_norm": {"status": "enveloped"}})
    with pytest.raises(ValueError, match="exactly the keys"):  # the older check comes first
        call(capture={**empty_capture(), "extra": 1})
    record = call()
    with pytest.raises(ValueError, match="does not fit"):
        with_gate_result(record, "final_norm", status="")  # a gate status is not empty


def test_with_gate_result_still_returns_a_valid_copy() -> None:
    record = with_gate_result(
        call(), "final_norm", status="PASS_UNVERIFIED", unverifiable=[{"a": 1}]
    )
    assert validate_record(record) == []


# ---- T-U65-11 --------------------------------------------------------------------------------


@requires_edge
def test_t_u65_11_the_record_of_an_empty_generation_keeps_the_conditions_and_the_origin() -> None:
    capture = make_capture(n_gen=0, provenance={"sampler": {"temperature": 0.7, "top_k": 1}})
    facts = make_facts(capture)
    built = contract_from_capture(capture, facts, session_id="s", synthetic=False)
    assert built.refusals == ("empty_generation",) and built.contract is None
    record = empty_generation_record(
        session_id=SESSION,
        observation_id=OBSERVATION,
        capture=built.capture_block,
        capture_extra=built.capture_extra,
    )
    assert record["outcome"] == "empty_generation" and record["capture"]["n_gen"] == 0
    values = record["capture_extra"]["values"]
    assert values["sampler"] == {"temperature": 0.7, "top_k": 1}
    assert values["max_tokens"] == 8 and values["chat_template_kwargs"] == {}
    assert values["origin"] == facts["origin"]
    assert record["positions"]["final_norm"]["reasons"] == ["empty_generation"]
    assert validate_record(record) == []
    assert json.loads(json.dumps(record)) == record


@requires_no_edge
def test_an_empty_generation_without_the_edge_package_keeps_the_conditions() -> None:
    capture = make_capture(n_gen=0, provenance={"sampler": {"temperature": 0.7, "top_k": 1}})
    facts = make_facts(capture)
    built = contract_from_capture(capture, facts, session_id="s", synthetic=False)
    assert built.refusals == ("edge_unavailable", "empty_generation") and built.contract is None
    assert built.refusal_details == ("edge_unavailable: nvs_edge", "empty_generation: n_gen is 0")
    record = empty_generation_record(
        session_id=SESSION,
        observation_id=OBSERVATION,
        capture=built.capture_block,
        capture_extra=built.capture_extra,
    )
    assert record["outcome"] == "empty_generation" and record["capture"]["n_gen"] == 0
    values = record["capture_extra"]["values"]
    assert values["sampler"] == {"temperature": 0.7, "top_k": 1}
    assert values["max_tokens"] == 8 and values["chat_template_kwargs"] == {}
    assert values["origin"] == facts["origin"]
    assert record["positions"]["final_norm"]["reasons"] == ["empty_generation"]
    assert validate_record(record) == []
    assert json.loads(json.dumps(record)) == record


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: make_capture(n_gen=4, dimension=65536),  # dimension_out_of_range
        lambda c: make_capture(n_gen=4, provenance={"revision": None}),  # origin_mismatch
    ],
)
def test_the_record_of_a_refused_capture_fits_the_schema(mutation: Any) -> None:
    capture = mutation(None)
    facts = make_facts(make_capture(n_gen=4))
    built = contract_from_capture(capture, facts, session_id="s", synthetic=False)
    assert built.refusals
    record = build_record(
        session_id=SESSION,
        observation_id=OBSERVATION,
        contract_session_id=None,
        capture=built.capture_block,
        contract_status="none",
        contract_sha256=None,
        hash_source="none",
        refusals=list(built.refusals),
        positions={p: position_entry("not_built", built.refusals) for p in capture.points},
        capture_extra=built.capture_extra,
    )
    assert record["outcome"] == "refused" and validate_record(record) == []
    assert record["capture_extra"]["values"]["origin"]["revision"] == facts["origin"]["revision"]


# ---- the reasons -----------------------------------------------------------------------------


def test_the_reasons_of_u6_5_are_constants_with_the_values_the_design_names() -> None:
    names = {
        "REASON_STEPS_OUT_OF_RANGE": "steps_out_of_range",
        "REASON_DIMENSION_OUT_OF_RANGE": "dimension_out_of_range",
        "REASON_CAPTURE_INVALID": "capture_invalid",
        "REASON_SPACE_COLLISION": "space_collision",
        "REASON_ORIGIN_MISMATCH": "origin_mismatch",
        "REASON_ORIGIN_FACTS_INVALID": "origin_facts_invalid",
        "REASON_CONTRACT_NOT_BUILDABLE": "contract_not_buildable",
        "REASON_EDGE_UNAVAILABLE": "edge_unavailable",
        "REASON_EDGE_INCOMPATIBLE": "edge_incompatible",
        "REASON_CAPTURE_EXTRA_OVERFLOW": "capture_extra_overflow",
        "REASON_EMPTY_GENERATION": "empty_generation",
    }
    for name, value in names.items():
        assert getattr(record_module, name) == value
    # the modules of U6-5 use these constants, not their own copies of the strings
    assert capture_check.CAPTURE_INVALID is record_module.REASON_CAPTURE_INVALID
    assert capture_check.STEPS_OUT_OF_RANGE is record_module.REASON_STEPS_OUT_OF_RANGE
    assert capture_check.DIMENSION_OUT_OF_RANGE is record_module.REASON_DIMENSION_OUT_OF_RANGE
    assert capture_edge.EDGE_UNAVAILABLE is record_module.REASON_EDGE_UNAVAILABLE
    assert capture_edge.EDGE_INCOMPATIBLE is record_module.REASON_EDGE_INCOMPATIBLE
    assert capture_map.ORIGIN_FACTS_INVALID is record_module.REASON_ORIGIN_FACTS_INVALID
    assert capture_map.CAPTURE_EXTRA_OVERFLOW is record_module.REASON_CAPTURE_EXTRA_OVERFLOW
    assert capture_map.EMPTY_GENERATION is record_module.REASON_EMPTY_GENERATION


@requires_edge
def test_origin_facts_of_a_wrong_shape_is_origin_facts_invalid_in_the_record() -> None:
    capture = make_capture()
    built = contract_from_capture(capture, {"origin": None}, session_id="s", synthetic=False)
    assert built.refusals == ("origin_facts_invalid",)
    assert built.refusal_details == (
        "origin_facts_invalid: origin_facts.origin is missing or not a mapping",
    )


@requires_no_edge
def test_origin_facts_of_a_wrong_shape_without_the_edge_package_adds_its_reason() -> None:
    capture = make_capture()
    built = contract_from_capture(capture, {"origin": None}, session_id="s", synthetic=False)
    assert built.refusals == ("origin_facts_invalid", "edge_unavailable")
    assert built.refusal_details == (
        "origin_facts_invalid: origin_facts.origin is missing or not a mapping",
        "edge_unavailable: nvs_edge",
    )
