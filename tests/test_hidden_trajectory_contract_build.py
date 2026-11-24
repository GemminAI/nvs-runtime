"""U6-5c: RAW and instrument records, refusals, ``contract_from_capture`` (U6-5 v0.4, 6.1, 7).

Tests: T-U65-01 up to "the contract is built and ``check_contract_v2`` passes" (the envelopes and
the Gate are U6-5d), T-U65-13, T-U65-16. The captures are fakes shaped like U1's
``ObservationCapture`` (``u65_support``); no model is loaded. Cases that call ``nvs_edge`` are
skipped where it is not usable.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from nvs_observation import capture_edge
from nvs_observation.adapter_session import CapturedWithOrigin
from nvs_observation.contract_build import (
    ContractBuild,
    contract_from_capture,
    contract_from_captured,
)
from nvs_observation.contract_check import ContractCheck, check_contract_v2
from u65_support import (
    HASH_A,
    make_capture,
    make_facts,
    with_provenance,
)

from nvs_runtime.hidden_trajectory import session_refusals
from nvs_runtime.hidden_trajectory.record import CAPTURE_KEYS, build_record, position_entry

EDGE, _ = capture_edge.current()
pytestmark = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")

CLOCK_TIME = datetime(2026, 11, 16, 9, 30, 15, tzinfo=timezone(timedelta(hours=9)))
KINDS = ["llama", "qwen"]


def clock() -> datetime:
    return CLOCK_TIME


def build(capture: Any, facts: dict[str, Any] | None = None, **kwargs: Any) -> ContractBuild:
    kwargs.setdefault("session_id", "session-1")
    kwargs.setdefault("synthetic", False)
    kwargs.setdefault("clock", clock)
    return contract_from_capture(capture, make_facts(capture) if facts is None else facts, **kwargs)


def verify(result: ContractBuild, **kwargs: Any) -> ContractCheck:
    assert result.contract is not None, result.refusal_details
    return check_contract_v2(
        result.contract,
        raw_bytes=result.raw_bytes,
        instrument_records=result.instrument_records,
        **kwargs,
    )


def sha256_of_document(document: dict[str, Any]) -> str:
    assert EDGE is not None
    import importlib

    return str(importlib.import_module("nvs_edge.contract").contract_sha256(document))


# ---- T-U65-01 (the part before the envelopes) ------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_t_u65_01_the_contract_is_built_and_verification_passes(kind: str) -> None:
    capture = make_capture(kind=kind)
    result = build(capture)
    assert result.refusals == () and result.refusal_details == ()
    assert result.contract is not None
    assert set(result.raw_bytes) == set(result.instrument_records) == set(capture.points)
    check = verify(result, recorded_sha256=sha256_of_document(result.contract))
    assert check.status == "VERIFIED" and check.reasons == ()
    report = check.report
    assert report is not None
    assert report["G1"]["result"] == "PASS"
    assert report["G2"]["result"] == "PASS"
    for point, entry in report["G1"]["positions"].items():
        assert entry["result"] == "PASS", (point, entry)
    assert report["eligibility"]["result"] == "ELIGIBLE"
    assert [u["field"] for u in result.contract["unknowns"]] == ["origin.blas"]
    assert result.contract["origin"]["synthetic"] is False
    assert result.session_id == "session-1" and result.contract["session_id"] == "session-1"


def test_t_u65_01_without_a_recorded_hash_the_contract_is_still_verified() -> None:
    check = verify(build(make_capture()))
    assert check.status == "VERIFIED" and check.hash_source == "recomputed_no_recorded"


@pytest.mark.parametrize("kind", KINDS)
def test_the_record_of_the_capture_has_what_a_record_needs(kind: str) -> None:
    capture = make_capture(kind=kind)
    result = build(capture)
    assert tuple(result.capture_block) == CAPTURE_KEYS or set(result.capture_block) == set(
        CAPTURE_KEYS
    )
    assert result.capture_block["n_in"] == 6 and result.capture_block["n_gen"] == 4
    assert result.capture_block["points"] == list(capture.points)
    assert result.capture_block["adapter"] == capture.adapter
    assert result.capture_block["prompt_ids_sha256"] == capture.provenance.prompt_ids_sha256
    # build_record (U6a, not changed) takes it as it is
    assert result.contract is not None
    record = build_record(
        session_id="session-1",
        observation_id="obs-1",
        contract_session_id="session-1",
        capture=result.capture_block,
        contract_status="VERIFIED",
        contract_sha256=sha256_of_document(result.contract),
        hash_source="recomputed_no_recorded",
        refusals=[],
        positions={p: position_entry("enveloped", artifact_id=f"a-{p}") for p in capture.points},
    )
    assert record["outcome"] == "enveloped"


def test_capture_extra_of_a_built_contract() -> None:
    capture = make_capture()
    capture = with_provenance(
        capture,
        sampler={"temperature": 0.7, "top_k": 1},
        chat_template_kwargs={"enable_thinking": False},
    )
    result = build(capture)
    extra = result.capture_extra
    assert extra is not None and extra["version"] == "nvs.capture-extra/1"
    values = extra["values"]
    assert values["sampler"] == {"temperature": 0.7, "top_k": 1}
    assert values["max_tokens"] == 8
    assert values["chat_template_kwargs"] == {"enable_thinking": False}
    assert values["capture_unknowns"] == []
    assert values["raw_encoding"] == {
        "created_at_ms": 0,
        "created_at_ms_meaning": "fixed_sentinel_not_a_time",
    }
    assert values["recorded_at"] == "2026-11-16T09:30:15+09:00"
    assert values["adapter_notes"] == {
        "capabilities_phase": "post_capture",
        "observed_state_dtype": "float16",
        "version_policy": "package_version_only",
    }
    assert values["origin_facts_unknowns"] == []
    assert "origin" not in values  # the origin of a built contract is in the contract
    assert extra["unknowns"] == []


# ---- T-U65-13 --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_t_u65_13_the_record_names_the_trajectory_id_not_the_file(kind: str) -> None:
    capture = make_capture(kind=kind)
    result = build(capture)
    assert result.contract is not None
    for position in result.contract["positions"]:
        point = position["point"]
        rows = np.ascontiguousarray(capture.states[point], dtype="<f4").tobytes()
        assert position["trajectory_id"] == hashlib.sha256(rows).hexdigest()
        assert result.instrument_records[point]["raw_sha256"] == position["trajectory_id"]
        assert hashlib.sha256(result.raw_bytes[point]).hexdigest() != position["trajectory_id"]


def test_t_u65_13_a_record_made_with_the_hash_of_the_whole_file_fails_g2() -> None:
    result = build(make_capture())
    assert result.contract is not None
    point = "final_norm"
    wrong = copy.deepcopy(result.instrument_records)
    wrong[point]["raw_sha256"] = hashlib.sha256(result.raw_bytes[point]).hexdigest()
    check = check_contract_v2(result.contract, raw_bytes=result.raw_bytes, instrument_records=wrong)
    assert check.report is not None
    assert check.report["G2"]["result"] == "FAIL"
    assert verify(result).report["G2"]["result"] == "PASS"  # type: ignore[index]


def test_the_instrument_measures_what_was_read_back() -> None:
    assert EDGE is not None
    capture = make_capture()
    result = build(capture)
    decoded, _header = EDGE.decode_raw(result.raw_bytes["final_norm"], dimension=None)
    again = EDGE.instrument_record(decoded, result.instrument_records["final_norm"]["raw_sha256"])
    assert again == result.instrument_records["final_norm"]


def test_the_contract_hash_changes_with_the_session_id_only_by_that() -> None:
    capture = make_capture()
    one, two = build(capture, session_id="a"), build(capture, session_id="b")
    assert one.contract is not None and two.contract is not None
    assert sha256_of_document(one.contract) != sha256_of_document(two.contract)
    assert {**one.contract, "session_id": "b"} == two.contract


# ---- refusals: T-U65-16 and the rest of 6.1 --------------------------------------------------


def assert_nothing_made(result: ContractBuild) -> None:
    assert result.contract is None
    assert result.raw_bytes == {} and result.instrument_records == {}
    assert result.refusals and len(result.refusal_details) >= 1
    assert set(result.capture_block) == set(CAPTURE_KEYS)


def test_t_u65_16_a_space_collision_makes_nothing_and_still_has_a_record() -> None:
    capture = make_capture(kind="qwen")
    clashing = dataclasses.replace(
        capture,
        row_rule=MappingProxyType({**capture.row_rule, "mlp_out:3": "qwen35-mlx-block-out/1"}),
    )
    result = build(clashing)
    assert_nothing_made(result)
    assert result.refusals == ("space_collision",)
    assert result.capture_extra is not None and "origin" in result.capture_extra["values"]
    assert result.capture_block["adapter"] == "qwen35-mlx-capture"


def test_t_u65_16_synthetic_is_recorded_and_the_session_is_refused_by_u6a() -> None:
    result = build(make_capture(), synthetic=True)
    assert result.contract is not None and result.contract["origin"]["synthetic"] is True
    check = verify(result)
    assert check.synthetic is True
    refusals = session_refusals(result.contract, check)
    assert refusals and "synthetic" in refusals[0]


def test_t_u65_16_the_inputs_are_not_changed() -> None:
    capture = make_capture(kind="qwen")
    facts = make_facts(capture)
    facts_before = copy.deepcopy(facts)
    states_before = {p: a.copy() for p, a in capture.states.items()}
    prompt_before, tokens_before = capture.prompt_ids.copy(), capture.token_ids.copy()
    build(capture, facts)
    assert facts == facts_before
    for point, array in capture.states.items():
        assert np.array_equal(array, states_before[point]) and not array.flags.writeable
    assert np.array_equal(capture.prompt_ids, prompt_before)
    assert np.array_equal(capture.token_ids, tokens_before)
    assert not capture.prompt_ids.flags.writeable


def test_an_empty_generation_is_refused_with_its_record() -> None:
    result = build(make_capture(n_gen=0))
    assert_nothing_made(result)
    assert result.refusals == ("empty_generation",)
    assert result.capture_block["n_gen"] == 0 and result.capture_block["n_in"] == 6
    assert result.capture_extra is not None
    values = result.capture_extra["values"]
    assert {"sampler", "max_tokens", "chat_template_kwargs", "origin", "recorded_at"} <= set(values)
    assert "raw_encoding" not in values  # no RAW was made


def test_a_bad_capture_is_refused_with_the_check_that_failed() -> None:
    capture = make_capture()
    broken = with_provenance(capture, prompt_ids_sha256=HASH_A)
    result = build(broken)
    assert_nothing_made(result)
    assert result.refusals == ("capture_invalid",)
    assert "V-5" in result.refusal_details[0]


def test_a_mismatch_between_the_capture_and_its_origin_facts_is_refused() -> None:
    capture = make_capture()
    result = build(capture, make_facts(capture, **{"origin.revision": "f" * 40}))
    assert_nothing_made(result)
    assert result.refusals == ("origin_mismatch",)
    assert result.refusal_details == ("origin_mismatch: origin.revision",)


def test_a_capture_that_is_not_the_last_of_its_adapter_is_refused() -> None:
    capture = make_capture()
    pair = CapturedWithOrigin(
        capture=capture,
        origin_facts=make_facts(capture),
        adapter_class="X",
        sequence=1,
        last_capture_check="not_last",
    )
    result = contract_from_captured(pair, session_id="s", synthetic=False, clock=clock)
    assert_nothing_made(result)
    assert result.refusal_details == ("origin_mismatch: capture_not_last",)
    same = dataclasses.replace(pair, last_capture_check="same")
    assert contract_from_captured(same, session_id="s", synthetic=False, clock=clock).contract


def test_the_edge_not_being_usable_is_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture_edge, "_EDGE", None)
    monkeypatch.setattr(
        capture_edge, "_FAILURE", capture_edge.EdgeFailure("edge_unavailable", "nvs_edge.raw_nd")
    )
    result = build(make_capture())
    assert_nothing_made(result)
    assert result.refusals == ("edge_unavailable",)


def test_a_raw_file_that_does_not_read_back_as_its_rows_is_not_buildable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert EDGE is not None

    def corrupt(data: bytes, *, dimension: int | None) -> Any:
        states, header = EDGE.decode_raw(data, dimension=dimension)
        states[0, 0] += 1.0
        return states, header

    monkeypatch.setattr(capture_edge, "_EDGE", dataclasses.replace(EDGE, decode_raw=corrupt))
    result = build(make_capture())
    assert_nothing_made(result)
    assert result.refusals == ("contract_not_buildable",)
    assert "does not read back" in result.refusal_details[0]


def test_an_encoder_error_is_not_buildable_and_nothing_partial_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert EDGE is not None
    calls: list[str] = []

    def encode(states: Any, created_at_ms: int = 0, *, dimension: int) -> Any:
        calls.append("x")
        if len(calls) == 3:
            raise ValueError("encoder failed")
        return EDGE.encode_raw(states, created_at_ms, dimension=dimension)

    monkeypatch.setattr(capture_edge, "_EDGE", dataclasses.replace(EDGE, encode_raw=encode))
    result = build(make_capture(kind="qwen"))
    assert_nothing_made(result)  # two RAW files were made before the third failed: none is returned
    assert result.refusals == ("contract_not_buildable",)


def test_a_contract_error_of_the_edge_is_not_buildable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert EDGE is not None

    def refuse(**_: Any) -> dict[str, Any]:
        raise EDGE.contract_error("positions[0]: broken")

    monkeypatch.setattr(capture_edge, "_EDGE", dataclasses.replace(EDGE, build_contract_v2=refuse))
    result = build(make_capture())
    assert_nothing_made(result)
    assert result.refusal_details == ("contract_not_buildable: positions[0]: broken",)
    assert result.capture_extra is not None and "origin" in result.capture_extra["values"]


def test_a_capture_extra_that_does_not_fit_makes_no_contract() -> None:
    capture = make_capture()
    facts = make_facts(capture, version_policy="p" * 20000)  # adapter_notes is never dropped
    result = build(capture, facts)
    assert_nothing_made(result)
    assert result.refusals == ("capture_extra_overflow",)
    assert result.capture_extra is None
    assert "kept key too large: adapter_notes" in result.refusal_details[0]
    assert result.capture_block["n_gen"] == 4  # the record of the capture is still possible


def test_a_capture_extra_that_loses_a_big_value_still_builds_and_says_so() -> None:
    capture = make_capture()
    capture = dataclasses.replace(
        capture, unknowns=tuple(f"provenance.field{i:05d}" for i in range(2000))
    )
    result = build(capture)
    assert result.contract is not None and result.capture_extra is not None
    assert "capture_unknowns" not in result.capture_extra["values"]
    assert [u["reason"] for u in result.capture_extra["unknowns"]] == ["too_large"]
    assert result.capture_extra["unknowns"][0]["field"] == "capture_unknowns"


def test_the_clock_gives_the_time_with_its_offset() -> None:
    result = build(make_capture())
    assert result.capture_extra is not None
    assert result.capture_extra["values"]["recorded_at"] == CLOCK_TIME.isoformat(timespec="seconds")
    default = contract_from_capture(
        make_capture(), make_facts(make_capture()), session_id="s", synthetic=False
    )
    stamp = datetime.fromisoformat(default.capture_extra["values"]["recorded_at"])  # type: ignore[index]
    assert stamp.utcoffset() is not None
