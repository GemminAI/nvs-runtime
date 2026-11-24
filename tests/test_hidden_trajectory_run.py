"""U6-5d: ``run_capture_session`` -- from a captured pair to envelopes, the Gate and the record.

T-U65-01 (whole), T-U65-11, T-U65-16, 6.1 (a record is always returned), 6.3. The Adapters are
fakes (``u65_support``); no model is loaded. The Gate tests need ``SENSOS_TOOLS`` and ``GATE_PY``
(as ``test_hidden_trajectory_gate.py``) and are skipped, with the reason, without them. A Gate
that is a Python function stands in for it in the other tests.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

import pytest
from nvs_observation import capture_edge
from nvs_observation.adapter_session import AdapterSession, CapturedWithOrigin
from test_hidden_trajectory_envelope import Ports
from test_hidden_trajectory_gate import EXAMPLE, _skip_reason
from u65_support import (
    HASH_A,
    FakeCapture,
    make_capture,
    make_facts,
    unknown,
    with_provenance,
)

from nvs_runtime.hidden_trajectory import run as run_module
from nvs_runtime.hidden_trajectory.envelope import ParentRef
from nvs_runtime.hidden_trajectory.gate_client import (
    GateUnavailable,
    LookupMissing,
    ParentAbsent,
    ParentEnvelopeMissing,
    ParentMismatch,
)
from nvs_runtime.hidden_trajectory.record_schema import validate_record
from nvs_runtime.hidden_trajectory.run import (
    GATE_ERROR,
    STAGE_BUILT,
    STAGE_EMPTY_GENERATION,
    STAGE_REFUSED,
    RunResult,
    run_capture_session,
    with_capture_extra,
)

EDGE, _ = capture_edge.current()
pytestmark = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")

CLOCK_TIME = datetime(2026, 11, 16, 9, 30, 15, tzinfo=timezone(timedelta(hours=9)))
EXTRA_KEYS = {
    "sampler",
    "max_tokens",
    "chat_template_kwargs",
    "adapter_notes",
    "origin_facts_unknowns",
    "raw_encoding",
    "recorded_at",
}


def clock() -> datetime:
    return CLOCK_TIME


class FakeAdapter:
    """What ``AdapterSession`` needs: ``capture``, ``origin_facts`` and ``_last_capture``."""

    def __init__(self, capture: FakeCapture, facts: dict[str, Any] | None = None) -> None:
        self._capture, self._facts = capture, facts
        self._last_capture: Any = None

    def capture(self, messages: Any, request: Any, **kwargs: Any) -> FakeCapture:
        self._last_capture = self._capture
        return self._capture

    def origin_facts(self) -> dict[str, Any]:
        return make_facts(self._capture) if self._facts is None else self._facts


def pair(capture: FakeCapture, facts: dict[str, Any] | None = None) -> CapturedWithOrigin:
    return AdapterSession(FakeAdapter(capture, facts)).capture_with_origin(
        [], {}, model_build_id="m", compute_environment="e"
    )


def stub_verdict(**changes: Any) -> dict[str, Any]:
    """The part of a ``GateVerdict`` that ``run_capture_session`` reads."""
    verdict: dict[str, Any] = {"result": "PASS", "violations": [], "unverifiable": []}
    verdict.update(changes)
    return verdict


class StubGate:
    """A Gate that is a function: it records its calls and answers with a ``GateVerdict``."""

    def __init__(self, verdict: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, bytes], Any]] = []
        self.lookups: list[Any] = []
        self.stored: list[Any] = []
        self.verdict = verdict or stub_verdict()

    def __call__(
        self, envelope: Any, *, lookup: Any, stored: Any, blobs: Any, contract_facts: Any
    ) -> dict[str, Any]:
        self.calls.append((envelope, dict(blobs), contract_facts))
        self.lookups.append(lookup)
        self.stored.append(stored)
        return self.verdict


def run(
    captured: CapturedWithOrigin,
    ports: Ports | None = None,
    gate: Any = None,
    **kwargs: Any,
) -> RunResult:
    ports = ports or Ports()
    kwargs.setdefault("synthetic", False)
    return run_capture_session(
        captured,
        contract_session_id="synthetic-session-1",
        ids=ports.ids,
        space_id_of=ports.space_id_of,
        seal=ports.seal,
        gate=gate,
        clock=clock,
        **kwargs,
    )


def assert_valid(result: RunResult) -> None:
    assert validate_record(result.record) == []


# ---- a contract was built -----------------------------------------------------------------


@pytest.mark.parametrize(("kind", "count"), [("llama", 1), ("qwen", 5)])
def test_the_pair_goes_through_to_envelopes_the_gate_and_the_record(kind: str, count: int) -> None:
    ports, gate = Ports(), StubGate()
    result = run(pair(make_capture(kind=kind)), ports, gate)
    assert result.stage == STAGE_BUILT and result.contract is not None
    assert len(result.envelopes) == count
    assert len(gate.calls) == count and len(ports.sealed) == count
    assert_valid(result)
    record = result.record
    assert record["outcome"] == "enveloped" and record["contract"]["status"] == "VERIFIED"
    assert record["refusals"] == []
    assert {e["gate"]["status"] for e in record["positions"].values()} == {"PASS"}
    assert EXTRA_KEYS <= set(record["capture_extra"]["values"])
    assert "origin" not in record["capture_extra"]["values"]  # the origin is in the contract
    _, blobs, facts = gate.calls[0]
    assert facts is result.contract_check.facts and facts is not None  # type: ignore[union-attr]
    assert set(blobs) == set(result.blobs)


def test_the_rows_are_the_raw_rows_and_the_check_got_the_raw() -> None:
    result = run(pair(make_capture(kind="qwen")))
    assert result.contract_check is not None and result.contract_check.status == "VERIFIED"
    assert result.contract_check.hash_source == "recomputed_no_recorded"
    from nvs_observation.contract_check import rows_from_raw

    rows = rows_from_raw(result.build.raw_bytes)
    assert set(rows) == set(result.envelopes) and len(rows) == 5
    for point, envelope in result.envelopes.items():
        assert result.blobs[envelope["payload"]["points"]["uri"]] == rows[point]


def test_without_a_gate_it_is_not_called() -> None:
    result = run(pair(make_capture()))
    assert result.verdicts == {} and result.envelopes
    assert {e["gate"]["status"] for e in result.record["positions"].values()} == {"not_called"}
    assert_valid(result)


def test_the_same_pair_gives_the_same_record_bytes() -> None:
    first = run(pair(make_capture(kind="qwen")), Ports())
    second = run(pair(make_capture(kind="qwen")), Ports())
    assert first.record == second.record


def test_a_gate_that_raises_leaves_a_record_with_gate_error() -> None:
    def broken(
        envelope: Any, *, lookup: Any, stored: Any, blobs: Any, contract_facts: Any
    ) -> dict[str, Any]:
        raise RuntimeError("door closed")

    result = run(pair(make_capture()), gate=broken)
    assert_valid(result)
    entry = result.record["positions"]["final_norm"]
    assert entry["gate"]["status"] == GATE_ERROR
    assert entry["gate"]["unverifiable"] == [{"check": "gate_call", "reason": "RuntimeError"}]
    assert result.gate_errors == {"final_norm": "RuntimeError: door closed"}


def test_an_envelope_that_fails_the_envelope_rule_is_a_failure_in_the_record() -> None:
    gate = StubGate(
        stub_verdict(
            result="FAIL",
            violations=[{"layer": "L1", "code": "IC00-X", "message": "bad"}],
        )
    )
    result = run(pair(make_capture()), gate=gate)
    assert_valid(result)
    entry = result.record["positions"]["final_norm"]["gate"]
    assert entry == {
        "status": "FAIL",
        "violations": [{"code": "IC00-X", "message": "bad"}],
        "unverifiable": [],
    }


def test_the_inputs_are_not_changed() -> None:
    capture = make_capture(kind="qwen")
    facts = make_facts(capture)
    before = copy.deepcopy(facts)
    captured = pair(capture, facts)
    run(captured)
    assert facts == before == captured.origin_facts


# ---- 6.3: a contract with unknown values still gives envelopes ------------------------------


def unknown_runtime() -> CapturedWithOrigin:
    """``runtime.mlx`` is unknown in both sources (the capture's ``None``, the facts' marker)."""
    capture = with_provenance(
        make_capture(),
        runtime={"backend": "mlx", "mlx": None, "mlx_lm": "0.32.2", "device": "Device(gpu, 0)"},
    )
    facts = make_facts(capture, **{"origin.runtime.mlx": unknown("origin.runtime.mlx")})
    return pair(capture, facts)


def test_an_unknown_runtime_value_does_not_stop_the_envelopes() -> None:
    result = run(unknown_runtime(), gate=StubGate())
    assert_valid(result)
    assert result.contract is not None
    assert result.contract["origin"]["runtime"]["mlx"] == "unknown"
    assert list(result.envelopes) == ["final_norm"]
    payload = result.envelopes["final_norm"]["payload"]
    reasons = {u["field"]: u["reason"] for u in payload["unknowns"]}
    assert any("mlx" in field for field in reasons), reasons
    assert result.record["positions"]["final_norm"]["status"] == "enveloped"


# ---- 6.1: refusals; a record is always returned ----------------------------------------------


def broken_cases() -> dict[str, tuple[FakeCapture, dict[str, Any] | None, str]]:
    capture = make_capture(kind="qwen")
    clashing = dataclasses.replace(
        capture,
        row_rule=MappingProxyType({**capture.row_rule, "mlp_out:3": "qwen35-mlx-block-out/1"}),
    )
    plain = make_capture()
    return {
        "capture_invalid": (
            with_provenance(plain, prompt_ids_sha256=HASH_A),
            None,
            "capture_invalid",
        ),
        "origin_mismatch": (
            plain,
            make_facts(plain, **{"origin.revision": "f" * 40}),
            "origin_mismatch",
        ),
        "origin_facts_invalid": (plain, {"origin": "not a mapping"}, "origin_facts_invalid"),
        "space_collision": (clashing, None, "space_collision"),
    }


@pytest.mark.parametrize(
    "name", ["capture_invalid", "origin_mismatch", "origin_facts_invalid", "space_collision"]
)
def test_a_refused_capture_gives_a_record_and_no_later_stage(name: str) -> None:
    capture, facts, code = broken_cases()[name]
    ports, gate = Ports(), StubGate()
    result = run(pair(capture, facts), ports, gate)
    assert result.stage == STAGE_REFUSED and result.contract is None
    assert result.envelopes == {} and result.blobs == {} and result.contract_check is None
    assert ports.spaces == [] and ports.sealed == [] and gate.calls == []
    assert_valid(result)
    record = result.record
    assert record["outcome"] == "refused" and record["contract"]["status"] == "none"
    assert record["contract"]["contract_sha256"] is None
    assert record["refusals"] and record["refusals"][0].split(":")[0] == code
    assert set(record["positions"]) == set(record["capture"]["points"]) != set()
    assert {e["status"] for e in record["positions"].values()} == {"not_built"}
    assert {e["gate"]["status"] for e in record["positions"].values()} == {"not_called"}
    assert "origin" in record["capture_extra"]["values"]
    assert "raw_encoding" not in record["capture_extra"]["values"]


def test_a_capture_that_is_not_the_last_of_its_adapter_is_refused_end_to_end() -> None:
    class Interrupted(FakeAdapter):
        def capture(self, messages: Any, request: Any, **kwargs: Any) -> FakeCapture:
            self._last_capture = object()  # somebody else's capture came in between
            return self._capture

    captured = AdapterSession(Interrupted(make_capture())).capture_with_origin(
        [], {}, model_build_id="m", compute_environment="e"
    )
    assert captured.last_capture_check == "not_last"
    ports, gate = Ports(), StubGate()
    result = run(captured, ports, gate)
    assert_valid(result)
    assert result.record["refusals"] == ["origin_mismatch: capture_not_last"]
    assert ports.spaces == [] and gate.calls == []
    assert "origin" in result.record["capture_extra"]["values"]


def test_the_edge_not_being_usable_is_refused_with_a_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = pair(make_capture())
    monkeypatch.setattr(capture_edge, "_EDGE", None)
    monkeypatch.setattr(
        capture_edge, "_FAILURE", capture_edge.EdgeFailure("edge_unavailable", "nvs_edge.raw_nd")
    )
    ports, gate = Ports(), StubGate()
    result = run(captured, ports, gate)
    assert_valid(result)
    assert result.record["refusals"][0].startswith("edge_unavailable")
    assert result.record["contract"]["status"] == "none"
    assert ports.spaces == [] and gate.calls == []
    assert "origin" in result.record["capture_extra"]["values"]


def test_a_capture_extra_that_does_not_fit_gives_a_record_without_capture_extra() -> None:
    capture = make_capture()
    captured = pair(capture, make_facts(capture, version_policy="p" * 20000))
    ports, gate = Ports(), StubGate()
    result = run(captured, ports, gate)
    assert_valid(result)
    assert result.contract is None and result.stage == STAGE_REFUSED
    assert result.record["refusals"][0].startswith("capture_extra_overflow")
    assert "capture_extra" not in result.record
    assert ports.spaces == [] and gate.calls == []
    assert result.record["capture"]["n_gen"] == 4


def test_steps_out_of_range_gives_a_record() -> None:
    import numpy as np

    capture = make_capture(n_gen=1)
    # the check reads n_gen from the token ids; a stand-in of 65,536 ids is cheap enough
    big = np.arange(65_536, dtype=np.int64)
    big.setflags(write=False)
    huge = dataclasses.replace(capture, token_ids=big)
    result = run(pair(huge), Ports(), StubGate())
    assert_valid(result)
    assert result.contract is None and result.record["contract"]["status"] == "none"
    assert result.record["refusals"]


# ---- T-U65-11, T-U65-16 --------------------------------------------------------------------


def test_t_u65_11_an_empty_generation_has_its_record_with_the_origin() -> None:
    ports, gate = Ports(), StubGate()
    result = run(pair(make_capture(n_gen=0)), ports, gate)
    assert result.stage == STAGE_EMPTY_GENERATION and result.contract is None
    assert_valid(result)
    record = result.record
    assert record["outcome"] == "empty_generation" and record["contract"]["status"] == "none"
    assert record["refusals"] == []
    assert {"sampler", "max_tokens", "chat_template_kwargs", "origin"} <= set(
        record["capture_extra"]["values"]
    )
    assert {e["status"] for e in record["positions"].values()} == {"not_built"}
    assert ports.spaces == [] and gate.calls == []


def test_t_u65_16_synthetic_has_a_contract_and_a_record_but_no_envelope() -> None:
    ports, gate = Ports(), StubGate()
    result = run(pair(make_capture()), ports, gate, synthetic=True)
    assert result.contract is not None and result.contract["origin"]["synthetic"] is True
    assert result.envelopes == {} and ports.spaces == [] and ports.sealed == []
    assert gate.calls == []
    assert_valid(result)
    assert any(r.startswith("synthetic") for r in result.record["refusals"])
    assert result.record["outcome"] == "refused"
    assert {e["status"] for e in result.record["positions"].values()} == {"not_built"}
    assert "capture_extra" in result.record


def test_with_capture_extra_does_not_change_its_input_and_checks_the_result() -> None:
    result = run(pair(make_capture()))
    record = copy.deepcopy(result.record)
    again = with_capture_extra(
        record, {"version": "nvs.capture-extra/1", "values": {}, "unknowns": []}
    )
    assert record == result.record and again["capture_extra"]["values"] == {}
    with pytest.raises(ValueError, match="does not fit"):
        with_capture_extra(record, {"version": "other", "values": {}, "unknowns": []})


# ---- the real Gate -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def door() -> Any:
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    sys.path.insert(0, str(EXAMPLE))
    try:
        from run_door import run_through_door
        from sensos_door import SensosDoor
    finally:
        sys.path.remove(str(EXAMPLE))
    return SensosDoor.from_environment(), run_through_door


def through_door(door: Any, captured: CapturedWithOrigin, item: str, **kwargs: Any) -> RunResult:
    kwargs.setdefault("synthetic", False)
    sensos, run_through_door = door
    result: RunResult = run_through_door(
        sensos,
        captured,
        contract_session_id=f"u65d-{item}",
        spec="0" * 64,
        item_id=item,
        produced_at="2026-11-16T00:00:00Z",
        clock=clock,
        **kwargs,
    )
    return result


@pytest.mark.parametrize(("kind", "count"), [("llama", 1), ("qwen", 5)])
def test_t_u65_01_whole_through_the_real_gate(door: Any, kind: str, count: int) -> None:
    result = through_door(door, pair(make_capture(kind=kind)), f"whole-{kind}")
    assert result.stage == STAGE_BUILT and len(result.envelopes) == count
    assert_valid(result)
    assert set(result.verdicts) == set(result.envelopes)
    for point, verdict in result.verdicts.items():
        assert verdict["layers"]["L1"] == {"status": "PASS", "codes": []}, (point, verdict)
        assert verdict["violations"] == [], (point, verdict["violations"])
        assert verdict["result"] in ("PASS", "PASS_UNVERIFIED"), (point, verdict)
        assert verdict["verdict_version"] == "gate-verdict/1"
        assert result.record["positions"][point]["gate"]["status"] == verdict["result"]
    values = result.record["capture_extra"]["values"]
    assert EXTRA_KEYS <= set(values) and "origin" not in values
    assert result.record["outcome"] == "enveloped"


def test_a_contract_with_an_unknown_value_passes_the_real_gate(door: Any) -> None:
    result = through_door(door, unknown_runtime(), "unknown-mlx")
    assert_valid(result)
    assert list(result.envelopes) == ["final_norm"]
    verdict = result.verdicts["final_norm"]
    assert verdict["layers"]["L1"]["status"] == "PASS" and verdict["violations"] == []
    payload = result.envelopes["final_norm"]["payload"]
    assert any("mlx" in u["field"] for u in payload["unknowns"])


def test_without_a_gate_through_the_door_the_gate_is_not_called(door: Any) -> None:
    result = through_door(door, pair(make_capture()), "no-gate", call_gate=False)
    assert result.verdicts == {} and result.envelopes
    assert {e["gate"]["status"] for e in result.record["positions"].values()} == {"not_called"}


def test_a_refusal_through_the_door_still_has_its_record(door: Any) -> None:
    capture = make_capture()
    facts = make_facts(capture, **{"origin.revision": "f" * 40})
    result = through_door(door, pair(capture, facts), "refused")
    assert_valid(result)
    assert result.stage == STAGE_REFUSED and result.envelopes == {} and result.verdicts == {}
    assert result.record["contract"]["status"] == "none"
    assert result.record["refusals"] == ["origin_mismatch: origin.revision"]


# ---- the parent's lookup (U6T-1c; T-T1-01, T-T1-02) ------------------------------------------

PARENT_ARTIFACT = "art:" + "7" * 64


def parent_ref(observation_id: str) -> ParentRef:
    return ParentRef(PARENT_ARTIFACT, "cnt:" + "8" * 64, 0, observation_id)


def parent_held(parent: ParentRef, **changes: Any) -> dict[str, Any]:
    """The parent's stored envelope as far as the run reads it (the fields of ``ParentRef``)."""
    held: dict[str, Any] = {
        "artifact_id": parent.artifact_id,
        "content_id": parent.content_id,
        "seq": parent.seq,
        "observation_id": parent.observation_id,
    }
    held.update(changes)
    return held


def test_the_lookup_of_the_parent_and_its_envelope_reach_the_gate() -> None:
    ports, gate = Ports(), StubGate()
    parent = parent_ref(ports.ids.observation_id)
    held = parent_held(parent)
    result = run(pair(make_capture()), ports, gate, parent=parent, parent_envelope=held)
    assert len(gate.calls) == 1 and result.stage == STAGE_BUILT
    assert gate.lookups[0] == {
        PARENT_ARTIFACT: {
            "content_id": parent.content_id,
            "seq": 0,
            "observation_id": ports.ids.observation_id,
        }
    }
    assert gate.stored[0] == {PARENT_ARTIFACT: {"envelope": held}}
    assert result.envelopes["final_norm"]["parents"][0]["artifact_ids"] == [PARENT_ARTIFACT]


def test_without_a_parent_the_lookup_and_stored_are_empty() -> None:
    gate = StubGate()
    run(pair(make_capture()), Ports(), gate)
    assert gate.lookups == [{}] and gate.stored == [{}]


def test_a_parent_whose_envelope_was_not_fetched_stops_before_the_gate() -> None:
    ports, gate = Ports(), StubGate()
    parent = parent_ref(ports.ids.observation_id)
    with pytest.raises(ParentEnvelopeMissing) as raised:
        run(pair(make_capture()), ports, gate, parent=parent)
    assert raised.value.artifact_id == PARENT_ARTIFACT
    assert gate.calls == []


def test_without_a_gate_an_unfetched_parent_envelope_is_no_error() -> None:
    ports = Ports()
    result = run(pair(make_capture()), ports, parent=parent_ref(ports.ids.observation_id))
    assert result.stage == STAGE_BUILT and result.verdicts == {}


def test_an_empty_mapping_is_not_an_absent_parent_and_not_an_envelope() -> None:
    ports, gate = Ports(), StubGate()
    parent = parent_ref(ports.ids.observation_id)
    with pytest.raises(ParentMismatch) as raised:
        run(pair(make_capture()), ports, gate, parent=parent, parent_envelope={})
    assert set(raised.value.fields) == {"artifact_id", "content_id", "seq", "observation_id"}
    assert gate.calls == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("artifact_id", "art:" + "1" * 64),
        ("content_id", "cnt:" + "1" * 64),
        ("seq", 5),
        ("observation_id", "obs:other"),
    ],
)
def test_a_parent_envelope_that_is_not_the_parents_is_a_mismatch_and_the_gate_is_not_called(
    name: str, value: Any
) -> None:
    ports, gate = Ports(), StubGate()
    parent = parent_ref(ports.ids.observation_id)
    held = parent_held(parent, **{name: value})
    with pytest.raises(ParentMismatch) as raised:
        run(pair(make_capture()), ports, gate, parent=parent, parent_envelope=held)
    assert raised.value.fields == (name,)
    assert raised.value.differences[name] == (getattr(parent, name), value)
    assert gate.calls == []


def test_a_parent_that_is_confirmed_absent_gives_a_refusal_record_and_no_gate_call() -> None:
    ports, gate = Ports(), StubGate()
    parent = parent_ref(ports.ids.observation_id)
    absent = ParentAbsent(PARENT_ARTIFACT, "IC04Store")
    result = run(pair(make_capture()), ports, gate, parent=parent, parent_envelope=absent)
    assert gate.calls == []
    assert_valid(result)
    assert result.stage == STAGE_REFUSED and result.envelopes == {} and result.verdicts == {}
    record = result.record
    assert record["outcome"] == "refused"
    assert record["refusals"] == [f"parent_not_stored: {PARENT_ARTIFACT}"]
    assert record["positions"]
    for entry in record["positions"].values():
        assert entry["status"] == "not_built" and entry["reasons"] == ["parent_not_stored"]
        assert entry["artifact_id"] is None and entry["gate"]["status"] == "not_called"
    assert record["capture"]["n_gen"] > 0 and "capture_extra" in record


def test_an_absent_parent_that_is_not_the_runs_parent_is_a_mismatch() -> None:
    ports, gate = Ports(), StubGate()
    absent = ParentAbsent("art:" + "1" * 64, "IC04Store")
    with pytest.raises(ParentMismatch) as raised:
        run(
            pair(make_capture()),
            ports,
            gate,
            parent=parent_ref(ports.ids.observation_id),
            parent_envelope=absent,
        )
    assert raised.value.fields == ("artifact_id",) and gate.calls == []


def test_a_parent_envelope_without_a_parent_is_not_used() -> None:
    gate = StubGate()
    run(pair(make_capture()), Ports(), gate, parent_envelope={"observation_id": "x"})
    assert gate.lookups == [{}] and gate.stored == [{}]


def test_t_t1_02_a_parent_the_lookup_does_not_name_stops_before_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports, gate = Ports(), StubGate()
    parent = parent_ref(ports.ids.observation_id)
    monkeypatch.setattr(run_module, "_gate_inputs", lambda parent, envelope: ({}, {}))
    with pytest.raises(LookupMissing) as raised:
        run(pair(make_capture()), ports, gate, parent=parent, parent_envelope=parent_held(parent))
    assert raised.value.missing == (PARENT_ARTIFACT,)
    assert gate.calls == []


def test_a_gate_that_raises_a_gate_error_is_a_gate_error_in_the_record_not_an_exception() -> None:
    def down(
        envelope: Any, *, lookup: Any, stored: Any, blobs: Any, contract_facts: Any
    ) -> dict[str, Any]:
        raise GateUnavailable("the child exited 1")

    result = run(pair(make_capture()), gate=down)
    assert result.record["positions"]["final_norm"]["gate"]["status"] == GATE_ERROR
    assert result.gate_errors == {"final_norm": "GateUnavailable: the child exited 1"}
    assert result.verdicts == {}


def test_t_t1_01_through_the_real_gate_a_child_with_a_parent_passes_the_envelope_rule(
    door: Any,
) -> None:
    sensos, run_through_door = door
    session_id = sensos.session_id(
        spec="0" * 64, item_id="parent", condition="run", replicate=0, rng_seed=0
    )
    observation_id = sensos.observation_id(session_id, 0)
    parent = parent_ref(observation_id)

    def go(**kwargs: Any) -> RunResult:
        result: RunResult = run_through_door(
            sensos,
            pair(make_capture()),
            contract_session_id="u65d-parent",
            synthetic=False,
            spec="0" * 64,
            item_id="parent",
            produced_at="2026-11-16T00:00:00Z",
            clock=clock,
            parent=parent,
            **kwargs,
        )
        return result

    # no parent envelope: nothing is sent to the Gate
    with pytest.raises(ParentEnvelopeMissing):
        go()
    # with the parent's envelope the envelope rule (L1) and the payload rule (L2, HT-19) both pass
    with_parent = go(parent_envelope=parent_held(parent))
    passed = with_parent.verdicts["final_norm"]
    assert passed["layers"]["L1"] == {"status": "PASS", "codes": []}, passed["violations"]
    assert passed["layers"]["L2"]["status"] == "PASS", passed["violations"]
    assert passed["result"] in ("PASS", "PASS_UNVERIFIED"), passed["violations"]
    assert passed["inputs"]["lookup"][PARENT_ARTIFACT]["observation_id"] == observation_id
    assert passed["inputs"]["stored"] == [PARENT_ARTIFACT]
    assert_valid(with_parent)
