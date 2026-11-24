"""``run_capture_session``: from one captured pair to envelopes, the Gate and the capture record
(design U6-5 v0.5, 3.2, 6.1, 6.3).

The stages, in order; a stage that refuses stops the later ones and the reason goes to the record:

1. ``contract_from_captured`` -- the contract, its RAW files and instrument records.
2. ``check_contract_v2`` -- with ``raw_bytes`` and ``instrument_records``. The hash is not compared
   with another one (``hash_source`` says so; DC-U6-4 (a)).
3. ``build_envelopes`` -- ``rows`` are the RAW's rows.
4. The Gate -- only when ``gate`` is given; without it every ``gate`` of the record stays
   ``not_called``. ``gate`` is ``GateClient.verdict``: it gets the lookup of the parent (made from
   ``parent``) and the stored envelopes (the parent's envelope, which ``parent_envelope`` must be
   when there is a parent), and answers with a ``GateVerdict``.
5. The record, with ``capture_extra``.

**A record is always returned** (6.1), by one of three paths:

* a contract was built: the record of ``build_envelopes`` with ``capture_extra`` added (the record
  is copied, the key is written, the whole is checked against the record schema);
* ``n_gen == 0``: ``empty_generation_record``;
* any other refusal: ``build_record`` with ``contract.status`` ``none``, the reasons in
  ``refusals`` (each begins with its code), every requested point ``not_built``, and
  ``capture_extra`` (which carries ``origin`` here).

Not here: the sensos contracts. ``space_id_of``, ``seal`` and ``gate`` are given by the caller
(by design, no cross-imports); see ``examples/hidden_trajectory`` for the door that reaches
them. Nothing about the
capture is filled in: a refused capture gives no envelope.

An exception of ``space_id_of``, ``seal`` or an exception of ``contract_from_captured`` itself
is a failure of the caller's side, not a refusal of the capture, and is raised. So is
``LookupMissing`` (an envelope with a parent the lookup does not name; DC-T1-3), raised before the
Gate is called. So are ``ParentEnvelopeMissing`` (a parent, and neither its envelope nor a
``ParentAbsent`` given) and ``ParentMismatch`` (the envelope given is not the parent's): both are
mistakes or disagreements of the caller's data, and no record is returned for them. A parent that
the caller *confirmed absent* (``ParentAbsent``) is different: it is a fact about the capture, so
a record **is** returned (a refusal: every position ``not_built``, reason ``parent_not_stored``,
``refusals`` with the same reason and the ``artifact_id``) and the Gate is not called. An
exception of ``gate`` is not: the position's ``gate`` is then written as
``gate_error`` (see ``GATE_ERROR``): ``GateUnavailable``, ``CatalogMismatch`` and the like are not
a judgment on the envelope, and no ``GateVerdict`` is made.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from nvs_observation.adapter_session import CapturedWithOrigin
from nvs_observation.contract_build import ContractBuild, contract_from_captured
from nvs_observation.contract_check import ContractCheck, check_contract_v2, rows_from_raw

from nvs_runtime.hidden_trajectory import record as rec
from nvs_runtime.hidden_trajectory.envelope import (
    BuildResult,
    EnvelopeIds,
    ParentRef,
    build_envelopes,
)
from nvs_runtime.hidden_trajectory.gate_client import (
    LookupMissing,
    ParentAbsent,
    ParentEnvelopeMissing,
    ParentMismatch,
    missing_lookup,
)
from nvs_runtime.hidden_trajectory.payload import POLICY_ALL_ROWS
from nvs_runtime.hidden_trajectory.record_schema import validate_record

#: The fields of ``ParentRef`` that the parent's envelope must repeat.
_PARENT_FIELDS: Final = ("artifact_id", "content_id", "seq", "observation_id")

#: ``RunResult.stage``: where the run ended.
STAGE_REFUSED: Final = "refused"
STAGE_EMPTY_GENERATION: Final = "empty_generation"
STAGE_BUILT: Final = "built"

#: ``gate.status`` of a position whose Gate call raised; the exception's class is the reason.
GATE_ERROR: Final = "gate_error"
#: ``gate.status`` of an envelope with a violation (a ``GateVerdict`` whose ``result`` is ``FAIL``).
GATE_ENVELOPE_FAIL: Final = "FAIL"


def _now() -> datetime:
    return datetime.now().astimezone()


GateFunction = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class RunResult:
    """``record``: the capture record (always). ``envelopes``, ``blobs``: as ``BuildResult``.
    ``verdicts``: point -> the ``GateVerdict`` the Gate returned (empty without a Gate).
    ``contract``: the contract document, or ``None`` if none was built. ``contract_check``,
    ``build``: the earlier stages (``contract_check`` is ``None`` if no contract was built).
    ``stage``: where it ended.
    ``gate_errors``: point -> ``"<class>: <message>"`` for a Gate call that raised.
    """

    record: dict[str, Any]
    envelopes: dict[str, dict[str, Any]]
    blobs: dict[str, bytes]
    verdicts: dict[str, Mapping[str, Any]]
    contract: dict[str, Any] | None
    contract_check: ContractCheck | None
    build: ContractBuild
    stage: str
    gate_errors: dict[str, str] = field(default_factory=dict)


def with_capture_extra(
    record: Mapping[str, Any], capture_extra: Mapping[str, Any]
) -> dict[str, Any]:
    """A copy of ``record`` with ``capture_extra`` at its top level; ``ValueError`` if it does not
    fit the record schema. ``record`` is not changed."""
    out = copy.deepcopy(dict(record))
    out["capture_extra"] = copy.deepcopy(dict(capture_extra))
    violations = validate_record(out)
    if violations:
        raise ValueError(
            f"the record does not fit {rec.RECORD_VERSION}: {'; '.join(violations[:5])}"
        )
    return out


def _refusal_lines(build: ContractBuild) -> list[str]:
    """The reasons for ``refusals``: each detail begins with its code; a code without one stands
    alone."""
    lines = list(dict.fromkeys(build.refusal_details))
    for code in build.refusals:
        if not any(line == code or line.startswith(f"{code}:") for line in lines):
            lines.append(code)
    return lines


def _without_contract(build: ContractBuild, ids: EnvelopeIds) -> tuple[dict[str, Any], str]:
    """The record of a capture for which no contract was built (6.1)."""
    capture = build.capture_block
    if build.refusals == (rec.REASON_EMPTY_GENERATION,):
        record = rec.empty_generation_record(
            session_id=ids.session_id,
            observation_id=ids.observation_id,
            capture=capture,
            capture_extra=build.capture_extra,
        )
        return record, STAGE_EMPTY_GENERATION
    points = capture.get("points") or []
    record = rec.build_record(
        session_id=ids.session_id,
        observation_id=ids.observation_id,
        contract_session_id=None,
        capture=capture,
        contract_status=rec.CONTRACT_NONE,
        contract_sha256=None,
        hash_source=rec.CONTRACT_NONE,
        refusals=_refusal_lines(build),
        positions={
            point: rec.position_entry(rec.POSITION_NOT_BUILT, (rec.REASON_SESSION_REFUSED,))
            for point in dict.fromkeys(points)
        },
        capture_extra=build.capture_extra,
    )
    return record, STAGE_REFUSED


def _gate_fields(verdict: Mapping[str, Any]) -> tuple[str, list[Any], list[Any]]:
    """``(status, violations, unverifiable)`` of the record, from a ``GateVerdict`` (its
    ``result``, ``violations`` and ``unverifiable``). The record keeps the form it had before
    ``GateVerdict`` (``{code, message}`` and ``{check, reason}``): the layer, and the rest of the
    verdict, are in ``RunResult.verdicts``."""
    return (
        str(verdict["result"]),
        [{"code": v["code"], "message": v["message"]} for v in verdict.get("violations") or []],
        [{"check": u["check"], "reason": u["reason"]} for u in verdict.get("unverifiable") or []],
    )


def _gate_inputs(
    parent: ParentRef | None, parent_envelope: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(lookup, stored)`` for the Gate: the parent's three ids (``ParentRef`` has them, and they
    were read from the store); and, if given, the parent's envelope, which the payload rule of a
    hidden trajectory reads (HT-19). Both are empty without a parent."""
    if parent is None:
        return {}, {}
    lookup = {
        parent.artifact_id: {
            "content_id": parent.content_id,
            "seq": parent.seq,
            "observation_id": parent.observation_id,
        }
    }
    stored = {} if parent_envelope is None else {parent.artifact_id: {"envelope": parent_envelope}}
    return lookup, stored


def _parent_differences(
    parent: ParentRef, envelope: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """The fields of ``envelope`` (``artifact_id``, ``content_id``, ``seq``, ``observation_id``)
    that differ from ``parent``: name -> ``(expected, found)``. A missing field is ``None``."""
    out: dict[str, tuple[Any, Any]] = {}
    for name in _PARENT_FIELDS:
        expected, found = getattr(parent, name), envelope.get(name)
        if type(expected) is not type(found) or expected != found:
            out[name] = (expected, found)
    return out


def _parent_not_stored_record(
    absent: ParentAbsent, build: ContractBuild, check: ContractCheck, ids: EnvelopeIds
) -> dict[str, Any]:
    """The record of a capture whose parent is confirmed absent: the capture and the contract as
    checked, no envelope, every position ``not_built``."""
    document = build.contract
    contract_session = document.get("session_id") if isinstance(document, Mapping) else None
    capture = rec.capture_block_of_contract(document)
    reason = rec.REASON_PARENT_NOT_STORED
    record = rec.build_record(
        session_id=ids.session_id,
        observation_id=ids.observation_id,
        contract_session_id=contract_session if isinstance(contract_session, str) else None,
        capture=capture,
        contract_status=check.status,
        contract_sha256=check.contract_sha256,
        hash_source=check.hash_source,
        refusals=[f"{reason}: {absent.artifact_id}"],
        positions={
            point: rec.position_entry(rec.POSITION_NOT_BUILT, (reason,))
            for point in dict.fromkeys(capture["points"])
        },
    )
    return with_capture_extra(record, build.capture_extra or {})


def run_capture_session(
    captured: CapturedWithOrigin,
    *,
    contract_session_id: str,
    synthetic: bool,
    ids: EnvelopeIds,
    space_id_of: Callable[[Mapping[str, Any]], str],
    seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
    gate: GateFunction | None = None,
    parent: ParentRef | None = None,
    parent_envelope: Mapping[str, Any] | ParentAbsent | None = None,
    producer_build: str | None = None,
    session_extensions: Mapping[str, Any] | None = None,
    extensions: Mapping[str, Mapping[str, Any]] | None = None,
    copied: bool = False,
    policy_name: str = POLICY_ALL_ROWS,
    clock: Callable[[], datetime] = _now,
) -> RunResult:
    """Run the stages on ``captured`` (``AdapterSession.capture_with_origin``'s result; L-c).

    ``contract_session_id`` is what the contract records as its ``session_id``; ``ids`` are the
    runtime's ids of the envelopes and the record. ``gate(envelope, *, lookup, stored, blobs,
    contract_facts)`` returns a ``GateVerdict`` (``GateClient.verdict`` does); without it the Gate
    is not called. The rest is ``build_envelopes``'s.

    ``parent_envelope`` says what the caller knows of ``parent`` (nothing of it matters without a
    parent, and nothing is made of it then: ``stored`` is empty). In U6-6 the envelope is read
    from the IC04Store (the Formal Store) by the caller; it is not implemented here.

    * the parent's stored envelope: it must match ``parent`` (``artifact_id``, ``content_id``,
      ``seq``, ``observation_id``) or ``ParentMismatch`` is raised; it is then the Gate's
      ``stored``;
    * ``ParentAbsent(artifact_id, source)``: the caller asked the store and the parent is not
      there. The Gate is not called and a refusal record is returned (see the module docstring);
    * ``None``: not fetched. With a parent and a ``gate``, ``ParentEnvelopeMissing`` is raised
      before the Gate. ``None`` is never read as "absent", and an empty mapping is not an envelope.
    """
    build = contract_from_captured(
        captured, session_id=contract_session_id, synthetic=synthetic, clock=clock
    )
    if build.contract is None or build.capture_extra is None:
        record, stage = _without_contract(build, ids)
        return RunResult(record, {}, {}, {}, None, None, build, stage)

    check = check_contract_v2(
        build.contract,
        raw_bytes=build.raw_bytes,
        instrument_records=build.instrument_records,
    )
    if parent is not None and isinstance(parent_envelope, ParentAbsent):
        if parent_envelope.artifact_id != parent.artifact_id:
            raise ParentMismatch({"artifact_id": (parent.artifact_id, parent_envelope.artifact_id)})
        return RunResult(
            _parent_not_stored_record(parent_envelope, build, check, ids),
            {},
            {},
            {},
            build.contract,
            check,
            build,
            STAGE_REFUSED,
        )
    built: BuildResult = build_envelopes(
        build.contract,
        check,
        ids=ids,
        space_id_of=space_id_of,
        seal=seal,
        rows=rows_from_raw(build.raw_bytes),
        parent=parent,
        producer_build=producer_build,
        session_extensions=session_extensions,
        extensions=extensions,
        copied=copied,
        policy_name=policy_name,
    )
    record = with_capture_extra(built.record, build.capture_extra)
    stored_parent: Mapping[str, Any] | None = None
    if parent is not None and parent_envelope is not None:
        assert not isinstance(parent_envelope, ParentAbsent)
        differences = _parent_differences(parent, parent_envelope)
        if differences:
            raise ParentMismatch(differences)
        stored_parent = parent_envelope
    if parent is not None and gate is not None and stored_parent is None:
        raise ParentEnvelopeMissing(parent.artifact_id)
    verdicts: dict[str, Mapping[str, Any]] = {}
    gate_errors: dict[str, str] = {}
    if gate is not None:
        lookup, stored = _gate_inputs(parent, stored_parent)
        for point, envelope in built.envelopes.items():
            if missing_lookup(envelope, lookup):
                raise LookupMissing(missing_lookup(envelope, lookup))
            try:
                verdict = gate(
                    envelope,
                    lookup=lookup,
                    stored=stored,
                    blobs=built.blobs,
                    contract_facts=check.facts,
                )
                status, violations, unverifiable = _gate_fields(verdict)
            except Exception as error:  # noqa: BLE001 -- the record is returned all the same
                gate_errors[point] = f"{type(error).__name__}: {error}"
                status = GATE_ERROR
                violations = []
                unverifiable = [{"check": "gate_call", "reason": type(error).__name__}]
            else:
                verdicts[point] = verdict
            record = rec.with_gate_result(
                record, point, status=status, violations=violations, unverifiable=unverifiable
            )
    return RunResult(
        record=record,
        envelopes=built.envelopes,
        blobs=built.blobs,
        verdicts=verdicts,
        contract=build.contract,
        contract_check=check,
        build=build,
        stage=STAGE_BUILT,
        gate_errors=gate_errors,
    )
