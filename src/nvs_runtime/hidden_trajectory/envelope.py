"""Envelopes ``nvs.hidden-trajectory`` 0.1.0 (IC-00 1.1.0) from a RawObservationContract v2:
whether to build one, and building it -- one envelope per position.

Whether an envelope is built:

* for the whole capture (no position is built): the contract is synthetic, or does not say
  whether it is; it failed verification; nvs_edge could not check it and its header is not that
  of a contract v2; or it holds a value no payload can hold (``payload.session_obstacles``);
* for one position (the others are still built): the verification of its RAW failed, or the
  position holds a value its payload cannot hold (``payload.position_obstacles``).

Nothing is filled in to get past a reason; every reason goes to the capture record.

The space id and the artifact ids are sensos's rules. They are not computed here: the caller
gives the two functions (``space_id_of``, ``seal``), which reach the sensos contracts.

Imports no numpy, no nvs_edge and no sensos contracts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from nvs_observation.contract_check import NOT_VERIFIED, REJECTED, VERIFIED, ContractCheck

from nvs_runtime.__version__ import __version__
from nvs_runtime.hidden_trajectory import record as rec
from nvs_runtime.hidden_trajectory.payload import (
    PAYLOAD_SCHEMA_ID,
    PAYLOAD_SCHEMA_VERSION,
    POLICY_ALL_ROWS,
    build_payload,
    position_obstacles,
    session_obstacles,
    token_index_bytes,
)

IC00_VERSION: Final = "1.1.0"
KIND: Final = "observation"
PRODUCER_NAME: Final = "sensos-inference"
PARENT_ROLE: Final = "generated_text"
CONTRACT_NAME: Final = "RawObservationContract"
CONTRACT_VERSION: Final = 2

EXT_CONTRACT_SESSION: Final = "x-contract-session-id"
EXT_ASSEMBLED_BY: Final = "x-assembled-by"
ASSEMBLED_BY: Final = f"nvs-runtime/{__version__}"

REASON_SYNTHETIC: Final = "synthetic"
REASON_SYNTHETIC_NOT_STATED: Final = "synthetic_not_stated"
REASON_NOT_A_CONTRACT_V2: Final = "not_a_contract_v2"
REASON_NO_POSITIONS: Final = "no_positions"
REASON_G1: Final = "g1"

_FAIL: Final = "FAIL"
_PLACEHOLDER_CONTENT: Final = "cnt:" + "0" * 64
_PLACEHOLDER_ARTIFACT: Final = "art:" + "0" * 64


@dataclass(frozen=True, slots=True)
class EnvelopeIds:
    """What the runtime that drives the capture decides: the ids, the time, the sequence."""

    session_id: str
    observation_id: str
    produced_at: str
    next_seq: Callable[[], int]


@dataclass(frozen=True, slots=True)
class ParentRef:
    """The ``generated_text`` artifact of the same observation, already stored."""

    artifact_id: str
    content_id: str
    seq: int
    observation_id: str


@dataclass(frozen=True, slots=True)
class BuildResult:
    """``envelopes``: point -> sealed envelope. ``blobs``: uri -> bytes, for the Gate and the
    store (``token_index`` always; ``points`` where the rows were given). ``record``: the
    capture record."""

    envelopes: dict[str, dict[str, Any]]
    blobs: dict[str, bytes]
    record: dict[str, Any]


def _synthetic_refusal(document: Any) -> str | None:
    origin = document.get("origin") if isinstance(document, Mapping) else None
    value = origin.get("synthetic", None) if isinstance(origin, Mapping) else None
    if value is True:
        return REASON_SYNTHETIC
    return None if value is False else REASON_SYNTHETIC_NOT_STATED


def session_refusals(document: Any, check: ContractCheck) -> tuple[str, ...]:
    """Every reason no envelope is built for any position of ``document`` (empty: none)."""
    out: list[str] = []
    synthetic = _synthetic_refusal(document)
    if synthetic is not None:
        out.append(synthetic)
    if check.status == REJECTED:
        out.extend(check.reasons)
    elif check.status == NOT_VERIFIED:
        header = document.get("contract") if isinstance(document, Mapping) else None
        if header != {"name": CONTRACT_NAME, "version": CONTRACT_VERSION}:
            out.append(REASON_NOT_A_CONTRACT_V2)
    elif check.status != VERIFIED:
        raise ValueError(f"unknown contract status: {check.status!r}")
    out.extend(session_obstacles(document))
    if not rec.points_of_contract(
        document.get("positions") if isinstance(document, Mapping) else None
    ):
        out.append(REASON_NO_POSITIONS)
    return tuple(out)


def contract_ref_of(check: ContractCheck) -> dict[str, Any] | None:
    """The payload's ``contract_ref``: the verified hash, else the recorded one, else ``None``."""
    if check.status == REJECTED or check.contract_sha256 is None:
        return None
    return {
        "name": CONTRACT_NAME,
        "version": CONTRACT_VERSION,
        "contract_sha256": check.contract_sha256,
    }


def _g1_failures(check: ContractCheck, point: str) -> list[str]:
    """``g1:<check>`` for every check of the position's RAW that failed in the report."""
    if check.report is None:
        return []
    position = check.report.get("G1", {}).get("positions", {}).get(point, {})
    checks = position.get("checks", {})
    return [f"{REASON_G1}:{name}" for name, result in checks.items() if result == _FAIL]


def _extensions(
    document: Mapping[str, Any],
    point: str,
    session_extensions: Mapping[str, Any] | None,
    extensions: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        EXT_CONTRACT_SESSION: document.get("session_id"),
        EXT_ASSEMBLED_BY: ASSEMBLED_BY,
    }
    for given in (session_extensions or {}, (extensions or {}).get(point, {})):
        for key, value in given.items():
            if not isinstance(key, str) or not key.startswith("x-"):
                raise ValueError(f"an extension key starts with 'x-': {key!r}")
            if key in (EXT_CONTRACT_SESSION, EXT_ASSEMBLED_BY):
                raise ValueError(f"the extension {key!r} is written by nvs-runtime")
            out[key] = value
    return out


def build_envelope(
    document: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    ids: EnvelopeIds,
    space_id: str,
    seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
    parent: ParentRef | None = None,
    producer_build: str | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The sealed envelope of one payload. ``seal(envelope, lookup)`` writes the two ids.

    ``producer.version`` is the capture package's version and ``producer.build`` the adapter
    (unless ``producer_build`` is given): both enter the ids, so they come from the contract and
    not from this assembler, whose version goes to the extensions.
    """
    capture = document["origin"]["capture"]
    parents: list[dict[str, Any]] = []
    lookup: dict[str, Any] = {}
    if parent is not None:
        parents.append(
            {"role": PARENT_ROLE, "ordered": False, "artifact_ids": [parent.artifact_id]}
        )
        lookup[parent.artifact_id] = {
            "content_id": parent.content_id,
            "seq": parent.seq,
            "observation_id": parent.observation_id,
        }
    envelope: dict[str, Any] = {
        "contract": {"id": "IC-00", "version": IC00_VERSION},
        "content_id": _PLACEHOLDER_CONTENT,
        "artifact_id": _PLACEHOLDER_ARTIFACT,
        "kind": KIND,
        "payload_schema": {"id": PAYLOAD_SCHEMA_ID, "version": PAYLOAD_SCHEMA_VERSION},
        "payload": dict(payload),
        "producer": {
            "name": PRODUCER_NAME,
            "version": capture["version"],
            "build": capture["adapter"] if producer_build is None else producer_build,
        },
        "session_id": ids.session_id,
        "observation_id": ids.observation_id,
        "event_span": None,
        "space_id": space_id,
        "parents": parents,
        "policy_refs": [],
        "calibration_ref": None,
        "rule_refs": [],
        "observed_at": None,
        "seq": ids.next_seq(),
        "produced_at": ids.produced_at,
        "extensions": dict(extensions or {}),
    }
    return seal(envelope, lookup)


def build_envelopes(
    document: Mapping[str, Any],
    check: ContractCheck,
    *,
    ids: EnvelopeIds,
    space_id_of: Callable[[Mapping[str, Any]], str],
    seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
    rows: Mapping[str, bytes] | None = None,
    parent: ParentRef | None = None,
    producer_build: str | None = None,
    session_extensions: Mapping[str, Any] | None = None,
    extensions: Mapping[str, Mapping[str, Any]] | None = None,
    copied: bool = False,
    policy_name: str = POLICY_ALL_ROWS,
) -> BuildResult:
    """One envelope per position of ``document`` that can be built, and the capture record.

    ``check`` is ``check_contract_v2(document, ...)``. ``rows`` (point -> the rows' bytes) is
    optional: without it the envelope refers to the rows by the contract's ``trajectory_id``.
    ``space_id_of(materials)`` and ``seal(envelope, lookup)`` reach the sensos contracts; they are
    not called for a capture that is refused. ``extensions`` is keyed by point,
    ``session_extensions`` goes to every envelope.
    """
    points = rec.points_of_contract(
        document.get("positions") if isinstance(document, Mapping) else None
    )
    refusals = session_refusals(document, check)
    envelopes: dict[str, dict[str, Any]] = {}
    blobs: dict[str, bytes] = {}
    entries: dict[str, dict[str, Any]] = {}
    contract_ref = contract_ref_of(check)
    for point in points:
        if point in entries:
            continue  # a second position of the same name: the first one stands
        if refusals:
            entries[point] = rec.position_entry(
                rec.POSITION_NOT_BUILT, (rec.REASON_SESSION_REFUSED,)
            )
            continue
        failed = _g1_failures(check, point)
        given = None if rows is None or failed else rows.get(point)
        reasons = [*failed, *position_obstacles(document, point, rows_bytes=given)]
        if reasons:
            entries[point] = rec.position_entry(rec.POSITION_NOT_BUILT, reasons)
            continue
        payload = build_payload(
            document,
            point,
            contract_ref=contract_ref,
            rows_bytes=given,
            copied=copied,
            policy_name=policy_name,
        )
        position = next(p for p in document["positions"] if p.get("point") == point)
        envelope = build_envelope(
            document,
            payload,
            ids=ids,
            space_id=space_id_of(dict(position["space"])),
            seal=seal,
            parent=parent,
            producer_build=producer_build,
            extensions=_extensions(document, point, session_extensions, extensions),
        )
        envelopes[point] = envelope
        blobs[payload["token_index"]["uri"]] = token_index_bytes(payload["length"])
        if given is not None:
            blobs[payload["points"]["uri"]] = given
        entries[point] = rec.position_entry(
            rec.POSITION_ENVELOPED, artifact_id=envelope["artifact_id"]
        )
    contract_session = document.get("session_id") if isinstance(document, Mapping) else None
    record = rec.build_record(
        session_id=ids.session_id,
        observation_id=ids.observation_id,
        contract_session_id=contract_session if isinstance(contract_session, str) else None,
        capture=rec.capture_block_of_contract(document),
        contract_status=check.status,
        contract_sha256=check.contract_sha256,
        hash_source=check.hash_source,
        refusals=refusals,
        positions=entries,
    )
    return BuildResult(envelopes=envelopes, blobs=blobs, record=record)
