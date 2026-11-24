"""The capture record (``nvs.capture-record/1``): what became of one captured generation.

One record per capture, whether or not an envelope was made: which positions were enveloped,
which were not and why, whether the whole capture was refused (a synthetic contract, a contract
that failed verification), and the capture that generated no token at all -- that one has no RAW,
no contract and no envelope, and this record is the only trace of it.

The record is kept on the nvs-runtime side. It is not an IC-00 artifact and is not stored in
HEKB. It carries hashes and counts only: no prompt text, no token ids, no key.

A record that a function of this module returns has passed ``record_schema.validate_record`` (the
strict schema of ``nvs.capture-record/1``); one that does not fit raises ``ValueError``. The
optional ``capture_extra`` (design U6-5 v0.5, 5.2) is a key of the record's top level, written only
when it is given.

``record_schema`` (standard library only) is reached by name when a record is checked, not by an
``import`` statement: the existing test T-U6-R08 fixes the import statements of this module.
``capture_record`` is the one import T-U6-R08 allows beyond the standard library: it holds the
shared capture vocabulary (and imports the standard library only) so this record and the
observation side name one definition, and the test checks that module does not reach ``nvs_edge``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Final

import nvs_observation.contracts.capture_record as _capture_record

CAPTURE_KEYS: Final = _capture_record.CAPTURE_KEYS
REASON_CAPTURE_EXTRA_OVERFLOW: Final = _capture_record.REASON_CAPTURE_EXTRA_OVERFLOW
REASON_CAPTURE_INVALID: Final = _capture_record.REASON_CAPTURE_INVALID
REASON_CONTRACT_NOT_BUILDABLE: Final = _capture_record.REASON_CONTRACT_NOT_BUILDABLE
REASON_DIMENSION_OUT_OF_RANGE: Final = _capture_record.REASON_DIMENSION_OUT_OF_RANGE
REASON_EDGE_INCOMPATIBLE: Final = _capture_record.REASON_EDGE_INCOMPATIBLE
REASON_EDGE_UNAVAILABLE: Final = _capture_record.REASON_EDGE_UNAVAILABLE
REASON_EMPTY_GENERATION: Final = _capture_record.REASON_EMPTY_GENERATION
REASON_ORIGIN_FACTS_INVALID: Final = _capture_record.REASON_ORIGIN_FACTS_INVALID
REASON_ORIGIN_MISMATCH: Final = _capture_record.REASON_ORIGIN_MISMATCH
REASON_SPACE_COLLISION: Final = _capture_record.REASON_SPACE_COLLISION
REASON_STEPS_OUT_OF_RANGE: Final = _capture_record.REASON_STEPS_OUT_OF_RANGE
capture_block = _capture_record.capture_block

RECORD_VERSION: Final = "nvs.capture-record/1"

OUTCOME_ENVELOPED: Final = "enveloped"
OUTCOME_PARTLY_ENVELOPED: Final = "partly_enveloped"
OUTCOME_EMPTY_GENERATION: Final = "empty_generation"
OUTCOME_REFUSED: Final = "refused"
OUTCOME_NOT_ENVELOPED: Final = "not_enveloped"

POSITION_ENVELOPED: Final = "enveloped"
POSITION_NOT_BUILT: Final = "not_built"

CONTRACT_NONE: Final = "none"
GATE_NOT_CALLED: Final = "not_called"

REASON_SESSION_REFUSED: Final = "session_refused"

#: The parent of the envelopes was looked up in the store and is not there (U6T-1).
REASON_PARENT_NOT_STORED: Final = "parent_not_stored"

#: U6T-2b': the save (``formal_save.py``) stopped at a stage that is not a refusal of the capture;
#: the record says which.
#: Each is a position reason and the code (before ``:``) of a ``refusals`` line.
REASON_CONTRACT_STAGE_FAILED: Final = "contract_stage_failed"
REASON_CONTRACT_CHECK_FAILED: Final = "contract_check_failed"
REASON_ARCHIVE_WRITE_FAILED: Final = "archive_write_failed"
REASON_SPACE_ID_FAILED: Final = "space_id_failed"
REASON_SEAL_FAILED: Final = "seal_failed"
REASON_BUILD_FAILED: Final = "build_failed"
#: ``store_refusal.code`` made by the save itself (the others are the Formal Store's own codes).
STORE_GATE_ERROR: Final = "gate_error"
STORE_PUT_FAILED: Final = "put_failed"


def _checked(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` if it fits the strict schema; else ``ValueError`` listing the violations."""
    module = __import__("nvs_runtime.hidden_trajectory.record_schema", fromlist=["validate_record"])
    violations: list[str] = module.validate_record(record)
    if violations:
        shown = "; ".join(violations[:5]) + ("; ..." if len(violations) > 5 else "")
        raise ValueError(f"the record does not fit {RECORD_VERSION}: {shown}")
    return record


def gate_not_called() -> dict[str, Any]:
    return {"status": GATE_NOT_CALLED, "violations": [], "unverifiable": []}


def position_entry(
    status: str, reasons: Sequence[str] = (), artifact_id: str | None = None
) -> dict[str, Any]:
    """One position of a record. ``artifact_id`` only for an enveloped position."""
    if status not in (POSITION_ENVELOPED, POSITION_NOT_BUILT):
        raise ValueError(f"unknown position status: {status!r}")
    if (status == POSITION_ENVELOPED) != (artifact_id is not None):
        raise ValueError("an enveloped position has an artifact_id, and only an enveloped one")
    return {
        "status": status,
        "reasons": list(reasons),
        "artifact_id": artifact_id,
        "gate": gate_not_called(),
    }


def _capture_types() -> Mapping[str, Any]:
    """The ``properties`` of ``capture`` in the record schema (read by name, not written twice)."""
    module = __import__("nvs_runtime.hidden_trajectory.record_schema", fromlist=["RECORD_SCHEMA"])
    properties: Mapping[str, Any] = module.RECORD_SCHEMA["$defs"]["capture"]["properties"]
    return properties


def _fits(value: Any, json_types: Any) -> bool:
    """Whether ``value`` has one of the JSON types ``json_types`` (``bool`` is not an integer)."""
    names = [json_types] if isinstance(json_types, str) else list(json_types)
    for name in names:
        if name == "null" and value is None:
            return True
        if name == "string" and isinstance(value, str):
            return True
        if name == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
    return False


def capture_block_of_contract(document: Any) -> dict[str, Any]:
    """The ``capture`` part read from a contract v2 document; a missing value is ``None``.

    A value whose type does not fit the ``capture`` definition of the record schema (``n_in`` and
    ``n_gen``: an integer other than ``bool``, or a string; the other fields: a string) is also
    ``None``. It is never converted (no ``str()``) and nothing is made up. This happens only to a
    document that did not pass the structural check, whose ``contract.status`` is then
    ``REJECTED``; the record of the capture is returned all the same.
    """

    def section(name: str) -> Mapping[str, Any]:
        found = document.get(name) if isinstance(document, Mapping) else None
        return found if isinstance(found, Mapping) else {}

    binding, origin = section("binding"), section("origin")
    capture = origin.get("capture")
    positions = document.get("positions") if isinstance(document, Mapping) else None
    raw: dict[str, Any] = {
        "n_in": binding.get("n_in"),
        "n_gen": binding.get("n_gen"),
        "finish_reason": binding.get("finish_reason"),
        "model_build_id": origin.get("model_build_id"),
        "compute_environment": origin.get("compute_environment"),
        "adapter": capture.get("adapter") if isinstance(capture, Mapping) else None,
        "prompt_ids_sha256": binding.get("prompt_ids_sha256"),
        "gen_token_ids_sha256": binding.get("gen_token_ids_sha256"),
    }
    types = _capture_types()
    typed = {
        key: (value if _fits(value, types[key]["type"]) else None) for key, value in raw.items()
    }
    return capture_block(points=points_of_contract(positions), **typed)


def points_of_contract(positions: Any) -> list[str]:
    """The point names of a contract's ``positions``, in order (those that have one)."""
    if not isinstance(positions, list):
        return []
    return [
        position["point"]
        for position in positions
        if isinstance(position, Mapping) and isinstance(position.get("point"), str)
    ]


def outcome_of(refusals: Sequence[str], positions: Mapping[str, Mapping[str, Any]]) -> str:
    """``refused`` if the capture was refused as a whole; else by how many positions were built."""
    if refusals:
        return OUTCOME_REFUSED
    built = sum(1 for entry in positions.values() if entry["status"] == POSITION_ENVELOPED)
    if built == 0:
        return OUTCOME_NOT_ENVELOPED
    return OUTCOME_ENVELOPED if built == len(positions) else OUTCOME_PARTLY_ENVELOPED


def build_record(
    *,
    session_id: str,
    observation_id: str,
    contract_session_id: str | None,
    capture: Mapping[str, Any],
    contract_status: str,
    contract_sha256: str | None,
    hash_source: str,
    refusals: Sequence[str],
    positions: Mapping[str, Mapping[str, Any]],
    capture_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A record of a capture that has a contract. ``positions``: point -> ``position_entry``.

    ``capture_extra`` (``capture_extra.build_capture_extra``'s document) is written to the top level
    of the record as a deep copy; ``None`` writes no such key, and the record is then the same, to
    the byte, as before the argument existed. ``ValueError`` if the record would not fit the schema.
    """
    if set(capture) != set(CAPTURE_KEYS):
        raise ValueError(f"capture must have exactly the keys {list(CAPTURE_KEYS)}")
    record: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "outcome": outcome_of(refusals, positions),
        "session_id": session_id,
        "observation_id": observation_id,
        "contract_session_id": contract_session_id,
        "capture": copy.deepcopy(dict(capture)),
        "contract": {
            "status": contract_status,
            "contract_sha256": contract_sha256,
            "hash_source": hash_source,
        },
        "refusals": list(refusals),
        "positions": {point: copy.deepcopy(dict(entry)) for point, entry in positions.items()},
    }
    if capture_extra is not None:
        if not isinstance(capture_extra, Mapping):
            raise ValueError("capture_extra must be a mapping (the document of capture_extra.py)")
        record["capture_extra"] = copy.deepcopy(dict(capture_extra))
    return _checked(record)


def empty_generation_record(
    *,
    session_id: str,
    observation_id: str,
    capture: Mapping[str, Any],
    capture_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The record of a capture that generated no token (``n_gen == 0``).

    There is no RAW and no contract for it (nvs_edge refuses both), so nothing is built and
    nothing is verified: every requested point is ``not_built`` with the reason
    ``empty_generation``. ``capture`` is a ``capture_block``. ``capture_extra`` is as in
    ``build_record`` (what is left of the capture's conditions and of the Adapter's facts).
    """
    n_gen = capture.get("n_gen")
    if isinstance(n_gen, bool) or n_gen != 0:
        raise ValueError(f"an empty generation has n_gen == 0, not {n_gen!r}")
    record = build_record(
        session_id=session_id,
        observation_id=observation_id,
        contract_session_id=None,
        capture=capture,
        contract_status=CONTRACT_NONE,
        contract_sha256=None,
        hash_source=CONTRACT_NONE,
        refusals=(),
        positions={
            point: position_entry(POSITION_NOT_BUILT, (REASON_EMPTY_GENERATION,))
            for point in capture["points"]
        },
        capture_extra=capture_extra,
    )
    record["outcome"] = OUTCOME_EMPTY_GENERATION
    return _checked(record)


def with_gate_result(
    record: Mapping[str, Any],
    point: str,
    *,
    status: str,
    violations: Sequence[Any] = (),
    unverifiable: Sequence[Any] = (),
) -> dict[str, Any]:
    """A copy of ``record`` with the Gate's result written to an enveloped position."""
    entry = record["positions"].get(point)
    if entry is None or entry["status"] != POSITION_ENVELOPED:
        raise ValueError(f"{point!r} is not an enveloped position of this record")
    out = copy.deepcopy(dict(record))
    out["positions"][point]["gate"] = {
        "status": status,
        "violations": copy.deepcopy(list(violations)),
        "unverifiable": copy.deepcopy(list(unverifiable)),
    }
    return _checked(out)


def with_store_result(
    record: Mapping[str, Any],
    point: str,
    *,
    stored: Mapping[str, Any] | None,
    verdict_sha256: str | None,
    store_refusal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A copy of ``record`` with what became of the envelope of ``point`` in the Formal Store
    (U6T-2b').

    ``stored`` is ``{artifact_id, hekb_seq, hekb_hash}`` (the store's receipt) or ``None``;
    ``store_refusal`` is ``{code, detail}`` for an envelope the store did not take; they are never
    both
    given. ``verdict_sha256`` is that of the GateVerdict the store was (or would have been) given.
    The whole GateVerdict is not in the record: the Formal Store holds it."""
    entry = record["positions"].get(point)
    if entry is None or entry["status"] != POSITION_ENVELOPED:
        raise ValueError(f"{point!r} is not an enveloped position of this record")
    out = copy.deepcopy(dict(record))
    out["positions"][point]["stored"] = copy.deepcopy(dict(stored)) if stored is not None else None
    out["positions"][point]["verdict_sha256"] = verdict_sha256
    out["positions"][point]["store_refusal"] = (
        copy.deepcopy(dict(store_refusal)) if store_refusal is not None else None
    )
    return _checked(out)


# ---- U6-6a: the K path of a position ------------------------------------------------------------
K_MEASURED: Final = "measured"
K_NOT_MEASURED: Final = "not_measured"
K_NOT_ELIGIBLE: Final = "not_eligible"
K_STATUSES: Final = (K_MEASURED, K_NOT_MEASURED, K_NOT_ELIGIBLE)
REASON_K_DIMENSION_NOT_REGISTERED: Final = "projection_dimension_not_registered"
REASON_K_DEFERRED: Final = "edge_deferred"
REASON_K_VERIFY_FAILED: Final = "board_record_verify_failed"
REASON_K_CAST_NOT_EXACT: Final = "edge_input_cast_not_exact"
REASON_K_EDGE_UNAVAILABLE: Final = "edge_unavailable"
REASON_K_FAILED: Final = "k_path_failed"
REASON_K_ARCHIVE_WRITE_FAILED: Final = "k_archive_write_failed"
K_VERIFY_CHECKS: Final = ("structure", "session_id", "crc", "record_hash", "hmac", "chain")


def with_k_path(record: Mapping[str, Any], point: str, k_path: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``record`` with the K path of ``point`` (U6-6a). Any position can carry one (an
    enveloped one, or one whose envelope was not made); ``ValueError`` if the record would not fit
    the schema."""
    if point not in record["positions"]:
        raise ValueError(f"{point!r} is not a position of this record")
    out = copy.deepcopy(dict(record))
    out["positions"][point]["k_path"] = copy.deepcopy(dict(k_path))
    return _checked(out)
