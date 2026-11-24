"""The save of one capture into a Formal Store and the capture archive (U6T-2 design 7, U6T-2b').

``CaptureSaver.save`` is the one entry that writes. In order:

1. the contract and the capture are checked (nothing written yet);
2. the contract document and the RAW files go to the capture archive (content hashes; never
   rewritten);
3. the envelopes are built (the caller's ``space_id_of`` and ``seal``);
4. each envelope goes to the Gate (``GateClient.verdict``) and then to the Formal Store (``put``
   with the GateVerdict);
5. the capture record is appended to the archive **last**: a record is there only when what it names
   was written before it.

**R-9: a record is always written.** A failure of ``space_id_of``, ``seal``, the archive write, the
Gate or the store does not lose
the capture: the record says which stage failed and why (see below), and what had been written stays
(append only). Only a failure
of the record's own append is raised (nothing may hide it). Not recorded, raised as before
(``run_capture_session``): a mistake of
the caller's data -- ``LookupMissing``, ``ParentEnvelopeMissing``, ``ParentMismatch`` -- which is
found before anything is written.

How a failure shows in the record (``outcome`` keeps its five values, see U6T-2 design 14):

* before an envelope exists (contract stage, contract check, archive write, ``space_id_of``,
  ``seal``, building): ``outcome``
  ``refused``, every position ``not_built`` with the reason ``<stage>_failed``, and a ``refusals``
  line ``<stage>_failed: <type>: <message>``;
* the Gate failed for an envelope: the position is ``enveloped``, its ``gate.status`` is
  ``gate_error`` and ``store_refusal`` is
  ``gate_error`` (the store was not asked);
* the store refused an envelope (a ``code`` of its own, e.g. ``gate_rejected``) or raised: the
  position is ``enveloped``,
  ``stored`` is ``null``, ``store_refusal`` is ``{code, detail}`` (``put_failed`` for anything
  without a code of its own);
* the store took it: ``stored`` is ``{artifact_id, hekb_seq, hekb_hash}``. ``verdict_sha256`` is
  that of the verdict whenever there is one.

A parent: given a ``ParentRef`` and no ``parent_envelope``, the parent's envelope is read from the
Formal Store (``store.get``); a
parent the store does not have is ``ParentAbsent`` (a refusal record, nothing sent to the Gate).

``store`` is any object with ``put(envelope, blobs, *, gate_verdict, checker_bundle)`` and
``get(artifact_id)`` (a hekb-vnext
``IC04Store`` opened with ``require_formal=True``; see ``integrations/hekb_formal.py``). This module
imports no hekb-vnext and no sensos.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from nvs_observation.adapter_session import CapturedWithOrigin
from nvs_observation.contract_build import ContractBuild, contract_from_captured
from nvs_observation.contract_check import ContractCheck, check_contract_v2, rows_from_raw

from nvs_runtime.hidden_trajectory import record as rec
from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
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
from nvs_runtime.hidden_trajectory.k_measure import (
    KMeasurer,
    default_k_measurer,
    default_key_source,
)
from nvs_runtime.hidden_trajectory.payload import POLICY_ALL_ROWS
from nvs_runtime.hidden_trajectory.record_schema import MAX_REFUSAL_DETAIL_CHARS
from nvs_runtime.hidden_trajectory.run import (
    GATE_ERROR,
    STAGE_BUILT,
    STAGE_EMPTY_GENERATION,
    STAGE_REFUSED,
    _gate_fields,
    _gate_inputs,
    _now,
    _parent_differences,
    _parent_not_stored_record,
    _without_contract,
    with_capture_extra,
)

#: ``SaveResult.stage`` besides the three of ``run.py``: where a save stopped.
STAGE_FAILED_PREFIX: Final = "failed:"
_FETCH: Final = object()


@dataclass(frozen=True)
class SaveResult:
    """``record`` is what was appended (and ``line`` its chain line). ``receipts``: point -> the
    store's receipt of a stored envelope;
    ``store_refusals``: point -> ``{code, detail}``; ``archived`` says what the archive wrote
    (``written`` false = it was there already)."""

    record: dict[str, Any]
    line: dict[str, Any]
    stage: str
    envelopes: dict[str, dict[str, Any]] = field(default_factory=dict)
    verdicts: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    receipts: dict[str, dict[str, Any]] = field(default_factory=dict)
    store_refusals: dict[str, dict[str, str]] = field(default_factory=dict)
    gate_errors: dict[str, str] = field(default_factory=dict)
    contract: dict[str, Any] | None = None
    archived: dict[str, Any] = field(default_factory=dict)
    k_paths: dict[str, dict[str, Any]] = field(default_factory=dict)


def _detail(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"
    return (
        text
        if len(text) <= MAX_REFUSAL_DETAIL_CHARS
        else text[: MAX_REFUSAL_DETAIL_CHARS - 3] + "..."
    )


class _CaptureSaver:
    """The save order of one capture (internal: ``FormalWriter`` is the only entry that owns a
    Formal Store handle and calls this, under its lock; U6T-2b-b).

    ``store``: the Formal Store handle; ``archive``: a ``CaptureArchive`` of the same ``data_dir``;
    ``gate``: ``GateClient.verdict``;
    ``checker_bundle``: the bytes of the checker bundle, or a function that makes them (called once,
    when a ``put`` first needs it);
    ``space_id_of`` and ``seal``: the caller's (sensos's, through the door)."""

    def __init__(
        self,
        *,
        store: Any,
        archive: CaptureArchive,
        gate: Callable[..., Mapping[str, Any]],
        space_id_of: Callable[[Mapping[str, Any]], str],
        seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
        checker_bundle: bytes | Callable[[], bytes] | None = None,
        clock: Callable[[], datetime] = _now,
        k_path: bool = False,
        key_source: Callable[[], bytes] | None = None,
        k_measurer: KMeasurer | None = None,
    ) -> None:
        """``k_path`` (U6-6a): also keep the id lists and run route K for each position that has a
        RAW file; off, the save is what it was. The measurement is ``k_measurer`` when one was
        passed, and otherwise the measurer found through the ``nvs_runtime.k_measurer`` entry point
        (with none installed K is recorded as not measured). ``key_source`` is for tests. When it is
        omitted, the default key is chosen when K is measured, not when the saver is made."""
        self._k_path = k_path
        self._k_measurer = k_measurer
        self._key_source = key_source
        self._store, self._archive, self._gate = store, archive, gate
        self._space_id_of, self._seal = space_id_of, seal
        self._bundle = checker_bundle
        self._clock = clock

    # ---- helpers --------------------------------------------------------------------------
    def _bundle_bytes(self) -> bytes | None:
        if callable(self._bundle):
            self._bundle = self._bundle()
        return self._bundle

    def _finish(self, record: dict[str, Any], stage: str, **parts: Any) -> SaveResult:
        """Append the record (the last write). A failure here is not hidden."""
        line = self._archive.append_record(record)
        return SaveResult(record=record, line=line, stage=stage, **parts)

    def _failed(
        self,
        code: str,
        error: BaseException,
        ids: EnvelopeIds,
        build: ContractBuild | None,
        check: ContractCheck | None,
        **parts: Any,
    ) -> SaveResult:
        """The record of a save that stopped before an envelope existed."""
        line = f"{code}: {_detail(error)}"
        if build is None:
            capture = rec.capture_block_of_contract(None)
            document: Any = None
            extra = None
        else:
            capture = build.capture_block or rec.capture_block_of_contract(build.contract)
            document = build.contract
            extra = build.capture_extra
        contract_session = document.get("session_id") if isinstance(document, Mapping) else None
        record = rec.build_record(
            session_id=ids.session_id,
            observation_id=ids.observation_id,
            contract_session_id=contract_session if isinstance(contract_session, str) else None,
            capture=capture,
            contract_status=check.status if check is not None else rec.CONTRACT_NONE,
            contract_sha256=check.contract_sha256 if check is not None else None,
            hash_source=check.hash_source if check is not None else rec.CONTRACT_NONE,
            refusals=[line],
            positions={
                point: rec.position_entry(rec.POSITION_NOT_BUILT, (code,))
                for point in dict.fromkeys(capture.get("points") or [])
            },
        )
        if extra is not None:
            record = with_capture_extra(record, extra)
        return self._finish(
            record,
            f"{STAGE_FAILED_PREFIX}{code}",
            contract=document if isinstance(document, dict) else None,
            **parts,
        )

    def _archive_ids(self, build: ContractBuild, captured: CapturedWithOrigin) -> dict[str, Any]:
        """The id lists, as little-endian int64 (how the contract hashes them), under the contract's
        own hashes. Bytes that do not hash to the contract's value are refused
        (``ArchiveError``)."""
        assert build.contract is not None
        binding = build.contract["binding"]
        done: dict[str, Any] = {}
        for name, hash_key, attribute in (
            ("input", "prompt_ids_sha256", "prompt_ids"),
            ("generated", "gen_token_ids_sha256", "token_ids"),
        ):
            values = [int(v) for v in getattr(captured.capture, attribute)]
            data = struct.pack(f"<{len(values)}q", *values)
            written = self._archive.write_ids(data, binding[hash_key])
            done[name] = {"sha256": binding[hash_key], "written": written}
        return done

    def _archive_contract(self, build: ContractBuild, check: ContractCheck) -> dict[str, Any]:
        assert build.contract is not None
        if (
            check.contract_sha256 is None
        ):  # the contract was not verified: nothing is kept as if it were
            return {"contract_sha256": None, "contract_written": False, "raw": {}}
        document = build.contract
        written = self._archive.write_contract(document, check.contract_sha256)
        trajectory_ids = {p["point"]: p["trajectory_id"] for p in document["positions"]}
        raw = self._archive.write_raw(build.raw_bytes, trajectory_ids)
        return {"contract_sha256": check.contract_sha256, "contract_written": written, "raw": raw}

    # ---- the save -------------------------------------------------------------------------
    def save(
        self,
        captured: CapturedWithOrigin,
        *,
        contract_session_id: str,
        synthetic: bool,
        ids: EnvelopeIds,
        parent: ParentRef | None = None,
        parent_envelope: Mapping[str, Any] | ParentAbsent | None | object = _FETCH,
        producer_build: str | None = None,
        session_extensions: Mapping[str, Any] | None = None,
        extensions: Mapping[str, Mapping[str, Any]] | None = None,
        copied: bool = False,
        policy_name: str = POLICY_ALL_ROWS,
    ) -> SaveResult:
        """Save one capture; see the module docstring. ``parent_envelope`` left out means "read it
        from the Formal Store"; ``None`` means
        "not fetched" (``ParentEnvelopeMissing``), a ``ParentAbsent`` says the caller asked the
        store and the parent is not there."""
        try:
            build = contract_from_captured(
                captured, session_id=contract_session_id, synthetic=synthetic, clock=self._clock
            )
        except Exception as error:  # noqa: BLE001 -- R-9: the capture is not lost
            return self._failed(rec.REASON_CONTRACT_STAGE_FAILED, error, ids, None, None)
        if build.contract is None or build.capture_extra is None:
            record, stage = _without_contract(build, ids)
            return self._finish(record, stage)
        try:
            check = check_contract_v2(
                build.contract,
                raw_bytes=build.raw_bytes,
                instrument_records=build.instrument_records,
            )
        except Exception as error:  # noqa: BLE001
            return self._failed(rec.REASON_CONTRACT_CHECK_FAILED, error, ids, build, None)

        # the caller's mistakes are found before anything is written
        if parent is not None and parent_envelope is _FETCH:
            held = self._store.get(parent.artifact_id)
            parent_envelope = (
                held["envelope"]
                if held is not None
                else ParentAbsent(parent.artifact_id, "formal-store")
            )
        if parent is not None and isinstance(parent_envelope, ParentAbsent):
            if parent_envelope.artifact_id != parent.artifact_id:
                raise ParentMismatch(
                    {"artifact_id": (parent.artifact_id, parent_envelope.artifact_id)}
                )
        elif parent is not None:
            if parent_envelope is None:
                raise ParentEnvelopeMissing(parent.artifact_id)
            assert isinstance(parent_envelope, Mapping)
            differences = _parent_differences(parent, parent_envelope)
            if differences:
                raise ParentMismatch(differences)

        # 1. the contract and the RAW files
        try:
            archived = self._archive_contract(build, check)
            if self._k_path and check.contract_sha256 is not None:
                archived["ids"] = self._archive_ids(build, captured)
        except Exception as error:  # noqa: BLE001
            return self._failed(rec.REASON_ARCHIVE_WRITE_FAILED, error, ids, build, check)
        if parent is not None and isinstance(parent_envelope, ParentAbsent):
            record = _parent_not_stored_record(parent_envelope, build, check, ids)
            return self._finish(record, STAGE_REFUSED, contract=build.contract, archived=archived)

        # 2. the envelopes (the caller's space_id_of and seal; which one failed is told)
        failing: list[str] = []

        def space_id_of(materials: Mapping[str, Any]) -> str:
            try:
                return self._space_id_of(materials)
            except BaseException:
                failing.append(rec.REASON_SPACE_ID_FAILED)
                raise

        def seal(envelope: dict[str, Any], lookup: Mapping[str, Any]) -> dict[str, Any]:
            try:
                return self._seal(envelope, lookup)
            except BaseException:
                failing.append(rec.REASON_SEAL_FAILED)
                raise

        try:
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
        except Exception as error:  # noqa: BLE001
            code = failing[0] if failing else rec.REASON_BUILD_FAILED
            return self._failed(code, error, ids, build, check, archived=archived)
        record = with_capture_extra(built.record, build.capture_extra)

        # 3. the Gate and the store, envelope by envelope
        stored_parent = parent_envelope if isinstance(parent_envelope, Mapping) else None
        lookup, stored = _gate_inputs(parent, stored_parent)
        verdicts: dict[str, Mapping[str, Any]] = {}
        gate_errors: dict[str, str] = {}
        receipts: dict[str, dict[str, Any]] = {}
        refusals: dict[str, dict[str, str]] = {}
        for envelope in built.envelopes.values():
            if missing_lookup(envelope, lookup):
                raise LookupMissing(missing_lookup(envelope, lookup))
        for point, envelope in built.envelopes.items():
            verdict: Mapping[str, Any] | None = None
            try:
                verdict = self._gate(
                    envelope,
                    lookup=lookup,
                    stored=stored,
                    blobs=built.blobs,
                    contract_facts=check.facts,
                )
                status, violations, unverifiable = _gate_fields(verdict)
            except Exception as error:  # noqa: BLE001
                verdict = None
                gate_errors[point] = f"{type(error).__name__}: {error}"
                status, violations = GATE_ERROR, []
                unverifiable = [{"check": "gate_call", "reason": type(error).__name__}]
                refusals[point] = {"code": rec.STORE_GATE_ERROR, "detail": _detail(error)}
            else:
                verdicts[point] = verdict
            record = rec.with_gate_result(
                record, point, status=status, violations=violations, unverifiable=unverifiable
            )
            stored_entry: dict[str, Any] | None = None
            if verdict is not None:
                try:
                    receipt = self._store.put(
                        envelope,
                        _blobs_of(envelope, built.blobs),
                        gate_verdict=verdict,
                        checker_bundle=self._bundle_bytes(),
                    )
                except Exception as error:  # noqa: BLE001 -- a refusal (it has a code) or a failure of the store
                    own = getattr(error, "code", None)
                    refusals[point] = {
                        "code": own if isinstance(own, str) and own else rec.STORE_PUT_FAILED,
                        "detail": _detail(error),
                    }
                else:
                    receipts[point] = receipt
                    stored_entry = {
                        "artifact_id": receipt["artifact_id"],
                        "hekb_seq": receipt["hekb_seq"],
                        "hekb_hash": receipt["hekb_hash"],
                    }
            record = rec.with_store_result(
                record,
                point,
                stored=stored_entry,
                verdict_sha256=None if verdict is None else verdict.get("verdict_sha256"),
                store_refusal=refusals.get(point),
            )

        # 3b. route K, position by position (never stops R; a key lives only in ``run_position``)
        k_paths: dict[str, dict[str, Any]] = {}
        if self._k_path and check.contract_sha256 is not None:
            record, k_paths, archived["k"] = self._run_k(record, build, ids)

        # 4. the record, last
        return self._finish(
            record,
            STAGE_BUILT,
            envelopes=built.envelopes,
            verdicts=verdicts,
            receipts=receipts,
            store_refusals=refusals,
            gate_errors=gate_errors,
            contract=build.contract,
            archived=archived,
            k_paths=k_paths,
        )

    def _run_k(
        self, record: dict[str, Any], build: ContractBuild, ids: EnvelopeIds
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
        """K for each position with a RAW file: ``(record with k_path, k_paths, what was kept)``."""
        assert build.contract is not None
        origin = build.contract.get("origin", {})
        trajectory = {p["point"]: p["trajectory_id"] for p in build.contract["positions"]}
        k_paths: dict[str, dict[str, Any]] = {}
        kept: dict[str, Any] = {}
        measurer = self._k_measurer if self._k_measurer is not None else default_k_measurer()
        key_source = self._key_source if self._key_source is not None else default_key_source()
        for point, raw in build.raw_bytes.items():
            outcome = measurer.run_position(
                observation_id=ids.observation_id,
                point=point,
                raw_bytes=raw,
                trajectory_id=trajectory.get(point),
                model_build_id=origin.get("model_build_id"),
                compute_environment=origin.get("compute_environment"),
                key_source=key_source,
            )
            entry = outcome.k_path
            if outcome.board is not None and outcome.causal is not None:
                try:
                    board = self._archive.write_k(outcome.board)
                    causal = self._archive.write_k(outcome.causal)
                    kept[point] = {"board": board, "causal": causal}
                except Exception:  # noqa: BLE001 -- nothing of K is claimed that was not kept
                    entry = {
                        **entry,
                        "status": rec.K_NOT_MEASURED,
                        "reason": rec.REASON_K_ARCHIVE_WRITE_FAILED,
                        "board_records_sha256": None,
                        "causal_log_sha256": None,
                        "chain_head": None,
                        "integrity": None,
                    }
            k_paths[point] = entry
            record = rec.with_k_path(record, point, entry)
        return record, k_paths, kept


def _blobs_of(envelope: Mapping[str, Any], blobs: Mapping[str, bytes]) -> dict[str, bytes]:
    """The Blobs this envelope names (its points and token index), from all the Blobs of the
    capture."""
    payload = envelope.get("payload")
    uris = [
        (payload.get(key) or {}).get("uri")
        for key in ("points", "token_index")
        if isinstance(payload, Mapping)
    ]
    return {uri: blobs[uri] for uri in uris if isinstance(uri, str) and uri in blobs}


__all__ = [
    "STAGE_BUILT",
    "STAGE_EMPTY_GENERATION",
    "STAGE_FAILED_PREFIX",
    "STAGE_REFUSED",
    "SaveResult",
]
