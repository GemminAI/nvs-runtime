"""``FormalWriter``: the one entry that writes the Formal Store and the capture archive (U6T-2b-b;
design U6T-2b v0.1, sections 2-4, W-1 to W-6).

* One writer per ``data_dir`` in this process (W-1): a module registry keyed by
  ``os.path.realpath(data_dir)``; ``writer_already_open`` for the second one.
* The writer owns the handles: the Formal Store (hekb-vnext ``IC04Store``, ``require_formal=True``),
  the capture archive and the save order. None is handed out (W-2).
* Every write is one line in one queue: a lock held for a whole save, so the order contract/RAW ->
  Gate -> put -> record is never interleaved with another capture's writes (W-3). Reads go through
  the writer too, under the same lock (W-4).
* One hekb-vnext checkout per process (U6T-2 section 15, condition 2): ``hekb_checkout_conflict``.
* An outside write to a chain is detected before an append (not a lock; U6T-8 closes the window):
  ``store_changed_externally``, and the writer never writes again. A half-written last line of an
  archive chain is set aside, evidence first, at open (see ``capture_chain``).
* An envelope that is in the Formal Store but in no capture record (the save stopped before the
  record) is listed at open (``unrecorded_envelopes``). No record is written for it (N-2).

This is detection and convention, not an OS lock (W-5). hekb-vnext's other entries do not write a
Formal Store (W-6).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from types import TracebackType
from typing import Any

from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.capture_chain import ChainChangedExternally
from nvs_runtime.hidden_trajectory.formal_save import SaveResult, _CaptureSaver
from nvs_runtime.hidden_trajectory.k_measure import KMeasurer
from nvs_runtime.hidden_trajectory.run import _now
from nvs_runtime.integrations import hekb_formal

CHANGED_EXTERNALLY = "store_changed_externally"


class WriterError(RuntimeError):
    """``code``: writer_already_open, hekb_checkout_conflict, writer_closed, saver_not_configured,
    store_changed_externally, wrong_payload, parent_not_stored, gate_not_configured."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


_registry_lock = threading.Lock()
_registry: dict[str, FormalWriter] = {}

_TOKEN_TEMPORAL = ("nvs.token-temporal", "0.1.0")
_TRAJECTORY_WINDOW = ("msr.trajectory-window", "0.10.0")
_MEANING_FRAME = ("nvs.meaning-frame", "0.1.0")
_MEANING_MEASUREMENT = ("nvs.meaning-measurement", "0.1.0")


def _schema_pair(envelope: Mapping[str, Any]) -> tuple[str, str]:
    schema = envelope.get("payload_schema") if isinstance(envelope, Mapping) else None
    if not isinstance(schema, Mapping):
        return ("", "")
    ident = schema.get("id")
    version = schema.get("version")
    return (
        ident if isinstance(ident, str) else "",
        version if isinstance(version, str) else "",
    )


def _parent_ids(envelope: Mapping[str, Any], role: str | None = None) -> list[str]:
    parents = envelope.get("parents")
    if not isinstance(parents, list):
        return []
    found: list[str] = []
    for entry in parents:
        if not isinstance(entry, Mapping):
            continue
        if role is not None and entry.get("role") != role:
            continue
        ids = entry.get("artifact_ids")
        if isinstance(ids, list):
            found.extend(item for item in ids if isinstance(item, str) and item)
    return found


def _lookup_entry(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "content_id": envelope["content_id"],
        "seq": envelope["seq"],
        "observation_id": envelope["observation_id"],
    }


def _open_store(hekb_vnext_root: str, data_dir: str) -> Any:  # a seam for the tests without hekb
    return hekb_formal._open_formal_store(hekb_vnext_root, data_dir)


def _create_store(hekb_vnext_root: str, data_dir: str, declaration: Mapping[str, Any]) -> Any:
    return hekb_formal._create_formal_store(hekb_vnext_root, data_dir, declaration)


class FormalWriter:
    """Make one with ``FormalWriter.create(...)`` (a new Formal Store) or ``FormalWriter.open(...)``
    (an existing one); use it as a context manager or call ``close()``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("use FormalWriter.create(...) or FormalWriter.open(...)")

    # ---- making one ----------------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        data_dir: str | os.PathLike[str],
        declaration: Mapping[str, Any],
        *,
        hekb_vnext_root: str | os.PathLike[str],
        gate: Callable[..., Mapping[str, Any]] | None = None,
        space_id_of: Callable[[Mapping[str, Any]], str] | None = None,
        seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]] | None = None,
        checker_bundle: bytes | Callable[[], bytes] | None = None,
        clock: Callable[[], datetime] = _now,
        k_path: bool = False,
        key_source: Callable[[], bytes] | None = None,
        k_measurer: KMeasurer | None = None,
    ) -> FormalWriter:
        """A new Formal Store (``create_formal``) in an empty folder, and the writer of it."""
        return cls._make(
            data_dir,
            hekb_vnext_root,
            declaration,
            gate,
            space_id_of,
            seal,
            checker_bundle,
            clock,
            (k_path, key_source, k_measurer),
        )

    @classmethod
    def open(
        cls,
        data_dir: str | os.PathLike[str],
        *,
        hekb_vnext_root: str | os.PathLike[str],
        gate: Callable[..., Mapping[str, Any]] | None = None,
        space_id_of: Callable[[Mapping[str, Any]], str] | None = None,
        seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]] | None = None,
        checker_bundle: bytes | Callable[[], bytes] | None = None,
        clock: Callable[[], datetime] = _now,
        k_path: bool = False,
        key_source: Callable[[], bytes] | None = None,
        k_measurer: KMeasurer | None = None,
    ) -> FormalWriter:
        """An existing Formal Store (``require_formal=True``) and its capture archive. Then
        ``unrecorded_envelopes`` says which stored envelopes have no capture record."""
        return cls._make(
            data_dir,
            hekb_vnext_root,
            None,
            gate,
            space_id_of,
            seal,
            checker_bundle,
            clock,
            (k_path, key_source, k_measurer),
        )

    @classmethod
    def _make(
        cls,
        data_dir: str | os.PathLike[str],
        hekb_vnext_root: str | os.PathLike[str],
        declaration: Mapping[str, Any] | None,
        gate: Any,
        space_id_of: Any,
        seal: Any,
        checker_bundle: Any,
        clock: Callable[[], datetime],
        k: tuple[bool, Callable[[], bytes] | None, KMeasurer | None],
    ) -> FormalWriter:
        key = os.path.realpath(os.fspath(data_dir))
        with _registry_lock:
            if key in _registry:
                raise WriterError("writer_already_open", key)
            try:
                root = hekb_formal._claim_checkout(os.fspath(hekb_vnext_root))
            except hekb_formal.CheckoutConflict as error:
                raise WriterError("hekb_checkout_conflict", str(error)) from error
            try:
                self = object.__new__(cls)
                self._init(
                    key, root, declaration, gate, space_id_of, seal, checker_bundle, clock, k
                )
            except BaseException:
                hekb_formal._release_checkout()
                raise
            _registry[key] = self
            return self

    def _init(
        self,
        key: str,
        root: str,
        declaration: Mapping[str, Any] | None,
        gate: Any,
        space_id_of: Any,
        seal: Any,
        checker_bundle: bytes | Callable[[], bytes] | None,
        clock: Callable[[], datetime],
        k: tuple[bool, Callable[[], bytes] | None, KMeasurer | None],
    ) -> None:
        self._key = key
        self._lock = threading.RLock()
        self._closed = False
        self._dead: str | None = None
        # the Formal Store first: a folder that is not one is left alone (no archive is made)
        if declaration is None:
            self._store = _open_store(root, key)
        else:
            self._store = _create_store(root, key, declaration)
        self._archive = CaptureArchive(key, guarded=True)
        self._archive.set_aside_torn_tails()
        self._gate = gate
        self._checker_bundle: bytes | Callable[[], bytes] | None = checker_bundle
        self._saver: _CaptureSaver | None = None
        if gate is not None and space_id_of is not None and seal is not None:
            self._saver = _CaptureSaver(
                store=self._store,
                archive=self._archive,
                gate=gate,
                space_id_of=space_id_of,
                seal=seal,
                checker_bundle=checker_bundle,
                clock=clock,
                k_path=k[0],
                key_source=k[1],
                k_measurer=k[2],
            )
        self._unrecorded = tuple(self._find_unrecorded())

    # ---- closing -------------------------------------------------------------------------------
    def close(self) -> None:
        """Release the folder and the hekb-vnext checkout. The writer is unusable afterwards."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        with _registry_lock:
            if _registry.get(self._key) is self:
                del _registry[self._key]
        hekb_formal._release_checkout()

    def __enter__(self) -> FormalWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def data_dir(self) -> str:
        return self._key

    @property
    def dead(self) -> str | None:
        """Why this writer no longer writes (an outside write was seen), or ``None``."""
        return self._dead

    def _check_usable(self, *, writing: bool) -> None:
        if self._closed:
            raise WriterError("writer_closed", self._key)
        if writing and self._dead:
            raise WriterError(CHANGED_EXTERNALLY, self._dead)

    # ---- writing -------------------------------------------------------------------------------
    def save(self, captured: Any, **kwargs: Any) -> SaveResult:
        """Save one capture: contract and RAW -> Gate -> put (per envelope) -> the capture record,
        last; see ``formal_save``. The whole order runs under the writer's lock. Another writer
        seen on a chain (``store_changed_externally``) stops this writer for good: the record of
        the capture in hand is written as far as it can be, and if it cannot, this raises."""
        with self._lock:
            self._check_usable(writing=True)
            if self._saver is None:
                raise WriterError("saver_not_configured", "gate, space_id_of and seal are needed")
            try:
                self._archive.check_unchanged()  # before the first byte of this capture
                result = self._saver.save(captured, **kwargs)
            except ChainChangedExternally as error:
                self._dead = str(error)
                raise WriterError(CHANGED_EXTERNALLY, str(error)) from error
            positions = result.record.get("positions", {})
            seen = [
                p["store_refusal"]["code"]
                for p in positions.values()
                if isinstance(p, dict) and p.get("store_refusal")
            ]
            if CHANGED_EXTERNALLY in seen:
                self._dead = "the Formal Store chain was written by somebody else"
            elif self._archive.dead:
                self._dead = self._archive.dead
            return result

    # ---- token temporal, window, frame, measurement (the capture save above is unchanged) --------
    def save_token_or_window(
        self, envelope: Mapping[str, Any], blobs: Mapping[str, bytes]
    ) -> dict[str, Any]:
        """Put a sealed ``nvs.token-temporal`` or ``msr.trajectory-window`` envelope.

        The parent observation must already be in the store. Another payload is refused
        before the Gate runs.
        """
        with self._lock:
            self._check_usable(writing=True)
            self._refuse_other(envelope, (_TOKEN_TEMPORAL, _TRAJECTORY_WINDOW))
            parents = _parent_ids(envelope)
            if not parents:
                raise WriterError("parent_not_stored", "the envelope has no parent")
            stored, lookup = self._parents_from_store(parents)
            return self._gate_and_put(
                envelope, blobs, lookup=lookup, stored=stored, gate_blobs=dict(blobs)
            )

    def save_meaning_frame(
        self, envelope: Mapping[str, Any], blobs: Mapping[str, bytes]
    ) -> dict[str, Any]:
        """Put a sealed ``nvs.meaning-frame`` envelope. A frame has no parents.

        Whether an observation of ``source_space`` is in the store is not checked.
        """
        with self._lock:
            self._check_usable(writing=True)
            self._refuse_other(envelope, (_MEANING_FRAME,))
            return self._gate_and_put(
                envelope, blobs, lookup={}, stored={}, gate_blobs=dict(blobs)
            )

    def save_meaning_measurement(
        self,
        envelope: Mapping[str, Any],
        blobs: Mapping[str, bytes],
        *,
        include_frame_blobs: bool = True,
    ) -> dict[str, Any]:
        """Put a sealed ``nvs.meaning-measurement`` envelope.

        The parent window, and the token-temporal parent when ``token_axis`` is present,
        must already be in the store. The Gate is given those envelopes, the observation
        the window names when it is stored, and any stored frame with the same
        ``frame_id``. ``include_frame_blobs`` false gives the Gate the frame envelope
        and not its bytes.
        """
        with self._lock:
            self._check_usable(writing=True)
            self._refuse_other(envelope, (_MEANING_MEASUREMENT,))
            windows = _parent_ids(envelope, "source_window")
            if len(windows) != 1:
                raise WriterError("wrong_payload", "source_window")
            parent_ids = [windows[0], *_parent_ids(envelope, "token_axis")]
            stored, lookup = self._parents_from_store(parent_ids)
            window = stored[windows[0]]["envelope"]
            for artifact_id in _parent_ids(window):
                held = self._store.get(artifact_id)
                if held is not None:
                    stored[artifact_id] = {"envelope": held["envelope"]}
            gate_blobs = dict(blobs)
            payload = envelope.get("payload")
            frame_id = payload.get("frame_id") if isinstance(payload, Mapping) else None
            if isinstance(frame_id, str) and frame_id:
                self._add_frames(stored, gate_blobs, frame_id, include_frame_blobs)
            return self._gate_and_put(
                envelope, blobs, lookup=lookup, stored=stored, gate_blobs=gate_blobs
            )

    def _refuse_other(
        self, envelope: Mapping[str, Any], allowed: tuple[tuple[str, str], ...]
    ) -> None:
        pair = _schema_pair(envelope)
        if pair not in allowed:
            raise WriterError("wrong_payload", f"{pair[0]} {pair[1]}".strip())

    def _parents_from_store(
        self, artifact_ids: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        stored: dict[str, Any] = {}
        lookup: dict[str, Any] = {}
        for artifact_id in artifact_ids:
            held = self._store.get(artifact_id)
            if held is None:
                raise WriterError("parent_not_stored", artifact_id)
            stored[artifact_id] = {"envelope": held["envelope"]}
            lookup[artifact_id] = _lookup_entry(held["envelope"])
        return stored, lookup

    def _add_frames(
        self,
        stored: dict[str, Any],
        gate_blobs: dict[str, bytes],
        frame_id: str,
        include_frame_blobs: bool,
    ) -> None:
        for artifact_id in self._stored_ids():
            held = self._store.get(artifact_id)
            if held is None:
                continue
            env = held["envelope"]
            if _schema_pair(env) != _MEANING_FRAME:
                continue
            payload = env.get("payload") if isinstance(env, Mapping) else None
            if not isinstance(payload, Mapping) or payload.get("frame_id") != frame_id:
                continue
            stored[artifact_id] = {"envelope": env}
            if include_frame_blobs:
                blobs = held.get("blobs")
                if isinstance(blobs, Mapping):
                    gate_blobs.update(blobs)

    def _bundle_bytes(self) -> bytes | None:
        bundle = self._checker_bundle
        if callable(bundle):
            bundle = bundle()
            self._checker_bundle = bundle
        return bundle

    def _gate_and_put(
        self,
        envelope: Mapping[str, Any],
        blobs: Mapping[str, bytes],
        *,
        lookup: Mapping[str, Any],
        stored: Mapping[str, Any],
        gate_blobs: Mapping[str, bytes],
    ) -> dict[str, Any]:
        """Make the verdict, then ``IC04Store.put``. The caller holds the writer lock.

        A refusal of the store (``IC04Error``) propagates with its code.
        """
        if self._gate is None:
            raise WriterError("gate_not_configured", "a gate is needed")
        verdict = self._gate(
            envelope, lookup=dict(lookup), stored=dict(stored), blobs=dict(gate_blobs)
        )
        receipt: dict[str, Any] = self._store.put(
            dict(envelope),
            dict(blobs),
            gate_verdict=verdict,
            checker_bundle=self._bundle_bytes(),
        )
        return receipt

    # ---- reading (through the writer, W-4) -------------------------------------------------------
    def get(self, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._check_usable(writing=False)
            held: dict[str, Any] | None = self._store.get(artifact_id)
            return held

    def gate_status(self, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._check_usable(writing=False)
            status: dict[str, Any] | None = self._store.gate_status(artifact_id)
            return status

    def records(self) -> list[dict[str, Any]]:
        """The capture records, in order (the chain is verified as it is read)."""
        with self._lock:
            self._check_usable(writing=False)
            return list(self._archive.records())

    def stored_artifact_ids(self) -> list[str]:
        with self._lock:
            self._check_usable(writing=False)
            return self._stored_ids()

    def torn_tails(self) -> list[dict[str, Any]]:
        """Every last line set aside: of the archive chains, then of the Formal Store chain."""
        with self._lock:
            self._check_usable(writing=False)
            tails = self._archive.torn_tails()
            of_store = getattr(self._store, "torn_tails", None)
            return [*tails, *(of_store() if callable(of_store) else [])]

    def observation(self, observation_id: str) -> dict[str, Any]:
        """Everything kept for one observation, read back (U6-6 design 4; a reading, not a recall):
        the capture record, the contract, the id lists, and for each position the envelope and
        GateVerdict of route R, the RAW file, and the K files of route K. Whatever is missing is
        named in ``missing`` (nothing is left out silently). ``record`` is the latest capture record
        of the observation."""
        with self._lock:
            self._check_usable(writing=False)
            found = [
                r for r in self._archive.records() if r.get("observation_id") == observation_id
            ]
            if not found:
                return {"observation_id": observation_id, "record": None, "missing": ["record"]}
            record = found[-1]
            missing: list[str] = []
            contract_sha = record["contract"]["contract_sha256"]
            contract = self._archive.read_contract(contract_sha) if contract_sha else None
            if contract is None:
                missing.append("contract")
            id_lists: dict[str, bytes | None] = {}
            if contract is not None:
                for name, key in (
                    ("input", "prompt_ids_sha256"),
                    ("generated", "gen_token_ids_sha256"),
                ):
                    digest = contract["binding"].get(key)
                    id_lists[name] = self._archive.read_ids(digest) if digest else None
                    if id_lists[name] is None:
                        missing.append(f"ids.{name}")
            trajectory = (
                {p["point"]: p["trajectory_id"] for p in contract["positions"]} if contract else {}
            )
            positions: dict[str, Any] = {}
            for point, entry in record["positions"].items():
                item: dict[str, Any] = {
                    "trajectory_id": trajectory.get(point),
                    "envelope": None,
                    "gate_verdict": None,
                    "raw": None,
                    "k": None,
                    "k_path": entry.get("k_path"),
                }
                held = None
                if entry.get("artifact_id"):
                    held = self._store.get(entry["artifact_id"])
                if entry.get("stored") is not None:
                    if held is None:
                        missing.append(f"{point}.envelope")
                    else:
                        item["envelope"] = held["envelope"]
                        item["gate_verdict"] = held.get("gate_verdict")
                tid = item["trajectory_id"]
                if tid is not None:
                    item["raw"] = self._archive.raw_for_trajectory(tid)
                    if item["raw"] is None:
                        missing.append(f"{point}.raw")
                k_entry = entry.get("k_path")
                if k_entry is not None and k_entry["status"] == "measured":
                    board = self._archive.read_k(k_entry["board_records_sha256"])
                    causal = self._archive.read_k(k_entry["causal_log_sha256"])
                    item["k"] = {"board_records": board, "causal_log": causal}
                    if board is None:
                        missing.append(f"{point}.k.board_records")
                    if causal is None:
                        missing.append(f"{point}.k.causal_log")
                positions[point] = item
            return {
                "observation_id": observation_id,
                "record": record,
                "contract": contract,
                "ids": id_lists,
                "positions": positions,
                "missing": missing,
            }

    @property
    def unrecorded_envelopes(self) -> tuple[str, ...]:
        """The ``artifact_id`` of every envelope that was in the Formal Store at open and in no
        capture record's ``stored`` (the save stopped before its record). Nothing is written for
        them."""
        return self._unrecorded

    def find_unrecorded(self) -> list[str]:
        """The same list, from the files as they are now."""
        with self._lock:
            self._check_usable(writing=False)
            return self._find_unrecorded()

    def _stored_ids(self) -> list[str]:
        ids, _more = self._store.select({}, 10**9)
        return list(ids)

    def _find_unrecorded(self) -> list[str]:
        recorded: set[str] = set()
        for record in self._archive.records():
            for position in record.get("positions", {}).values():
                stored = position.get("stored") if isinstance(position, dict) else None
                if isinstance(stored, dict) and isinstance(stored.get("artifact_id"), str):
                    recorded.add(stored["artifact_id"])
        return [a for a in self._stored_ids() if a not in recorded]
