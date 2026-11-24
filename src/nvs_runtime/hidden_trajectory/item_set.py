"""RSA-X1: an entry for many short strings, one capture each, and the K position of each.

``load_item_set`` reads a JSON file of items (``item_id``, ``text``, ``role``) and fixes its
sha256. ``run_item_set`` makes one capture per item through a caller-supplied ``capture_fn`` (the
real adapter lives in tests and examples: nvs-runtime imports neither ``sensos_inference`` nor a
sensos contract, by design) and saves it with ``FormalWriter.save`` (route R through the Gate
into the
Formal Store; route K when the writer was made with ``k_path=True``). The text of an item goes to
the model and nowhere else: not into an envelope, not into the capture record (their key sets are
closed and are not changed here). What is kept about an item is its ``item_id``, the sha256 of its
text, and the ids of the capture it made: ``ItemRun``, one row of the mapping table.

``k_position`` reads the K position of a saved capture back from the capture archive alone (the
stored causal log, ``cycle_raw.geometry.position`` of cycle 1): nothing taken from the run's memory.

The set file is not one of the contract's inputs; its sha256 goes to the evidence table.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.envelope import EnvelopeIds

SET_VERSION: Final = "rsa-item-set/1"
SET_KEYS: Final = {"version", "set_id", "system", "items"}
ITEM_KEYS: Final = {"item_id", "text", "role"}
MAX_TEXT: Final = 8192
POINT: Final = "final_norm"
K_POSITION_DIMENSION: Final = 64

OUT_SAVED: Final = "saved"
OUT_CAPTURE_ERROR: Final = "capture_error"
OUT_SAVE_ERROR: Final = "save_error"

#: ``k_position`` statuses besides ``measured``
KPOS_NO_RECORD: Final = "no_capture_record"
KPOS_NO_POSITION: Final = "no_such_point"
KPOS_NOT_K: Final = "k_not_measured"
KPOS_FILE_MISSING: Final = "causal_log_missing"
KPOS_BAD_LOG: Final = "causal_log_unreadable"
KPOS_MEASURED: Final = "measured"


class ItemSetError(ValueError):
    """The set file is not a valid item set."""


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    text: str
    role: str

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ItemSet:
    set_id: str
    items: tuple[Item, ...]
    sha256: str  # of the bytes of the file, as read
    system: str | None = None


def load_item_set(path: str | Path, *, expected_sha256: str | None = None) -> ItemSet:
    """Read and check a set file. ``expected_sha256`` (the hash the caller recorded earlier): a file
    that hashes to something else raises, so a changed set is found before anything is captured."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ItemSetError(f"the set file hashes to {digest}, not the expected {expected_sha256}")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ItemSetError(f"the set file is not UTF-8 JSON: {type(error).__name__}") from error
    if not isinstance(doc, dict) or set(doc) - SET_KEYS or doc.get("version") != SET_VERSION:
        raise ItemSetError(
            f"expected an object with version {SET_VERSION!r} and keys {sorted(SET_KEYS)}"
        )
    set_id, raw_items, system = doc.get("set_id"), doc.get("items"), doc.get("system")
    if not isinstance(set_id, str) or not set_id:
        raise ItemSetError("set_id must be a non-empty string")
    if system is not None and (not isinstance(system, str) or not system):
        raise ItemSetError("system must be a non-empty string when present")
    if not isinstance(raw_items, list) or not raw_items:
        raise ItemSetError("items must be a non-empty list")
    items: list[Item] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict) or set(raw) != ITEM_KEYS:
            raise ItemSetError(f"items[{index}] must be an object with exactly {sorted(ITEM_KEYS)}")
        item_id, text, role = raw["item_id"], raw["text"], raw["role"]
        if not all(isinstance(v, str) and v for v in (item_id, text, role)):
            raise ItemSetError(f"items[{index}]: item_id, text and role must be non-empty strings")
        if len(text) > MAX_TEXT:
            raise ItemSetError(f"items[{index}]: text is over {MAX_TEXT} characters")
        if item_id in seen:
            raise ItemSetError(f"items[{index}]: item_id {item_id!r} is repeated")
        seen.add(item_id)
        items.append(Item(item_id, text, role))
    return ItemSet(set_id=set_id, items=tuple(items), sha256=digest, system=system)


def messages_for(item: Item, system: str | None = None) -> list[dict[str, str]]:
    """The chat messages of an item: the same shape for every item (an optional fixed system
    message of the set, then the text as one user message)."""
    out: list[dict[str, str]] = []
    if system is not None:
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": item.text})
    return out


@dataclass
class ItemRun:
    """One row of the mapping table: item -> capture. No text, only its sha256."""

    item_id: str
    role: str
    text_sha256: str
    outcome: str
    observation_id: str | None = None
    contract_session_id: str | None = None
    record_outcome: str | None = None
    position_artifact_id: str | None = None
    k_status: str | None = None
    k_reason: str | None = None
    k_reason_detail: str | None = None
    k_n_records: int | None = None
    error: str | None = None
    save_seconds: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


CaptureFn = Callable[[list[dict[str, str]]], Any]
IdsFn = Callable[[Item], EnvelopeIds]


def run_item_set(
    item_set: ItemSet,
    writer: Any,
    capture_fn: CaptureFn,
    ids_for: IdsFn,
    *,
    contract_session_id: Callable[[Item], str] | None = None,
    synthetic: bool = False,
    point: str = POINT,
) -> list[ItemRun]:
    """One capture and one save per item, in the order of the set. A failure of one item is a row
    (``capture_error`` / ``save_error`` with the class name, no message); the next
    item goes on."""
    rows: list[ItemRun] = []
    for item in item_set.items:
        row = ItemRun(item.item_id, item.role, item.text_sha256, OUT_SAVED)
        session = contract_session_id(item) if contract_session_id else f"rsa-x1-{item.item_id}"
        row.contract_session_id = session
        try:
            captured = capture_fn(messages_for(item, item_set.system))
        except Exception as error:  # noqa: BLE001 -- recorded as a row, not hidden
            row.outcome, row.error = OUT_CAPTURE_ERROR, type(error).__name__
            rows.append(row)
            continue
        began = time.perf_counter()
        try:
            ids = ids_for(item)
            row.observation_id = ids.observation_id
            result = writer.save(
                captured, contract_session_id=session, synthetic=synthetic, ids=ids
            )
        except Exception as error:  # noqa: BLE001
            row.outcome, row.error = OUT_SAVE_ERROR, type(error).__name__
            rows.append(row)
            continue
        finally:
            row.save_seconds = time.perf_counter() - began
        record = result.record
        row.record_outcome = record.get("outcome")
        entry = (record.get("positions") or {}).get(point)
        if isinstance(entry, Mapping):
            row.position_artifact_id = entry.get("artifact_id")
            k = entry.get("k_path")
            if isinstance(k, Mapping):
                row.k_status = k.get("status")
                row.k_reason = k.get("reason")
                row.k_reason_detail = k.get("reason_detail")
                row.k_n_records = k.get("n_records")
        rows.append(row)
    return rows


@dataclass(frozen=True, slots=True)
class KPosition:
    observation_id: str
    status: str
    reason: str | None = None
    position: tuple[float, ...] | None = None
    n_cycles: int | None = None
    edge_session_id: str | None = None
    causal_log_sha256: str | None = None

    @property
    def position_sha256(self) -> str | None:
        if self.position is None:
            return None
        import struct

        return hashlib.sha256(struct.pack(f"<{len(self.position)}d", *self.position)).hexdigest()


def k_position(data_dir: str | Path, observation_id: str, *, point: str = POINT) -> KPosition:
    """The K position (64-D, ``cycle_raw.geometry.position`` of cycle 1) of one saved capture, from
    the capture archive of ``data_dir`` only. Never raises for what the archive holds: a position
    that cannot be given says why (``status``/``reason``)."""
    latest: dict[str, Any] | None = None
    for record in CaptureArchive(data_dir).records():
        if record.get("observation_id") == observation_id:
            latest = record
    if latest is None:
        return KPosition(observation_id, KPOS_NO_RECORD)
    entry = (latest.get("positions") or {}).get(point)
    if not isinstance(entry, Mapping):
        return KPosition(observation_id, KPOS_NO_POSITION)
    k = entry.get("k_path")
    if not isinstance(k, Mapping) or k.get("status") != "measured":
        why = k.get("reason") if isinstance(k, Mapping) else None
        return KPosition(observation_id, KPOS_NOT_K, reason=str(why) if why else None)
    sha = k.get("causal_log_sha256")
    common: dict[str, Any] = {"edge_session_id": k.get("edge_session_id"), "causal_log_sha256": sha}
    data = CaptureArchive(data_dir).read_k(str(sha)) if sha else None
    if data is None:
        return KPosition(observation_id, KPOS_FILE_MISSING, **common)
    try:
        log = json.loads(data)
        cycle = log[0]["cycle_raw"]
        position = tuple(float(x) for x in cycle["geometry"]["position"])
        n_cycles = len(log)
    except (ValueError, KeyError, IndexError, TypeError):
        return KPosition(observation_id, KPOS_BAD_LOG, **common)
    if len(position) != K_POSITION_DIMENSION:
        return KPosition(
            observation_id,
            KPOS_BAD_LOG,
            reason=f"dimension {len(position)}",
            n_cycles=n_cycles,
            **common,
        )
    return KPosition(observation_id, KPOS_MEASURED, position=position, n_cycles=n_cycles, **common)


def mapping_table(
    item_set: ItemSet, rows: Sequence[ItemRun], data_dir: str | Path
) -> dict[str, Any]:
    """The evidence table: set sha256, and per item item_id -> capture, text sha256, K position
    (read back from the archive, with its sha256) or why there is none."""
    table: list[dict[str, Any]] = []
    for row in rows:
        entry = row.as_row()
        if row.observation_id is not None and row.outcome == OUT_SAVED:
            found = k_position(data_dir, row.observation_id)
            entry["k_position"] = {
                "status": found.status,
                "reason": found.reason,
                "n_cycles": found.n_cycles,
                "dimension": None if found.position is None else len(found.position),
                "position_sha256": found.position_sha256,
                "position": None if found.position is None else list(found.position),
                "causal_log_sha256": found.causal_log_sha256,
            }
        table.append(entry)
    return {
        "table_version": "rsa-x1-table/1",
        "set_id": item_set.set_id,
        "set_sha256": item_set.sha256,
        "n_items": len(item_set.items),
        "system_message_sha256": None
        if item_set.system is None
        else hashlib.sha256(item_set.system.encode("utf-8")).hexdigest(),
        "items": table,  # not "rows": the evidence check forbids that key (RAW rows)
    }


def template_condition(prompt_ids: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    """What the prompts of a set share, from their token ids alone (AC-R1): the length of the common
    prefix and suffix, and whether every prompt ends with the same token. Row 0 of a capture is the
    state at the last prompt token, so one last token for all items means one condition for all."""
    if not prompt_ids:
        raise ValueError("no prompts")
    lists = {key: [int(i) for i in ids] for key, ids in prompt_ids.items()}
    ordered = list(lists.values())
    prefix = 0
    while all(len(x) > prefix for x in ordered) and len({x[prefix] for x in ordered}) == 1:
        prefix += 1
    suffix = 0
    while all(len(x) > suffix for x in ordered) and len({x[-1 - suffix] for x in ordered}) == 1:
        suffix += 1
    last = sorted({x[-1] for x in ordered})
    return {
        "n_items": len(ordered),
        "n_tokens": {key: len(ids) for key, ids in lists.items()},
        "common_prefix_tokens": prefix,
        "common_suffix_tokens": suffix,
        "last_token_ids": last,
        "all_items_end_with_the_same_token": len(last) == 1,
    }
