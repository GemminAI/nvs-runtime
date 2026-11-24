"""RSA-X2: the fixed item set, the K positions read back from the capture archive, and the
string-side
positions, as plain arrays for ``rsa.compare``.

* ``load_fixed_set`` reads the draft/fixed set file as it is and refuses a file whose sha256
  is not the
  registered one.
* ``read_positions`` reads the K position (row 0, 64-D) of every capture of a capture archive by
  ``item_id``, through ``item_set.k_position`` (the stored causal log of the K path), never from a
  run's memory. The item id of a capture is the one the entry gave its ``contract_session_id``
  (``rsa-x1-<item_id>``), which the capture record keeps.
* ``integrity_report`` says only numbers: counts, dimensions, finiteness, the smallest and largest
  norm, and which item ids are missing or extra. No position, and no hash of one.
* ``string_positions`` calls ``pipeline.project(text)`` as a calculation (a read of the kernel's
  projector). Nothing is put into a kernel session, no anchor, no memory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from nvs_runtime.hidden_trajectory import item_set as isx
from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive

SESSION_PREFIX: Final = "rsa-x1-"
FIXED_SET_SHA256: Final = "33f4c0498bb3dd7566d4d44fafb6a8a7f3b1fa6822cc49b7fb5ae51703754b81"
N_ITEMS: Final = 72
N_REFERENCES: Final = 3
DIMENSION: Final = 64


class ReadoutError(ValueError):
    """The set file or an archive is not what the registration says."""


@dataclass(frozen=True, slots=True)
class FixedSet:
    sha256: str
    item_ids: tuple[str, ...]  # the 72 items, in file order
    reference_ids: tuple[str, ...]  # the references, in file order
    texts: Mapping[str, str]
    roles: Mapping[str, str]
    categories: Mapping[str, str]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.item_ids + self.reference_ids


def load_fixed_set(path: str | Path, *, expected_sha256: str = FIXED_SET_SHA256) -> FixedSet:
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise ReadoutError(f"the set file hashes to {digest}, not the registered {expected_sha256}")
    doc = json.loads(data.decode("utf-8"))
    items, references = list(doc["items"]), list(doc.get("references", []))
    if len(items) != N_ITEMS or len(references) != N_REFERENCES:
        raise ReadoutError(f"expected {N_ITEMS} items and {N_REFERENCES} references")
    entries = items + references
    ids = [str(e["item_id"]) for e in entries]
    if len(set(ids)) != len(ids):
        raise ReadoutError("an item_id is repeated")
    return FixedSet(
        sha256=digest,
        item_ids=tuple(ids[:N_ITEMS]),
        reference_ids=tuple(ids[N_ITEMS:]),
        texts={str(e["item_id"]): str(e["text"]) for e in entries},
        roles={str(e["item_id"]): str(e["role"]) for e in entries},
        categories={str(e["item_id"]): str(e["category"]) for e in entries},
    )


def read_positions(
    data_dir: str | Path, *, prefix: str = SESSION_PREFIX, point: str = isx.POINT
) -> tuple[dict[str, isx.KPosition], dict[str, Any]]:
    """``(positions by item_id, notes)`` for every capture record of the archive in ``data_dir``.
    ``notes`` names what could not be mapped: a record whose session id has no prefix, and an
    item id
    that has more than one capture (the latest record of each observation is used; with more
    than one
    observation for an item the item is listed and its position is not given)."""
    archive = CaptureArchive(data_dir)
    latest: dict[str, dict[str, Any]] = {}
    for record in archive.records():
        latest[str(record.get("observation_id"))] = record
    by_item: dict[str, list[str]] = {}
    unmapped = 0
    for observation_id, record in latest.items():
        name = str(record.get("contract_session_id") or "")
        if not name.startswith(prefix) or len(name) == len(prefix):
            unmapped += 1
            continue
        by_item.setdefault(name[len(prefix) :], []).append(observation_id)
    positions: dict[str, isx.KPosition] = {}
    repeated = sorted(i for i, obs in by_item.items() if len(obs) > 1)
    for item_id, observations in by_item.items():
        if len(observations) == 1:
            positions[item_id] = isx.k_position(data_dir, observations[0], point=point)
    return positions, {
        "records": len(latest),
        "unmapped_records": unmapped,
        "items_with_more_than_one_capture": repeated,
    }


def integrity_report(
    positions: Mapping[str, isx.KPosition],
    expected_ids: Sequence[str],
    *,
    notes: Mapping[str, Any] | None = None,
    unit_norm_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Numbers only: how many were read, their dimension and finiteness, the smallest and
    largest norm,
    and whether the item ids agree with ``expected_ids`` exactly."""
    expected = set(expected_ids)
    found = set(positions)
    measured = {
        i: p
        for i, p in positions.items()
        if p.status == isx.KPOS_MEASURED and p.position is not None
    }
    dims = sorted({len(p.position or ()) for p in measured.values()})
    finite = all(bool(np.all(np.isfinite(np.asarray(p.position)))) for p in measured.values())
    norms = [
        float(np.linalg.norm(np.asarray(p.position, dtype=np.float64))) for p in measured.values()
    ]
    statuses: dict[str, int] = {}
    for p in positions.values():
        statuses[p.status] = statuses.get(p.status, 0) + 1
    cycles = sorted({p.n_cycles for p in measured.values() if p.n_cycles is not None})
    report: dict[str, Any] = {
        "n_expected": len(expected),
        "n_found": len(found),
        "n_measured": len(measured),
        "status_counts": statuses,
        "dimensions": dims,
        "all_finite": bool(finite),
        "n_cycles_values": cycles,
        "norm_min": min(norms) if norms else None,
        "norm_max": max(norms) if norms else None,
        "unit_norm_within_tolerance": bool(norms)
        and all(abs(n - 1.0) <= unit_norm_tolerance for n in norms),
        "unit_norm_tolerance": unit_norm_tolerance,
        "missing_item_ids": sorted(expected - found),
        "extra_item_ids": sorted(found - expected),
        "item_ids_match_exactly": found == expected and len(found) == len(expected_ids),
        "notes": dict(notes or {}),
    }
    report["ok"] = bool(
        report["item_ids_match_exactly"]
        and len(measured) == len(expected)
        and dims == [DIMENSION]
        and finite
        and not report["notes"].get("items_with_more_than_one_capture")
    )
    return report


def position_array(
    positions: Mapping[str, isx.KPosition], item_ids: Sequence[str]
) -> NDArray[np.float64]:
    """The positions of ``item_ids`` in that order, as an (n, 64) array (a missing one raises)."""
    rows = []
    for item_id in item_ids:
        p = positions.get(item_id)
        if p is None or p.position is None:
            raise ReadoutError(f"no measured position for {item_id}")
        rows.append(p.position)
    return np.asarray(rows, dtype=np.float64)


def string_positions(
    texts: Mapping[str, str],
    item_ids: Sequence[str],
    project: Callable[[str], Any],
) -> NDArray[np.float64]:
    """The string-side positions of ``item_ids``, in that order: ``project(text)`` for each text,
    read
    as a calculation. ``project`` is required: the kernel's projector comes from the Developer
    Tools (``kernel_project``), not from here."""
    return np.asarray(
        [np.asarray(project(texts[i]), dtype=np.float64) for i in item_ids], dtype=np.float64
    )
