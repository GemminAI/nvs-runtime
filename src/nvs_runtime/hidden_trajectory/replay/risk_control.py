"""Phase 4, method B, the part that stays in nvs-runtime: read the stored cycles of a K record
and compare them with a replay of Risk and Control.

``read_cycles`` reads the ``cycle_raw`` of each cycle of a stored K causal log from a capture
archive (``CaptureArchive``). ``compare`` sets every field of the stored Risk and Control against
the replayed one and reports counts, the largest differences and field names; positions, directions
and other vectors are never returned or written. The replay itself calls the kernel's own classes
and lives in the Developer Tools.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive

POINT: Final = "final_norm"
RISK_FIELDS: Final = (
    "raw",
    "smoothed",
    "state",
    "previous_state",
    "defined",
    "dominant",
    "escalated",
)
COMPONENT_FIELDS: Final = ("value", "weight", "defined", "detail")
CONTROL_FIELDS: Final = (
    "tier",
    "previous_tier",
    "intervention",
    "confirmation_count",
    "reason",
    "escalated",
    "requires_approval",
    "authorized",
    "actionable",
)
GAIN_FIELDS: Final = ("guidance", "damping", "throttle", "containment")
CHECKPOINT_FIELDS: Final = ("checkpoint_id", "risk", "timestamp_ns")


class ReplayError(ValueError):
    """The stored record is not what the replay needs."""


# ---- reading ---------------------------------------------------------------------------------


def read_cycles(
    data_dir: str | Path, observation_id: str | None = None, *, point: str = POINT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(cycle_raw of each cycle in token order, info)`` of one capture of the archive. With no
    ``observation_id`` the archive must hold exactly one capture."""
    archive = CaptureArchive(data_dir)
    latest: dict[str, dict[str, Any]] = {}
    for record in archive.records():
        latest[str(record.get("observation_id"))] = record
    if observation_id is None:
        if len(latest) != 1:
            raise ReplayError(f"the archive holds {len(latest)} captures: name one")
        observation_id = next(iter(latest))
    found = latest.get(observation_id)
    if found is None:
        raise ReplayError("no such capture")
    k = ((found.get("positions") or {}).get(point) or {}).get("k_path")
    if (
        not isinstance(k, Mapping)
        or k.get("status") != "measured"
        or not k.get("causal_log_sha256")
    ):
        raise ReplayError("the position has no measured K path")
    sha = str(k["causal_log_sha256"])
    data = archive.read_k(sha)
    if data is None:
        raise ReplayError("the causal log is not in the archive")
    if hashlib.sha256(data).hexdigest() != sha:
        raise ReplayError("the causal log does not hash to its name")
    log = json.loads(data)
    ordered = sorted(log, key=lambda item: int(item["token_index"]))
    cycles = [item["cycle_raw"] for item in ordered]
    info = {
        "observation_id": observation_id,
        "causal_log_sha256": sha,
        "n_cycles": len(cycles),
        "anchor_policy_sha256": ((k.get("policies") or {}).get("anchor")),
        "token_indices": [int(item["token_index"]) for item in ordered],
    }
    return cycles, info


# ---- the comparison --------------------------------------------------------------------------


class _Diff:
    """Largest absolute difference per field group, and the fields that did not match."""

    def __init__(self) -> None:
        self.max_abs: dict[str, float] = {}
        self.compared: dict[str, int] = {}
        self.mismatches: list[dict[str, Any]] = []

    def check(self, cycle: int, group: str, field: str, stored: Any, again: Any) -> None:
        self.compared[group] = self.compared.get(group, 0) + 1
        numeric = (
            isinstance(stored, (int, float))
            and isinstance(again, (int, float))
            and not isinstance(stored, bool)
            and not isinstance(again, bool)
        )
        if numeric:
            diff = abs(float(stored) - float(again))
            self.max_abs[group] = max(self.max_abs.get(group, 0.0), diff)
            same = stored == again
        else:
            same = stored == again and type(stored) is type(again)
            if not same and (stored is None) != (again is None):
                self.max_abs[group] = float("inf")
            else:
                self.max_abs.setdefault(group, 0.0)
        if not same:
            self.mismatches.append({"cycle": cycle, "field": f"{group}.{field}"})


def compare(
    cycles: Sequence[Mapping[str, Any]], replayed: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Every field of the stored Risk and Control against the recomputed one. Numbers and field
    names
    only; the largest difference per group and the list of fields that did not match."""
    if len(cycles) != len(replayed):
        raise ReplayError("different number of cycles")
    diff = _Diff()
    for index, (stored, again) in enumerate(zip(cycles, replayed, strict=True)):
        s_risk, a_risk = stored["risk"], again["risk"]
        for name in RISK_FIELDS:
            diff.check(index, "risk", name, s_risk[name], a_risk[name])
        diff.check(
            index,
            "risk",
            "barrier_potential",
            s_risk["barrier_potential"],
            a_risk["barrier_potential"],
        )
        diff.check(
            index, "risk", "n_components", len(s_risk["components"]), len(a_risk["components"])
        )
        for number, (sc, ac) in enumerate(
            zip(s_risk["components"], a_risk["components"], strict=False)
        ):
            diff.check(index, "component", f"{number}.kind", sc["kind"], ac["kind"])
            for name in COMPONENT_FIELDS:
                diff.check(index, "component", f"{number}.{name}", sc[name], ac[name])
        s_ctl, a_ctl = stored["control"], again["control"]
        for name in CONTROL_FIELDS:
            diff.check(index, "control", name, s_ctl[name], a_ctl[name])
        for name in GAIN_FIELDS:
            diff.check(index, "gains", name, s_ctl["gains"][name], a_ctl["gains"][name])
        s_ck, a_ck = s_ctl["checkpoint"], a_ctl["checkpoint"]
        diff.check(index, "checkpoint", "present", s_ck is not None, a_ck is not None)
        if s_ck is not None and a_ck is not None:
            for name in CHECKPOINT_FIELDS:
                diff.check(index, "checkpoint", name, s_ck[name], a_ck[name])
        stored_direction = stored["control"].get("guidance_direction")
        again_direction = again["guidance_direction"]
        diff.check(
            index,
            "guidance_direction",
            "present",
            stored_direction is not None,
            again_direction["present"],
        )
        if again_direction["present"]:
            diff.check(
                index,
                "guidance_direction",
                "equal_to_stored",
                True,
                again_direction["equal_to_stored"],
            )
    return {
        "n_cycles": len(cycles),
        "n_fields_compared": sum(diff.compared.values()),
        "fields_compared_by_group": diff.compared,
        "max_abs_difference_by_group": {
            k: ("inf" if math.isinf(v) else v) for k, v in diff.max_abs.items()
        },
        "n_mismatches": len(diff.mismatches),
        "mismatched_fields": diff.mismatches,
        "identical": not diff.mismatches,
    }
