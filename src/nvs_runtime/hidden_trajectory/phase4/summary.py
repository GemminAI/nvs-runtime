"""The aggregates Q1-Q4 of the Phase 4 pre-registration (v0.8), from stored K cycles only.

Input per series (one prompt): the ``cycle_raw`` of each generated token in order, the anchor
counts of the board record of each cycle, the finish reason and the group. Output: numbers and
classes; no interpretation. The unit of aggregation is the prompt: cycles are never pooled across
prompts.

* Q1: the tier of the first cycle whose Risk is defined; a prompt whose Risk is never defined is
  its own category (not dropped from a denominator); counts are always given as fractions; the 0.75
  rule is applied only when each group has at least 3 prompts with a defined Risk, otherwise the
  shares are reference values and no verdict field is produced.
* Q2: the cycle at which a dynamic anchor was added (anchor counts) and the spread of the positions
  of that cycle, recomputed from the stored positions (the record keeps no reason); for a series
  with no anchor, the spread of each of the first cycles and a class.
* Q3: one of five classes per series.
* Q4: per prompt, the median and maximum of ``smoothed`` over the defined cycles; the share of L7
  over all cycles (a) and over the defined cycles (b); the tier of the first defined cycle. A group
  is summarised by the median and the interquartile range (numpy percentile 25 and 75, linear) of
  each value, with the number of prompts left out because the value is "none" (``None``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

HIGH_TIER: Final = 5
L7: Final = 7
RULE_SHARE: Final = 0.75
RULE_MIN_DEFINED: Final = 3
BOOTSTRAP_CYCLES: Final = 3
STABILITY: Final = 0.5
HISTORY: Final = 8  # history_capacity of the default settings: max(window 8, risk_window 8, 3 + 2)
CLASSES: Final = (
    "not_reached",
    "recovered",
    "L7_kept_ended_by_eos",
    "L7_kept_ended_by_limit",
    "other_end",
)


@dataclass(frozen=True)
class Series:
    prompt_id: str
    group: str
    cycles: Sequence[Mapping[str, Any]]  # cycle_raw, in token order
    anchor_counts: Sequence[Mapping[str, int]]  # the board record of each cycle (after the cycle)
    finish_reason: str | None
    end_note: str | None = (
        None  # an exception type or another reason for a stop that is not EOS/limit
    )
    n_gen: int = field(default=0)


def _defined(cycle: Mapping[str, Any]) -> bool:
    return bool(cycle["risk"]["defined"]) and cycle["risk"]["state"] != "UNDEFINED"


def _tier(cycle: Mapping[str, Any]) -> int:
    return int(cycle["control"]["tier"])


def first_defined(series: Series) -> int | None:
    for index, cycle in enumerate(series.cycles):
        if _defined(cycle):
            return index
    return None


def spread(positions: Sequence[Sequence[float]]) -> float | None:
    """mean_i ||x_i - centroid||^2 (``crystallize._local_variance``); None for fewer than 2."""
    if len(positions) < 2:
        return None
    x = np.asarray(positions, dtype=np.float64)
    return float(np.mean(np.sum((x - x.mean(axis=0)) ** 2, axis=1)))


def anchor_formation(series: Series) -> dict[str, Any]:
    """Q2 for one series. The record keeps no reason text, so the spread is recomputed from the
    stored positions of the positions window (the last ``HISTORY`` positions up to the cycle)."""
    counts = list(series.anchor_counts)
    positions = [c["geometry"]["position"] for c in series.cycles]
    previous = 0
    formed: int | None = None
    for index, c in enumerate(counts):
        dynamic = int(c.get("dynamic", 0))
        if dynamic > previous and formed is None:
            formed = index
        previous = dynamic

    def at(index: int) -> float | None:
        return spread(positions[max(0, index - HISTORY + 1) : index + 1])

    out: dict[str, Any] = {
        "recorded_reason": None,
        "spread_is": "recomputed from the stored positions of the window",
        "anchor_formed_cycle": formed,
        "anchors_at_end": dict(counts[-1]) if counts else None,
    }
    if formed is not None:
        value = at(formed)
        out["spread_at_formation"] = value
        out["spread_within_stability"] = None if value is None else value <= STABILITY
        out["cycle_count_at_formation"] = formed
    else:
        first = [at(i) for i in range(min(BOOTSTRAP_CYCLES, len(positions)))]
        out["spread_in_the_first_cycles"] = first
        if len(positions) <= 1:
            klass = "fewer_than_two_positions"
        elif any(v is not None and v <= STABILITY for v in first[1:]):
            klass = "spread_within_stability_but_no_anchor"
        else:
            klass = "spread_above_stability_in_the_bootstrap_cycles"
        out["not_formed_class"] = klass
    return out


def classify_l7(series: Series) -> dict[str, Any]:
    """Q3: exactly one class (decided before the results were seen)."""
    tiers = [_tier(c) for c in series.cycles]
    n = len(tiers)
    reached = [i for i, t in enumerate(tiers) if t >= L7]
    out: dict[str, Any] = {"n_cycles": n, "finish_reason": series.finish_reason}
    if n == 0:
        out.update(
            klass="other_end", reason=series.end_note or "no generated cycle", reached_L7=False
        )
        return out
    if not reached:
        out.update(klass="not_reached", reached_L7=False)
        return out
    first = reached[0]
    after = [i for i in range(first + 1, n) if tiers[i] < L7]
    out.update(reached_L7=True, first_L7_cycle=first, cycles_from_first_L7_to_end=n - first)
    if after:
        back = [i for i in range(after[0] + 1, n) if tiers[i] >= L7]
        out.update(
            klass="recovered",
            first_transition_cycle=after[0],
            tier_after_transition=tiers[after[0]],
            returned_to_L7=bool(back),
            returned_to_L7_cycle=back[0] if back else None,
        )
    elif series.end_note is not None:
        out.update(klass="other_end", reason=series.end_note)
    elif series.finish_reason == "eos":
        out.update(klass="L7_kept_ended_by_eos")
    elif series.finish_reason == "length":
        out.update(klass="L7_kept_ended_by_limit")
    else:
        out.update(klass="other_end", reason=f"finish_reason {series.finish_reason}")
    return out


def per_prompt(series: Series) -> dict[str, Any]:
    """Q1 and Q4 for one prompt."""
    cycles = list(series.cycles)
    n = len(cycles)
    defined = [c for c in cycles if _defined(c)]
    smoothed = [float(c["risk"]["smoothed"]) for c in defined]
    first = first_defined(series)
    tiers = [_tier(c) for c in cycles]
    return {
        "prompt_id": series.prompt_id,
        "group": series.group,
        "n_cycles": n,
        "n_defined_cycles": len(defined),
        "first_defined_cycle": first,
        "first_defined_tier": None if first is None else tiers[first],
        "risk_never_defined": first is None,
        "smoothed_median": float(np.median(smoothed)) if smoothed else None,
        "smoothed_max": float(np.max(smoothed)) if smoothed else None,
        "l7_share_all_cycles": (sum(t >= L7 for t in tiers) / n) if n else None,
        "l7_share_defined_cycles": (
            sum(_tier(c) >= L7 for c in defined) / len(defined) if defined else None
        ),
        "l7_cycles": sum(t >= L7 for t in tiers),
    }


def _fraction(num: int, den: int) -> dict[str, Any]:
    return {
        "count": num,
        "of": den,
        "fraction": f"{num}/{den}",
        "share": (num / den) if den else None,
    }


def q1_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per group: fractions as counts, the tier distribution of the first defined cycle, and (only
    when every group has at least ``RULE_MIN_DEFINED`` defined prompts) the 0.75 rule's fields."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r["group"]), []).append(r)
    out: dict[str, Any] = {"groups": {}}
    for name, items in groups.items():
        defined = [r for r in items if not r["risk_never_defined"]]
        high = [r for r in defined if int(r["first_defined_tier"]) >= HIGH_TIER]
        distribution = {
            f"L{t}": sum(1 for r in defined if r["first_defined_tier"] == t) for t in range(8)
        }
        out["groups"][name] = {
            "n_prompts": len(items),
            "risk_defined": _fraction(len(defined), len(items)),
            "risk_never_defined": _fraction(len(items) - len(defined), len(items)),
            "L5_or_above_of_defined": _fraction(len(high), len(defined)),
            "L5_or_above_of_all": _fraction(len(high), len(items)),
            "first_defined_tier_distribution": distribution,
            "n_never_defined": len(items) - len(defined),
        }
    applicable = bool(out["groups"]) and all(
        g["risk_defined"]["count"] >= RULE_MIN_DEFINED for g in out["groups"].values()
    )
    out["rule_0_75"] = {
        "applicable": applicable,
        "min_defined_per_group": RULE_MIN_DEFINED,
        "reason_not_applicable": None
        if applicable
        else "a group has fewer than 3 prompts with a defined Risk: "
        "the shares are reference values",
    }
    if applicable:
        shares = {k: g["L5_or_above_of_defined"]["share"] for k, g in out["groups"].items()}
        out["rule_0_75"]["shares"] = shares
        out["rule_0_75"]["all_groups_at_least_0_75"] = all(
            s is not None and s >= RULE_SHARE for s in shares.values()
        )
        out["rule_0_75"]["groups_below_0_75"] = sorted(
            k for k, s in shares.items() if s is None or s < RULE_SHARE
        )
    return out


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    kept = [float(v) for v in values if v is not None]
    left_out = len(values) - len(kept)
    if not kept:
        return {"n": 0, "median": None, "iqr": None, "n_left_out_none": left_out}
    q1, q3 = np.percentile(kept, [25, 75])
    return {
        "n": len(kept),
        "median": float(np.median(kept)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "n_left_out_none": left_out,
    }


def q4_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r["group"]), []).append(r)
    metrics = ("smoothed_median", "smoothed_max", "l7_share_all_cycles", "l7_share_defined_cycles")
    return {
        name: {m: _summary([r[m] for r in items]) for m in metrics}
        for name, items in groups.items()
    }


def q3_counts(classes: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {k: 0 for k in CLASSES}
    for c in classes:
        counts[str(c["klass"])] += 1
    return counts
