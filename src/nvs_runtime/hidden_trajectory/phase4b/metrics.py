"""Phase 4B metrics (pre-registration section 4), from replayed Risk and Control.

Everything is a description: no ranking, no label other than "moved a lot" (section 5: the number of
series in the Q3 classes "L7 kept" (EOS and limit) changes by 3 or more from V0, which has 9 of 12).
The aggregates of Q1, Q3 and Q4 are those of ``phase4.summary`` (the 0.75 rule is not applied: its
shares stay as reference values). A denominator of 0 gives ``None`` (never 0 %).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from nvs_runtime.hidden_trajectory.phase4 import summary as sm

L7_KEPT: Final = ("L7_kept_ended_by_eos", "L7_kept_ended_by_limit")
MOVED_BY: Final = 3


def replayed_series(base: sm.Series, replayed: Sequence[Mapping[str, Any]]) -> sm.Series:
    """``base`` with the Risk and Control of a replay (the stored geometry is kept)."""
    cycles = [
        {"risk": r["risk"], "control": r["control"], "geometry": c["geometry"]}
        for c, r in zip(base.cycles, replayed, strict=True)
    ]
    return sm.Series(
        base.prompt_id,
        base.group,
        cycles,
        base.anchor_counts,
        base.finish_reason,
        base.end_note,
        base.n_gen,
    )


def tiers(series: sm.Series) -> list[int]:
    return [int(c["control"]["tier"]) for c in series.cycles]


def transitions(series: sm.Series) -> dict[str, int]:
    t = tiers(series)
    return {
        "up": sum(1 for a, b in zip(t, t[1:], strict=False) if b > a),
        "down": sum(1 for a, b in zip(t, t[1:], strict=False) if b < a),
    }


def share_different_from(series: sm.Series, default: sm.Series) -> float | None:
    a, b = tiers(series), tiers(default)
    if not a or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b, strict=True) if x != y) / len(a)


def per_series(series: sm.Series, default: sm.Series) -> dict[str, Any]:
    row = sm.per_prompt(series)
    klass = sm.classify_l7(series)
    base_klass = sm.classify_l7(default)
    first_l7 = klass.get("first_L7_cycle")
    return {
        **row,
        "q3_class": klass["klass"],
        "q3_class_v0": base_klass["klass"],
        "q3_class_differs_from_v0": klass["klass"] != base_klass["klass"],
        "first_L7_cycle": first_l7,
        "first_L7_cycle_v0": base_klass.get("first_L7_cycle"),
        "transitions": transitions(series),
        "transitions_v0": transitions(default),
        "share_of_cycles_with_a_tier_different_from_v0": share_different_from(series, default),
        "n_cycles_with_a_tier_different_from_v0": sum(
            1 for x, y in zip(tiers(series), tiers(default), strict=True) if x != y
        ),
    }


def variant_summary(
    rows: Sequence[Mapping[str, Any]], v0_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    q1 = sm.q1_groups(rows)
    q1.pop("rule_0_75", None)  # not applied per variant (pre-registration section 4)
    counts = sm.q3_counts([{"klass": r["q3_class"]} for r in rows])
    kept = sum(counts[k] for k in L7_KEPT)
    kept_v0 = sum(1 for r in v0_rows if r["q3_class"] in L7_KEPT)
    return {
        "Q1": q1,
        "Q3_counts": counts,
        "Q4_groups": sm.q4_groups(rows),
        "n_series_q3_class_differs_from_v0": sum(1 for r in rows if r["q3_class_differs_from_v0"]),
        "L7_kept_series": kept,
        "L7_kept_series_v0": kept_v0,
        "change_of_L7_kept_series": kept - kept_v0,
        "moved_a_lot": abs(kept - kept_v0) >= MOVED_BY,
        "moved_a_lot_rule": f"the number of series in the classes {list(L7_KEPT)} changes by "
        f"{MOVED_BY} or more from V0",
    }
