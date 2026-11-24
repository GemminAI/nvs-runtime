"""The metrics M1-M5 of the Phase 2 pre-registration (v0.1), for 72 items x 7 anchor sets x methods.

Everything is a description: counts as fractions, rates, agreement and kappa, AUC, item ids of
disagreements. A denominator of 0 gives ``None`` (never 0 %). The items with the L_anchor value
`ambiguous` are left out of the aggregates of M1 and their number is given. Nothing here ranks the
methods or judges a radius.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from nvs_runtime.hidden_trajectory.phase2 import geometry as gm
from nvs_runtime.hidden_trajectory.phase2.labels import Labels, category_of, role_of
from nvs_runtime.hidden_trajectory.rsa import compare as rc

METHODS: Final = ("R", "G1_k3", "G1_k5", "G1_k7", "G2_k5")
ROLES: Final = ("near", "far", "reversed")
Array = NDArray[np.float64]


def fraction(count: int, of: int) -> dict[str, Any]:
    return {
        "count": int(count),
        "of": int(of),
        "fraction": f"{count}/{of}",
        "share": (count / of) if of else None,
    }


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1.0 else (po - pe) / (1 - pe)


def _auc(first: Sequence[float], other: Sequence[float]) -> float | None:
    if not first or not other:
        return None
    return float(rc.auc_nearer(np.asarray(first, dtype=float), np.asarray(other, dtype=float)))


def _summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def evaluate(
    points: Array,
    ids: Sequence[str],
    labels: Labels,
    *,
    n_references: int = gm.N_REFERENCES,
) -> dict[str, Any]:
    """``points``: (75, 64); ``ids``: the references first, then the 72 items in set order."""
    n_points = points.shape[0]
    items = list(ids[n_references:])
    n_items = len(items)
    distances = rc.pairwise_euclidean(points)
    xs = list(range(n_references, n_points))  # the index of each item
    set_names = ["+".join(f"ref{a + 1}" for a in s) for s in gm.ANCHOR_SETS]
    same = [labels.l_anchor[i] == "same" for i in ids]
    k_of = {"G1_k3": 3, "G1_k5": 5, "G1_k7": 7}

    dec: dict[str, list[NDArray[np.bool_]]] = {m: [] for m in METHODS}
    dec_alt: dict[str, list[NDArray[np.bool_]]] = {m: [] for m in METHODS}
    d_radius: list[Array] = []
    ranks: list[NDArray[np.int64]] = []
    ranks_alt: list[NDArray[np.int64]] = []
    for anchors in gm.ANCHOR_SETS:
        radius = [gm.decide_radius(distances, x, anchors) for x in xs]
        dec["R"].append(np.array([r[0] for r in radius]))
        dec_alt["R"].append(dec["R"][-1])
        d_radius.append(np.array([r[1] for r in radius]))
        for name, k in k_of.items():
            dec[name].append(np.array([gm.decide_graph(distances, x, anchors, k) for x in xs]))
            dec_alt[name].append(
                np.array(
                    [gm.decide_graph(distances, x, anchors, k, descending_ties=True) for x in xs]
                )
            )
        labelled = np.array([gm.decide_labelled(distances, x, same) for x in xs])
        labelled_alt = np.array(
            [gm.decide_labelled(distances, x, same, descending_ties=True) for x in xs]
        )
        dec["G2_k5"].append(labelled)
        dec_alt["G2_k5"].append(labelled_alt)
        ranks.append(np.array([gm.anchor_rank(distances, x, anchors) for x in xs]))
        ranks_alt.append(
            np.array([gm.anchor_rank(distances, x, anchors, descending_ties=True) for x in xs])
        )

    l_anchor = [labels.l_anchor[i] for i in items]
    groups = {
        v: [n for n, lab in enumerate(l_anchor) if lab == v] for v in ("same", "opposite", "other")
    }
    n_ambiguous = sum(1 for lab in l_anchor if lab == "ambiguous")
    categories = sorted({category_of(i) for i in items})
    category = [category_of(i) for i in items]
    role = [role_of(i) for i in items]

    # ---- ties ----------------------------------------------------------------------------------
    ties: dict[str, Any] = {
        "tie_rule": "equal distances by index ascending (refs first, then the set order)"
    }
    for m in METHODS:
        per_set = []
        for s_index, name in enumerate(set_names):
            changed = int(np.sum(dec[m][s_index] != dec_alt[m][s_index]))
            row: dict[str, Any] = {
                "anchor_set": name,
                "n_decisions_changed_by_the_reverse_tie_rule": changed,
            }
            if m in k_of:
                row["n_boundary_ties"] = sum(
                    1 for x in xs if gm.boundary_tie(distances, x, k_of[m])
                )
            elif m == "G2_k5":
                row["n_boundary_ties"] = sum(1 for x in xs if gm.boundary_tie(distances, x, 5))
            per_set.append(row)
        ties[m] = per_set
    ties["rank_score"] = [
        {
            "anchor_set": n,
            "n_ranks_changed_by_the_reverse_tie_rule": int(np.sum(ranks[i] != ranks_alt[i])),
        }
        for i, n in enumerate(set_names)
    ]
    ties["any_effect"] = any(
        r["n_decisions_changed_by_the_reverse_tie_rule"] for m in METHODS for r in ties[m]
    ) or any(r["n_ranks_changed_by_the_reverse_tie_rule"] for r in ties["rank_score"])

    # ---- M1 -----------------------------------------------------------------------------------
    m1: dict[str, Any] = {"n_ambiguous_left_out": n_ambiguous, "per_method": {}}
    for m in METHODS:
        per_set = []
        for s_index, name in enumerate(set_names):
            inside = dec[m][s_index]
            by_label = {
                v: fraction(int(sum(inside[n] for n in idx)), len(idx)) for v, idx in groups.items()
            }
            cells = {
                f"{c}-{r}": fraction(
                    int(
                        sum(inside[n] for n in range(n_items) if category[n] == c and role[n] == r)
                    ),
                    sum(1 for n in range(n_items) if category[n] == c and role[n] == r),
                )
                for c in categories
                for r in ROLES
            }
            entry: dict[str, Any] = {
                "anchor_set": name,
                "inside_by_L_anchor": by_label,
                "cells": cells,
            }
            if m == "R":
                score = d_radius[s_index]
            elif m == "G1_k5":
                score = ranks[s_index].astype(float)
            else:
                score = None
            if score is not None:
                entry["auc_same_vs_opposite"] = _auc(
                    [score[n] for n in groups["same"]], [score[n] for n in groups["opposite"]]
                )
                entry["auc_same_vs_other"] = _auc(
                    [score[n] for n in groups["same"]], [score[n] for n in groups["other"]]
                )
                entry["auc_score"] = (
                    "-d(x) (R)" if m == "R" else "-(rank of the earliest anchor) (G1)"
                )
            per_set.append(entry)
        m1["per_method"][m] = per_set
    m1["note"] = (
        "the G1 rank score does not depend on k: its AUC is given once, under G1_k5; "
        "G2 does not depend on the anchor set"
    )

    # ---- M2 -----------------------------------------------------------------------------------
    m2: dict[str, Any] = {}
    for m in METHODS:
        per_set = []
        for s_index, name in enumerate(set_names):
            inside = dec[m][s_index]
            c1 = {
                r: fraction(
                    int(
                        sum(
                            inside[n]
                            for n in range(n_items)
                            if category[n] == "c1" and role[n] == r
                        )
                    ),
                    sum(1 for n in range(n_items) if category[n] == "c1" and role[n] == r),
                )
                for r in ROLES
            }
            others: dict[str, Any] = {}
            for c in categories:
                if c == "c1":
                    continue
                values = sorted(
                    {str(labels.l_rel[items[n]]) for n in range(n_items) if category[n] == c}
                )
                others[c] = {
                    v: fraction(
                        int(
                            sum(
                                inside[n]
                                for n in range(n_items)
                                if category[n] == c and str(labels.l_rel[items[n]]) == v
                            )
                        ),
                        sum(
                            1
                            for n in range(n_items)
                            if category[n] == c and str(labels.l_rel[items[n]]) == v
                        ),
                    )
                    for v in values
                }
            per_set.append(
                {"anchor_set": name, "c1_by_role": c1, "other_categories_by_L_rel": others}
            )
        m2[m] = per_set

    # ---- M3 -----------------------------------------------------------------------------------
    m3: dict[str, Any] = {}
    for m in METHODS:
        rates = [
            float(np.mean(dec[m][i] == dec[m][j]))
            for i in range(len(set_names))
            for j in range(i + 1, len(set_names))
        ]
        m3[m] = {"n_pairs": len(rates), **(_summary(rates) or {})}

    # ---- M4 -----------------------------------------------------------------------------------
    m4 = []
    for s_index, name in enumerate(set_names):
        r_dec, g_dec = dec["R"][s_index], dec["G1_k5"][s_index]
        disagree = [items[n] for n in range(n_items) if r_dec[n] != g_dec[n]]
        m4.append(
            {
                "anchor_set": name,
                "agreement": fraction(int(np.sum(r_dec == g_dec)), n_items),
                "cohen_kappa": cohen_kappa(list(r_dec), list(g_dec)),
                "R_inside_G1_outside": sum(1 for n in range(n_items) if r_dec[n] and not g_dec[n]),
                "R_outside_G1_inside": sum(1 for n in range(n_items) if g_dec[n] and not r_dec[n]),
                "n_inside_R": int(np.sum(r_dec)),
                "n_inside_G1_k5": int(np.sum(g_dec)),
                "disagreeing_item_ids": disagree,
            }
        )

    # ---- M5 -----------------------------------------------------------------------------------
    item_indices = list(range(n_references, n_points))
    m5_items = []
    shares, expecteds = [], []
    for n, x in enumerate(item_indices):
        if labels.l_rel[items[n]] != "same":
            continue
        pool = [j for j in item_indices if j != x]  # the other 71 items (references left out)
        order = sorted(pool, key=lambda j: (float(distances[x, j]), j))
        nearest = order[: gm.K_MAIN]
        eligible = {
            j
            for j in pool
            if category_of(ids[j]) == category_of(ids[x]) and labels.l_rel[ids[j]] == "same"
        }
        share = sum(1 for j in nearest if j in eligible) / gm.K_MAIN
        expected = len(eligible) / len(pool)
        shares.append(share)
        expecteds.append(expected)
        m5_items.append(
            {
                "item_id": ids[x],
                "share_same_category_L_rel_same": share,
                "expected_for_random": expected,
            }
        )
    m5 = {
        "n_items": len(m5_items),
        "k": gm.K_MAIN,
        "candidates": "the other 71 items (the references are not candidates)",
        "share": _summary(shares),
        "mean_share": float(np.mean(shares)) if shares else None,
        "mean_expected_for_random_neighbours": float(np.mean(expecteds)) if expecteds else None,
        "per_item": m5_items,
        "reading": "an indicator of the construction (L_rel follows the role of the item id), "
        "not a human confirmation",
    }

    return {
        "n_items": n_items,
        "anchor_sets": set_names,
        "methods": list(METHODS),
        "radius": gm.RADIUS,
        "distance": "Euclidean on the 64-D unit vectors",
        "ties": ties,
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "M4": m4,
        "M5": m5,
    }
