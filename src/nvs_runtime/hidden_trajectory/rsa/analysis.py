"""RSA-X4: the registered analysis of the real O and S, as one function over arrays, plus the parts
of the plan that are fixed before a position is read (design v1.0, spec/2).

Nothing here reads a store or a kernel. ``analysis_plan`` is the plan that is written (and hashed)
before any position is read; ``run_x4`` computes every registered number with ``compare`` and
adds the category descriptives, the application of the decision table and the eligibility of X-5.
No threshold, rule or method of ``compare`` is changed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from nvs_runtime.hidden_trajectory.rsa import compare as rc

Array = NDArray[np.float64]

PLAN_VERSION: Final = "rsa-x4-plan/1"
CATEGORY_KINDS: Final = ("near", "far", "reversed")
PROCRUSTES_MIN_PAIRS: Final = 300

LIMITS: Final[tuple[str, ...]] = (
    "X-3b did not find the calibration of the partial Mantel acceptable: in the cases A and C the "
    "generator keeps structure that comes from the word overlap L after L is controlled linearly "
    "(partial Spearman about 0.97 in A, 0.12 in C, with p below 0.01 in every simulation); only D "
    "was acceptable; B, E, A_64 and A_full were not run (calibration unmet).",
    "The power of the method (case E) is unconfirmed: a null result of H2 does not mean there is "
    "no "
    "common structure beyond L.",
    "H2 is therefore exploratory, and a row of the decision table that says 'not confirmed' "
    "carries "
    "the note 'with the limit of power'.",
    "That case D of X-3b is calibrated does not show that the tests are valid on the real data.",
    "One model (Llama 3.1 8B 4-bit), one template, greedy, max_tokens=1, no system message; row 0 "
    "is "
    "the state at the last template token, not at the last word of the sentence.",
    "Meaning is defined by the categories and paraphrase relations of the fixed set; that a hidden "
    "state represents meaning is not tested.",
)


def interpretation_scope(h1: bool, h2: bool, h3: bool) -> dict[str, Any]:
    """The three claims kept apart; only (1) can be confirmatory, (2) is exploratory, (3) is not
    made
    from the significance of H2."""
    return {
        "1_distance_structure_correlation_between_O_and_S": {
            "status": "confirmatory",
            "H1_holds": h1,
            "statement": "H1 is the rank correlation of the O and S distances against the item "
            "permutation null (one-sided p < 0.01)",
        },
        "2_correlation_remains_after_controlling_L_linearly": {
            "status": "exploratory (calibration not acceptable; power of E unconfirmed)",
            "H2_holds": h2,
            "statement": "the partial Spearman controls L as one linear covariate of the ranks; "
            "structure that comes from L non-linearly can remain",
        },
        "3_semantic_commonality_beyond_lexical_similarity": {
            "status": "not claimed",
            "H3_holds": h3,
            "statement": "not asserted from the significance of H2 (or of H3) alone",
        },
    }


def analysis_plan(references: Mapping[str, str], *, code_commit: str) -> dict[str, Any]:
    """The plan, written before any position is read. ``references`` maps a name to the sha256 of
    the
    frozen document it refers to (design v1.0, item set, spec/2, calibration spec)."""
    return {
        "plan_version": PLAN_VERSION,
        "references_sha256": dict(references),
        "code_commit": code_commit,
        "subject": {
            "items": 72,
            "references_for_7_2": ["ref1", "ref2", "ref3"],
            "pairs": 2556,
            "exclusion": "an item whose position cannot be read, is not finite, whose item_id or "
            "text_sha256 does not match is excluded and counted; one exclusion stops the run "
            "before any statistic is computed and the stop is reported",
        },
        "primary_analysis": "raw O",
        "primary_indicators": ["H1", "H2", "H3"],
        "judgment": {
            "H1": "confirmatory (Spearman Mantel one-sided p < 0.01)",
            "H2": "exploratory: calibration not acceptable in X-3b, power of case E unconfirmed "
            "(partial Spearman >= 0.2 and p < 0.01)",
            "H3": "judged by the rule of design v1.0 (k-NN overlap above the null 95th "
            "percentile); "
            "the result of case D alone is not taken as validity on the real data",
            "table": "design v1.0 3B, applied mechanically; rows that include H2 carry "
            "'exploratory', "
            "rows that say 'not confirmed' carry 'with the limit of power'",
        },
        "permutations": {
            "n": rc.N_PERMUTATIONS,
            "rng": "PCG64",
            "seed": rc.SEED,
            "target": "rows and columns of the S matrix together",
            "sides": "one-sided",
        },
        "sensitivity": "centered O (not used in the judgment)",
        "descriptive_only": [
            "triple agreement (all triples, no null)",
            "7.1 distributions by pair kind and AUC",
            "7.2 distances to ref1..ref3 and the share inside 0.65 (diagnostic; the radius is not "
            "changed)",
            "Spearman by word-overlap quartile",
            "retrieval of near paraphrases (K=3)",
            "raw partial Pearson (reference)",
            "per category: Spearman of the O and S distances among its 12 items; median distances "
            "of "
            "near / far / reversed pairs in O and S; no test, no p value",
        ],
        "procrustes_X5": {
            "executed": False,
            "reported": "only whether the conditions of v1.0 are met (H1 and H2 hold, pairs >= "
            "300, "
            "category split), with H2 marked exploratory",
        },
        "interpretation_scope": [
            "(1) O and S have a correlated distance structure (H1)",
            "(2) the correlation remains after L is controlled linearly (H2; exploratory; "
            "non-linear structure from L may remain)",
            "(3) semantic commonality beyond lexical similarity (not claimed from H2 alone)",
        ],
        "limits_to_state": list(LIMITS),
        "not_in_the_evidence": ["position values", "K values", "item texts"],
    }


def category_descriptives(
    d_o: Array, d_s: Array, categories: Sequence[str], roles: Sequence[str]
) -> dict[str, Any]:
    """Per category (descriptive only; no test, no p value): the Spearman of the O and S distances
    among the items of the category, and the median distances of its near / far / reversed pairs."""
    cats = np.asarray(categories)
    code = {"near_paraphrase": "n", "far_paraphrase": "f", "reversed_meaning": "r"}
    out: dict[str, Any] = {}
    for name in sorted(set(categories)):
        idx = np.flatnonzero(cats == name)
        sub_o, sub_s = d_o[np.ix_(idx, idx)], d_s[np.ix_(idx, idx)]
        kinds = rc.pair_kinds([name] * len(idx), [roles[int(i)] for i in idx])
        v_o, v_s = rc.upper(sub_o), rc.upper(sub_s)
        entry: dict[str, Any] = {
            "n_items": int(idx.size),
            "n_pairs": int(v_o.size),
            "spearman_O_S": rc.spearman(v_o, v_s),
            "median_distance": {},
        }
        for kind in CATEGORY_KINDS:
            mask = kinds == kind
            entry["median_distance"][kind] = {
                "n_pairs": int(mask.sum()),
                "O": float(np.median(v_o[mask])) if mask.any() else None,
                "S": float(np.median(v_s[mask])) if mask.any() else None,
            }
        assert set(code) >= {roles[int(i)] for i in idx}
        out[name] = entry
    return out


def procrustes_eligibility(
    h1: bool, h2: bool, n_point_pairs: int, n_distance_pairs: int
) -> dict[str, Any]:
    """Whether the conditions of DC-RSA-3 for X-5 are met (nothing is executed). The design says
    'pairs >= 300' without saying which pairs: both readings are given."""
    by_points = n_point_pairs >= PROCRUSTES_MIN_PAIRS
    by_distances = n_distance_pairs >= PROCRUSTES_MIN_PAIRS
    return {
        "executed": False,
        "H1_holds": h1,
        "H2_holds": h2,
        "H2_status": "exploratory (calibration not acceptable; power of E unconfirmed)",
        "pairs_threshold": PROCRUSTES_MIN_PAIRS,
        "pairs_reading_point_correspondences": {"n": n_point_pairs, "met": by_points},
        "pairs_reading_distance_pairs": {"n": n_distance_pairs, "met": by_distances},
        "category_split_available": True,
        "conditions_met_if_pairs_are_point_correspondences": bool(h1 and h2 and by_points),
        "conditions_met_if_pairs_are_distance_pairs": bool(h1 and h2 and by_distances),
        "note": "the wording of the design does not fix which pairs are meant; the decision is "
        "left "
        "to the reader of this report",
    }


def run_x4(
    o_points: Array,
    s_points: Array,
    d_l: Array,
    categories: Sequence[str],
    roles: Sequence[str],
    reference_points: Array,
    reference_category: str,
    *,
    n_permutations: int = rc.N_PERMUTATIONS,
    seed: int = rc.SEED,
) -> dict[str, Any]:
    """Every registered number for the real data, as a dict of numbers (no text, no vector)."""
    report = rc.compare_report(
        o_points,
        s_points,
        d_l,
        categories,
        roles,
        reference_points,
        reference_category,
        n_permutations=n_permutations,
        seed=seed,
    )
    raw = report["O"]["raw"]
    judgement = raw["criterion_DC_RSA_2"]
    h1 = bool(judgement["H1_spearman_mantel_p_below_0_01"])
    h2 = bool(judgement["partial_at_least_0_2_and_p_below_0_01"])
    h3 = bool(judgement["knn_above_null_95th_percentile"])
    sensitivity = rc.criterion_dc_rsa_2(report["O"]["centered"]["mantel"])
    report["O"]["centered"]["sensitivity_H_values"] = {
        k: sensitivity[k]
        for k in (
            "H1_spearman_mantel_p_below_0_01",
            "partial_at_least_0_2_and_p_below_0_01",
            "knn_above_null_95th_percentile",
        )
    }
    report["decision_raw_O"] = {
        **judgement["interpretation"],
        "labels": [
            label
            for label, on in (
                ("探索的", h2),
                ("検出力の限界つき", judgement["interpretation"]["power_caveat_applies"]),
            )
            if on
        ],
    }
    report["interpretation_scope"] = interpretation_scope(h1, h2, h3)
    d_o, d_s = rc.pairwise_euclidean(o_points), rc.pairwise_euclidean(s_points)
    report["per_category_descriptive"] = category_descriptives(d_o, d_s, categories, roles)
    report["procrustes_X5"] = procrustes_eligibility(
        h1, h2, o_points.shape[0], o_points.shape[0] * (o_points.shape[0] - 1) // 2
    )
    report["limits"] = list(LIMITS)
    return report
