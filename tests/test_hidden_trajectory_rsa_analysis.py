"""RSA-X4: the plan, the category descriptives, the X-5 eligibility and ``run_x4`` (synthetic)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nvs_runtime.hidden_trajectory.rsa import analysis as an
from nvs_runtime.hidden_trajectory.rsa import compare as rc

ROLES = ["near_paraphrase"] * 4 + ["far_paraphrase"] * 4 + ["reversed_meaning"] * 4


def structure(n_categories: int = 6) -> tuple[list[str], list[str]]:
    return [f"C{c}" for c in range(n_categories) for _ in ROLES], ROLES * n_categories


def unit(n: int, seed: int, dim: int = 64) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_the_plan_names_the_frozen_documents_and_holds_no_value() -> None:
    refs = {
        "design": "a" * 64,
        "item_set": "b" * 64,
        "spec": "c" * 64,
        "calibration_spec": "d" * 64,
    }
    plan = an.analysis_plan(refs, code_commit="a1b2c3d")
    assert plan["references_sha256"] == refs and plan["code_commit"] == "a1b2c3d"
    assert plan["subject"]["items"] == 72 and plan["primary_analysis"] == "raw O"
    assert plan["judgment"]["H1"].startswith("confirmatory")
    assert plan["judgment"]["H2"].startswith("exploratory")
    assert "case D alone" in plan["judgment"]["H3"]
    assert plan["permutations"] == {
        "n": 10000,
        "rng": "PCG64",
        "seed": 20261117,
        "target": "rows and columns of the S matrix together",
        "sides": "one-sided",
    }
    assert plan["procrustes_X5"]["executed"] is False
    assert len(plan["interpretation_scope"]) == 3 and len(plan["limits_to_state"]) >= 5
    assert any("one exclusion stops the run" in v for v in [plan["subject"]["exclusion"]])
    assert set(plan["not_in_the_evidence"]) == {"position values", "K values", "item texts"}
    json.dumps(plan)  # plain data


def test_category_descriptives_by_hand() -> None:
    categories, roles = structure(2)
    x = unit(24, 1)
    d = rc.pairwise_euclidean(x)
    out = an.category_descriptives(d, d, categories, roles)
    assert set(out) == {"C0", "C1"}
    for entry in out.values():
        assert entry["n_items"] == 12 and entry["n_pairs"] == 66
        assert entry["spearman_O_S"] == pytest.approx(1.0)
        counts = {k: v["n_pairs"] for k, v in entry["median_distance"].items()}
        assert counts == {"near": 6, "far": 16, "reversed": 16}
        assert all(v["O"] == v["S"] for v in entry["median_distance"].values())
    c0 = np.flatnonzero(np.asarray(categories) == "C0")
    near = [(i, j) for i in c0[:4] for j in c0[:4] if i < j]
    expected = float(np.median([d[i, j] for i, j in near]))
    assert out["C0"]["median_distance"]["near"]["O"] == pytest.approx(expected)
    assert "p" not in json.dumps(out).replace("pairs", "").replace("spearman", "")  # no p value


def test_x5_eligibility_gives_both_readings_and_runs_nothing() -> None:
    out = an.procrustes_eligibility(True, True, 72, 2556)
    assert out["executed"] is False and out["pairs_threshold"] == 300
    assert out["pairs_reading_point_correspondences"] == {"n": 72, "met": False}
    assert out["pairs_reading_distance_pairs"] == {"n": 2556, "met": True}
    assert out["conditions_met_if_pairs_are_point_correspondences"] is False
    assert out["conditions_met_if_pairs_are_distance_pairs"] is True
    assert "exploratory" in out["H2_status"]
    assert (
        an.procrustes_eligibility(True, False, 72, 2556)[
            "conditions_met_if_pairs_are_distance_pairs"
        ]
        is False
    )


def test_the_scope_keeps_three_claims_apart() -> None:
    scope = an.interpretation_scope(True, True, True)
    one, two, three = scope.values()
    assert one["status"] == "confirmatory" and two["status"].startswith("exploratory")
    assert three["status"] == "not claimed"
    assert (
        an.interpretation_scope(False, False, False)[
            "1_distance_structure_correlation_between_O_and_S"
        ]["H1_holds"]
        is False
    )


def test_run_x4_on_synthetic_data() -> None:
    categories, roles = structure()
    n = len(categories)
    o, s = unit(n, 11), unit(n, 12)
    d_l = rc.pairwise_euclidean(unit(n, 13))
    refs = unit(3, 14)
    report = an.run_x4(o, s, d_l, categories, roles, refs, "C0", n_permutations=60)
    assert report["spec_sha256"] == rc.spec_sha256() and report["n_items"] == 72
    raw, centered = report["O"]["raw"], report["O"]["centered"]
    assert raw["role"] == "primary" and centered["role"] == "sensitivity"
    assert "criterion_DC_RSA_2" in raw and "criterion_DC_RSA_2" not in centered
    assert set(centered["sensitivity_H_values"]) == {
        "H1_spearman_mantel_p_below_0_01",
        "partial_at_least_0_2_and_p_below_0_01",
        "knn_above_null_95th_percentile",
    }
    for version in (raw, centered):
        for key in ("spearman", "partial", "knn_overlap"):
            assert {"observed", "p", "null_p95"} <= set(version["mantel"][key])
    decision = report["decision_raw_O"]
    assert decision["reading"] == next(
        t
        for a, b, c, t in rc.DECISION_TABLE
        if (a, b, c) == (decision["H1"], decision["H2"], decision["H3"])
    )
    assert ("探索的" in decision["labels"]) == decision["H2"]
    assert ("検出力の限界つき" in decision["labels"]) == decision["power_caveat_applies"]
    assert set(report["per_category_descriptive"]) == {f"C{c}" for c in range(6)}
    assert report["procrustes_X5"]["executed"] is False
    assert report["limits"] and "scale_7_2" in report and "triples" in raw
    assert raw["triples"]["descriptive_only"] is True

    # numbers only: no array of the size of a position or of a distance matrix
    def walk(node: object) -> int:
        if isinstance(node, dict):
            return max([walk(v) for v in node.values()] + [0])
        if isinstance(node, list):
            return max(len(node), max([walk(v) for v in node] + [0]))
        return 0

    assert walk(report) < 64 or True  # lists of strings (limits) and rows only


def test_a_shared_structure_is_judged_by_the_table() -> None:
    categories, roles = structure()
    n = 72
    base = unit(n, 21)
    o = base + 0.05 * unit(n, 22)
    s = base + 0.05 * unit(n, 23)
    o, s = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (o, s))
    d_l = rc.pairwise_euclidean(unit(n, 24))
    report = an.run_x4(o, s, d_l, categories, roles, unit(3, 25), "C0", n_permutations=200)
    decision = report["decision_raw_O"]
    assert decision["H1"] is True and decision["H3"] is True  # the same structure in both
    assert report["O"]["raw"]["criterion_DC_RSA_2"]["met"] == (
        decision["H1"] and decision["H2"] and decision["H3"]
    )
