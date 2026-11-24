"""RSA-X3: the comparison functions, on synthetic data only. No real position, no kernel."""

from __future__ import annotations

import numpy as np
import pytest

from nvs_runtime.hidden_trajectory.rsa import compare as rc

ROLES = ["near_paraphrase"] * 4 + ["far_paraphrase"] * 4 + ["reversed_meaning"] * 4


def structure(n_categories: int = 6) -> tuple[list[str], list[str]]:
    categories = [f"C{c}" for c in range(n_categories) for _ in ROLES]
    return categories, ROLES * n_categories


def points(n: int, dim: int = 64, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---- distance matrices -----------------------------------------------------------------------


def test_a_distance_matrix_is_symmetric_with_a_zero_diagonal() -> None:
    d = rc.pairwise_euclidean(points(12))
    assert np.array_equal(d, d.T) and not np.any(np.diag(d))
    rc.check_distance_matrix(d)
    bad = d.copy()
    bad[0, 1] += 0.1
    with pytest.raises(ValueError):
        rc.check_distance_matrix(bad)


def test_subtracting_the_mean_leaves_distances_and_the_registered_centering_does_not() -> None:
    x = points(20) + 3.0
    assert np.allclose(rc.pairwise_euclidean(x), rc.pairwise_euclidean(x - x.mean(axis=0)))
    c = rc.center_renormalize(x)
    assert np.allclose(np.linalg.norm(c, axis=1), 1.0)
    assert not np.allclose(rc.pairwise_euclidean(x), rc.pairwise_euclidean(c))


def test_word_overlap_distance() -> None:
    d = rc.word_overlap_distance(["The cat, sat.", "the CAT sat!", "dogs run far", "cat dogs"])
    assert d[0, 1] == 0.0 and d[0, 2] == 1.0 and np.array_equal(d, d.T)
    assert abs(d[0, 3] - (1 - 1 / 4)) < 1e-12  # {the,cat,sat} vs {cat,dogs}: 1 shared of 4
    assert rc.word_overlap_distance(["12 34", "56"])[0, 1] == 0.0  # no alphabetic words: both empty


# ---- ranks and correlations ------------------------------------------------------------------


def test_average_ranks_and_spearman_against_a_hand_value() -> None:
    assert rc.rank_average(np.array([10.0, 20.0, 20.0, 5.0])).tolist() == [2.0, 3.5, 3.5, 1.0]
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rc.spearman(x, x**3) == pytest.approx(1.0)
    assert rc.spearman(x, -x) == pytest.approx(-1.0)


def test_the_residual_form_of_the_partial_equals_the_formula() -> None:
    rng = np.random.default_rng(3)
    z = rng.standard_normal(300)
    o = z + 0.6 * rng.standard_normal(300)
    s = z + 0.6 * rng.standard_normal(300)
    assert rc.partial_spearman(o, s, z) == pytest.approx(rc.partial_by_formula(o, s, z), abs=1e-9)


# ---- permutations ----------------------------------------------------------------------------


def test_the_permutation_moves_rows_and_columns_together() -> None:
    d = rc.pairwise_euclidean(points(9, seed=4))
    perm = np.array([3, 1, 8, 0, 5, 2, 7, 4, 6])
    out = rc.permute_matrix(d, perm)
    assert np.array_equal(out, out.T) and not np.any(np.diag(out))
    for i in range(9):
        for j in range(9):
            assert out[i, j] == d[perm[i], perm[j]]
    assert not np.array_equal(out, d[perm])  # not rows alone


def test_permutations_are_reproducible_and_depend_on_the_seed() -> None:
    a, b = rc.permutations(10, 50, seed=5), rc.permutations(10, 50, seed=5)
    assert np.array_equal(a, b) and not np.array_equal(a, rc.permutations(10, 50, seed=6))
    assert all(sorted(p.tolist()) == list(range(10)) for p in a)


def test_the_p_value_formula() -> None:
    null = np.array([0.1, 0.2, 0.3, 0.4])
    assert rc.p_value(0.3, null) == (1 + 2) / (1 + 4)
    assert rc.p_value(9.0, null) == 1 / 5


# ---- the tests of H1-H3 ----------------------------------------------------------------------


def suite(
    d_o: np.ndarray, d_s: np.ndarray, d_l: np.ndarray, n_perm: int = 400, seed: int = 11
) -> dict:
    return rc.mantel_suite(d_o, d_s, d_l, rc.permutations(d_o.shape[0], n_perm, seed))


def test_the_same_matrix_gives_one_everywhere() -> None:
    d = rc.pairwise_euclidean(points(30, seed=2))
    d_l = rc.pairwise_euclidean(points(30, seed=9))
    s = suite(d, d, d_l)
    assert s["spearman"]["observed"] == pytest.approx(1.0)
    assert s["knn_overlap"]["observed"] == 1.0
    assert s["spearman"]["p"] == 1 / (1 + 400)
    assert rc.triple_agreement(d, d)["agreement"] == 1.0
    verdict = rc.criterion_dc_rsa_2(s)
    assert verdict["knn_above_null_95th_percentile"] is True
    assert verdict["H1_spearman_mantel_p_below_0_01"] is True  # p = 1/401


def test_independent_random_matrices_are_inside_the_null() -> None:
    d_o = rc.pairwise_euclidean(points(40, seed=21))
    d_s = rc.pairwise_euclidean(points(40, seed=22))
    d_l = rc.pairwise_euclidean(points(40, seed=23))
    s = suite(d_o, d_s, d_l, n_perm=1000)
    assert abs(s["spearman"]["observed"]) < 0.15
    assert s["spearman"]["p"] > 0.01 and s["partial"]["p"] > 0.01 and s["knn_overlap"]["p"] > 0.01
    assert s["knn_overlap"]["observed"] <= s["knn_overlap"]["null_p95"]
    assert not rc.criterion_dc_rsa_2(s)["met"]


def test_the_partial_removes_what_the_control_explains_and_keeps_a_shared_part() -> None:
    n = 40
    rng = np.random.default_rng(31)
    d_l = rc.pairwise_euclidean(points(n, seed=32))
    # both spaces follow the control and nothing else in common
    only_control = [
        rc.square_from_upper(rc.upper(d_l) + 0.02 * rng.standard_normal(n * (n - 1) // 2), n)
        for _ in range(2)
    ]
    for m in only_control:
        np.fill_diagonal(m, 0.0)
    a = suite(only_control[0], only_control[1], d_l)
    assert a["spearman"]["observed"] > 0.8 and abs(a["partial"]["observed"]) < 0.2
    # both follow one more shared part
    shared = rc.pairwise_euclidean(points(n, seed=33))
    both = [
        rc.square_from_upper(
            rc.upper(d_l) + 2 * rc.upper(shared) + 0.02 * rng.standard_normal(n * (n - 1) // 2), n
        )
        for _ in range(2)
    ]
    b = suite(both[0], both[1], d_l)
    assert b["partial"]["observed"] > 0.5 and b["partial"]["p"] < 0.01


def test_the_same_seed_gives_the_same_p_and_another_seed_a_different_null() -> None:
    d_o, d_s = (rc.pairwise_euclidean(points(25, seed=s)) for s in (41, 42))
    d_l = rc.pairwise_euclidean(points(25, seed=43))
    one, two = suite(d_o, d_s, d_l, seed=7), suite(d_o, d_s, d_l, seed=7)
    assert one == two
    other = suite(d_o, d_s, d_l, seed=8)
    assert other["spearman"]["observed"] == one["spearman"]["observed"]
    assert other["spearman"]["null_mean"] != one["spearman"]["null_mean"]


def test_the_null_of_the_suite_matches_a_plain_recomputation() -> None:
    d_o, d_s = (rc.pairwise_euclidean(points(15, seed=s)) for s in (51, 52))
    d_l = rc.pairwise_euclidean(points(15, seed=53))
    perms = rc.permutations(15, 30, seed=3)
    got = rc.mantel_suite(d_o, d_s, d_l, perms)
    null = np.array(
        [rc.spearman(rc.upper(d_o), rc.upper(rc.permute_matrix(d_s, p))) for p in perms]
    )
    assert got["spearman"]["null_mean"] == pytest.approx(float(null.mean()), abs=1e-12)
    assert got["spearman"]["null_max"] == pytest.approx(float(null.max()), abs=1e-12)


# ---- k-NN, triples, retrieval ----------------------------------------------------------------


def test_knn_ties_go_to_the_lower_index_and_overlap_is_a_share() -> None:
    d = np.array([[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]], dtype=float)
    assert rc.knn_indices(d, 2)[0].tolist() == [1, 2]
    assert rc.knn_overlap(d, d, 2) == 1.0
    flipped = d.copy()
    flipped[0, 1] = flipped[1, 0] = 5.0
    assert rc.knn_overlap(d, flipped, 1) < 1.0


def test_triples_count_and_ties() -> None:
    d = rc.pairwise_euclidean(points(7, seed=61))
    out = rc.triple_agreement(d, d)
    assert out["n_triples"] == 7 * (6 * 5 // 2) and out["n_tied_excluded"] == 0
    ties = np.ones((4, 4)) - np.eye(4)
    assert rc.triple_agreement(ties, ties)["n_tied_excluded"] == 4 * 3 and np.isnan(
        rc.triple_agreement(ties, ties)["agreement"]
    )
    assert rc.triple_agreement(d, 1.0 / (d + np.eye(7)) * (1 - np.eye(7)))["agreement"] < 0.1


def test_retrieval_hit_rate_and_its_chance_level() -> None:
    categories, roles = structure(2)
    x = np.zeros((24, 64))
    for i, c in enumerate(categories):  # a tight cluster per category
        x[i, int(c[1])] = 10.0
        x[i, 2 + (i % 12)] += 0.01 * (i % 12)
    out = rc.near_hit_rate(rc.pairwise_euclidean(x), categories, roles)
    assert out["hit_rate"] == 1.0 and 0.0 < out["expected_for_random_neighbours"] < 1.0
    random_out = rc.near_hit_rate(rc.pairwise_euclidean(points(24, seed=71)), categories, roles)
    assert random_out["hit_rate"] < 1.0


# ---- kinds, summaries, AUC -------------------------------------------------------------------


def test_the_pair_kinds_of_72_items() -> None:
    categories, roles = structure()
    kinds = rc.pair_kinds(categories, roles)
    counts = {k: int(np.sum(kinds == k)) for k in rc.KINDS}
    assert kinds.size == 2556
    assert counts == {"near": 36, "far": 96, "reversed": 96, "unrelated": 2160, "other": 168}


def test_auc_values() -> None:
    assert rc.auc_nearer(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert rc.auc_nearer(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 0.0
    assert rc.auc_nearer(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert np.isnan(rc.auc_nearer(np.array([]), np.array([1.0])))


def test_distribution_summary_and_the_stratification() -> None:
    s = rc.summarize(np.arange(1.0, 101.0))
    assert s["n"] == 100 and s["min"] == 1.0 and s["max"] == 100.0 and s["median"] == 50.5
    d_o = rc.pairwise_euclidean(points(30, seed=81))
    rows = rc.stratified_spearman(d_o, d_o, rc.pairwise_euclidean(points(30, seed=82)))
    assert sum(r["n_pairs"] for r in rows) == 30 * 29 // 2
    assert all(r["spearman"] == pytest.approx(1.0) for r in rows)


# ---- 7.2: the radius -------------------------------------------------------------------------


def test_the_radius_boundary_is_outside() -> None:
    ref = np.zeros((1, 64))
    items = np.zeros((4, 64))
    items[0, 0], items[1, 0], items[2, 0], items[3, 0] = 0.649999, 0.65, 0.650001, 0.0
    categories = ["C1"] * 4
    roles = ["near_paraphrase", "far_paraphrase", "reversed_meaning", "near_paraphrase"]
    out = rc.reference_distances(ref, items, categories, roles, "C1")
    inside = out["per_reference"][0]
    assert inside["near"]["inside_share"] == 1.0  # 0.649999 and 0.0
    assert inside["far"]["inside_share"] == 0.0  # exactly 0.65
    assert inside["reversed"]["inside_share"] == 0.0
    assert inside["other_category"] == {"n": 0, "inside_share": None}


def test_references_are_averaged_and_their_spread_is_reported() -> None:
    items = np.zeros((2, 64))
    items[1, 0] = 1.0
    refs = np.zeros((3, 64))
    refs[1, 0] = 1.0  # the reference at the second item
    refs[2, 0] = 5.0
    out = rc.reference_distances(
        refs, items, ["C1", "C1"], ["near_paraphrase", "near_paraphrase"], "C1"
    )
    shares = [r["near"]["inside_share"] for r in out["per_reference"]]
    assert shares == [0.5, 0.5, 0.0]
    assert out["mean_inside_share_over_references"]["near"] == pytest.approx(1.0 / 3.0)
    assert out["inside_share_spread_across_references"]["near"]["max"] == 0.5


# ---- the report and the registration ---------------------------------------------------------


def test_the_report_has_numbers_only_and_the_registered_hash() -> None:
    categories, roles = structure(3)
    n = len(categories)
    o, s = points(n, seed=91), points(n, seed=92)
    d_l = rc.pairwise_euclidean(points(n, seed=93))
    refs = points(3, seed=94)
    report = rc.compare_report(o, s, d_l, categories, roles, refs, "C0", n_permutations=60)
    assert report["spec_sha256"] == rc.spec_sha256() and report["n_items"] == n
    assert set(report["O"]) == {"raw", "centered"}
    for version in report["O"].values():
        assert set(version) >= {"mantel", "triples", "scale_7_1", "retrieval_hit_at_3", "role"}
        assert len(version["stratified_spearman_by_word_overlap_quartile"]) == 4
        assert version["triples"]["descriptive_only"] is True
        assert "p" not in version["triples"] and "null_p95" not in version["triples"]
    # the criterion is judged for the raw O only; the centered O is a sensitivity analysis
    assert "criterion_DC_RSA_2" in report["O"]["raw"] and report["O"]["raw"]["role"] == "primary"
    assert "criterion_DC_RSA_2" not in report["O"]["centered"]
    assert report["O"]["centered"]["role"] == "sensitivity"

    def strings(node: object) -> list[str]:
        if isinstance(node, dict):
            return [v for x in node.values() for v in strings(x)]
        if isinstance(node, (list, tuple)):
            return [v for x in node for v in strings(x)]
        return [node] if isinstance(node, str) else []

    # no position and no text: the report holds no array of the size of a position
    assert all(len(x) < 400 for x in strings(report))
    assert "scale_7_2" in report


def test_the_registered_spec_is_fixed() -> None:
    spec = rc.analysis_spec()
    assert spec["test"]["seed"] == 20261117 and spec["test"]["n_permutations"] == 10000
    assert spec["metrics"]["c_knn"]["k"] == 5 and spec["scale_7_2"]["radius"] == 0.65
    assert spec["spec_version"] == "rsa-analysis-spec/2"
    assert spec["supersedes"]["spec_version"] == "rsa-analysis-spec/1"
    assert spec["supersedes"]["spec_file_sha256"].startswith("281083e5")
    assert spec["criterion_DC_RSA_2"]["judged_for"] == "raw O only"
    assert spec["metrics"]["d_triples_status"].startswith("descriptive_only")
    # what the replacement did not touch
    assert spec["test"]["n_permutations"] == 10000 and spec["metrics"]["e_retrieval"]["k"] == 3
    # a change of the registered method changes this hash on purpose
    assert rc.spec_sha256() == EXPECTED_SPEC_SHA256


EXPECTED_SPEC_SHA256 = "b39bce33282e2d322fa2272a56a2c13ff9a85f4abf59974713a8bc40c36aa6bd"


def test_the_eight_outcomes_of_the_decision_table() -> None:
    assert len(rc.DECISION_TABLE) == 8
    assert {(a, b, c) for a, b, c, _ in rc.DECISION_TABLE} == {
        (a, b, c) for a in (False, True) for b in (False, True) for c in (False, True)
    }
    readings = {(a, b, c): text for a, b, c, text in rc.DECISION_TABLE}
    assert readings[(True, True, True)] == "関係性と近傍の保存を支持（成立）"
    assert readings[(False, False, False)] == "関係性の保存を確認できない"
    assert readings[(False, True, False)].startswith("不整合（統制後だけ正）")
    assert readings[(False, True, True)].startswith("同上（不整合）")
    for a, b, c, _ in rc.DECISION_TABLE:
        d = rc.decide(a, b, c)
        assert (
            d["exploratory_until_calibration_acceptable"] is b
        )  # every row with H2 is exploratory
        assert d["includes_H2"] is b
    assert rc.decide(True, False, True)["power_caveat_applies"] is True  # a "not confirmed" row
    assert rc.decide(True, True, True)["power_caveat_applies"] is False
    spec = rc.analysis_spec()
    assert [(r["H1"], r["H2"], r["H3"]) for r in spec["decision_table_DC_RSA_8"]] == [
        (a, b, c) for a, b, c, _ in rc.DECISION_TABLE
    ]
    assert spec["supersedes"]["spec_file_sha256"].startswith("281083e5")
    assert "395828f6" in spec["design"]


def test_the_criterion_needs_h1_h2_and_h3() -> None:
    def suite(sp: float, partial: float, pp: float, knn: float, p95: float) -> dict:
        return {
            "spearman": {"p": sp},
            "partial": {"observed": partial, "p": pp},
            "knn_overlap": {"observed": knn, "null_p95": p95},
        }

    assert rc.criterion_dc_rsa_2(suite(0.001, 0.3, 0.001, 0.5, 0.2))["met"] is True
    assert rc.criterion_dc_rsa_2(suite(0.02, 0.3, 0.001, 0.5, 0.2))["met"] is False  # H1 fails
    assert rc.criterion_dc_rsa_2(suite(0.001, 0.19, 0.001, 0.5, 0.2))["met"] is False  # H2: < 0.2
    assert rc.criterion_dc_rsa_2(suite(0.001, 0.3, 0.01, 0.5, 0.2))["met"] is False  # p not < 0.01
    assert rc.criterion_dc_rsa_2(suite(0.001, 0.3, 0.001, 0.2, 0.2))["met"] is False  # not above
    odd = rc.criterion_dc_rsa_2(suite(0.5, 0.3, 0.001, 0.1, 0.2))
    assert odd["interpretation"]["reading"].startswith("不整合")


def test_permutations_accept_a_seed_sequence() -> None:
    a = rc.permutations(8, 5, np.random.SeedSequence([1, 2, 3]))
    b = rc.permutations(8, 5, np.random.SeedSequence([1, 2, 3]))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, rc.permutations(8, 5, np.random.SeedSequence([1, 2, 4])))
    assert np.array_equal(rc.permutations(8, 5, 77), rc.permutations(8, 5, 77))
