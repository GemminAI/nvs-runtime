"""RSA calibration (design v1.0, 7B): the functions, on a small synthetic L. The real run is opt-in
(``test_hidden_trajectory_rsa_calibration_real.py``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from nvs_runtime.hidden_trajectory.rsa import calibration as cal
from nvs_runtime.hidden_trajectory.rsa import compare as rc

N = 36


def world() -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(5)
    return rc.pairwise_euclidean(rng.standard_normal((N, 8))), [f"C{i // 6}" for i in range(N)]


def test_mds_basis_of_a_euclidean_matrix() -> None:
    d_l, _ = world()
    basis = cal.mds_basis(d_l)
    assert basis.p_full == 8 and basis.n_negative == 0 and basis.negative_abs_sum == 0.0
    assert list(basis.eigenvalues) == sorted(basis.eigenvalues, reverse=True)
    z = cal.coordinates(basis, 8, 8)
    assert rc.spearman(rc.upper(rc.pairwise_euclidean(z)), rc.upper(d_l)) == pytest.approx(1.0)
    # the scale is a single constant: the distances are proportional to L
    ratio = rc.upper(rc.pairwise_euclidean(z)) / rc.upper(d_l)
    assert np.allclose(ratio, ratio[0])


def test_a_non_euclidean_matrix_has_negative_eigenvalues() -> None:
    d = np.ones((6, 6)) - np.eye(6)
    d[0, 1] = d[1, 0] = 5.0  # breaks the triangle inequality
    basis = cal.mds_basis(d)
    assert basis.n_negative >= 1 and basis.negative_abs_sum > 0.0
    assert 0.0 < basis.negative_abs_sum / basis.abs_sum < 1.0


def test_coordinates_scale_and_position() -> None:
    d_l, _ = world()
    basis = cal.mds_basis(d_l)
    z = cal.coordinates(basis, 5, 20)
    assert z.shape == (N, 20) and not np.any(z[:, 5:])
    assert float(np.mean(np.linalg.norm(z, axis=1))) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cal.coordinates(basis, 9, 20)  # only 8 positive eigenvalues
    with pytest.raises(ValueError):
        cal.coordinates(basis, 5, 4)


def test_the_setup_of_every_case() -> None:
    d_l, _ = world()
    basis = cal.mds_basis(d_l)
    a, a64, full, d = (cal.case_setup(c, basis) for c in ("A", "A_64", "A_full", "D"))
    assert (a.m, a.dim) == (8, 64)  # min(10, p_full) with p_full = 8
    assert (a64.m, a64.dim) == (8, 64) and (full.m, full.dim) == (8, 8)
    assert d.z is None and d.dim == 64
    for c in ("B", "C", "E"):
        assert (cal.case_setup(c, basis).m, cal.case_setup(c, basis).dim) == (8, 64)
    with pytest.raises(ValueError):
        cal.case_setup("Z", basis)
    info = cal.basis_diagnostics(basis, a, d_l)
    assert info["p_full"] == 8 and info["n_negative_eigenvalues"] == 0
    assert info["adopted_share_of_positive_sum"] == pytest.approx(1.0)
    assert info["spearman_Z_distance_vs_L"] == pytest.approx(1.0)


def test_the_streams_are_registered_and_reproducible() -> None:
    first = cal.rng_for("A", 3, 0).standard_normal(4)
    assert np.array_equal(first, cal.rng_for("A", 3, 0).standard_normal(4))
    others = [
        cal.rng_for("A", 3, 1),
        cal.rng_for("A", 4, 0),
        cal.rng_for("B", 3, 0),
        cal.calibration_rng("A", 3),
    ]
    assert all(not np.array_equal(first, o.standard_normal(4)) for o in others)
    expected = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([20261201, 4, 999999, 7]))
    ).standard_normal(3)
    assert np.array_equal(cal.calibration_rng("E", 7).standard_normal(3), expected)
    assert cal.CASE_NUMBER == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "A_full": 5, "A_64": 6}


def test_a_pair_is_unit_norm_independent_and_follows_the_basis() -> None:
    d_l, categories = world()
    basis = cal.mds_basis(d_l)
    setup = cal.case_setup("A", basis)
    o, s = cal.make_pair(setup, 0.3, 0.0, categories, cal.rng_for("A", 0, 0))
    assert np.allclose(np.linalg.norm(o, axis=1), 1.0) and np.allclose(
        np.linalg.norm(s, axis=1), 1.0
    )
    assert not np.allclose(o, s)
    o2, _ = cal.make_pair(setup, 0.3, 0.0, categories, cal.rng_for("A", 0, 0))
    assert np.array_equal(o, o2)
    quiet_o, quiet_s = cal.make_pair(setup, 1e-9, 0.0, categories, cal.rng_for("A", 0, 0))
    assert np.allclose(quiet_o, quiet_s, atol=1e-6)  # no noise: both are the basis
    assert np.allclose(quiet_o, cal.l2_rows(setup.z), atol=1e-6)  # type: ignore[arg-type]


def test_the_same_seed_gives_the_same_draws_for_every_sigma() -> None:
    d_l, categories = world()
    setup = cal.case_setup("A", cal.mds_basis(d_l))
    # the noise is one N(0,1) draw per seed, multiplied by sigma
    rng_a, rng_b = cal.calibration_rng("A", 2), cal.calibration_rng("A", 2)
    n_a = rng_a.standard_normal((N, 64))
    n_b = rng_b.standard_normal((N, 64))
    assert np.array_equal(n_a, n_b)
    o1, _ = cal.make_pair(setup, 0.5, 0.0, categories, cal.calibration_rng("A", 2))
    o2, _ = cal.make_pair(setup, 0.5, 0.0, categories, cal.calibration_rng("A", 2))
    assert np.array_equal(o1, o2)


def test_case_e_shares_its_centroids_and_d_is_independent() -> None:
    d_l, categories = world()
    basis = cal.mds_basis(d_l)
    e = cal.case_setup("E", basis)
    o, s = cal.make_pair(e, 1e-9, 3.0, categories, cal.rng_for("E", 0, 0))
    assert np.allclose(o, s, atol=1e-6)  # the same centroids, no noise
    within = np.mean([np.linalg.norm(o[i] - o[j]) for i in range(6) for j in range(i + 1, 6)])
    across = np.mean([np.linalg.norm(o[i] - o[j]) for i in range(6) for j in range(30, 36)])
    assert within < across  # the blocks are by category
    do, ds = cal.make_pair(cal.case_setup("D", basis), 0.0, 0.0, categories, cal.rng_for("D", 0, 0))
    assert do.shape == (N, 64) and abs(float(np.mean(do * ds))) < 0.05
    assert np.allclose(np.linalg.norm(do, axis=1), 1.0)


def test_bisection_and_the_monotonicity_check() -> None:
    out = cal.bisect_search(lambda x: 1.0 / x, 0.01, 5.0, 0.5, increasing=False, geometric=True)
    assert out["evaluations"][0][0] == pytest.approx(math.sqrt(0.01 * 5.0))
    assert out["result"] == pytest.approx(2.0, rel=2e-3) and out["iterations"] <= 40
    lin = cal.bisect_search(lambda t: t / 5.0, 0.0, 5.0, 0.3, increasing=True, geometric=False)
    assert lin["evaluations"][0][0] == 2.5 and lin["result"] == pytest.approx(1.5, abs=2e-3)
    assert cal.monotone([(1, 0.9), (2, 0.5), (3, 0.1)], increasing=False)[
        "monotone_within_tolerance"
    ]
    small = cal.monotone([(1, 0.5), (2, 0.505), (3, 0.1)], increasing=False)
    assert small["monotone_within_tolerance"] and small["largest_inversion"] == pytest.approx(0.005)
    broken = cal.monotone([(1, 0.5), (2, 0.52), (3, 0.1)], increasing=False)
    assert not broken["monotone_within_tolerance"]
    assert not cal.monotone([(1, 0.5), (2, 0.48)], increasing=True)["monotone_within_tolerance"]


def test_sigma_calibration_succeeds_and_is_unmet_when_the_target_is_out_of_reach() -> None:
    d_l, categories = world()
    setup = cal.case_setup("A", cal.mds_basis(d_l))
    ok = cal.calibrate_sigma(setup, categories, d_l, 0.5, n_cal=12)
    assert ok["status"] == "calibrated" and not ok["unmet_reasons"]
    assert abs(ok["m_O"] - 0.5) <= 0.03 and abs(ok["m_S"] - 0.5) <= 0.03
    assert ok["check_i_brackets_target"] and ok["check_ii_monotone"]["monotone_within_tolerance"]
    sigmas = [row["sigma"] for row in ok["evaluated"]]
    assert sigmas == sorted(sigmas) and 0.01 in sigmas and 5.0 in sigmas
    assert ok["iterations"] <= 40
    out_of_reach = cal.calibrate_sigma(setup, categories, d_l, 0.9999, n_cal=12)
    assert out_of_reach["status"] == "unmet"
    assert any(r.startswith("i:") for r in out_of_reach["unmet_reasons"])
    assert out_of_reach["highest_metric_reached"] < 0.9999  # reported, the tolerance is not widened


def test_tau_calibration_flags() -> None:
    d_l, categories = world()
    basis = cal.mds_basis(d_l)
    sigma = cal.calibrate_sigma(cal.case_setup("A", basis), categories, d_l, 0.5, n_cal=12)["sigma"]
    out = cal.calibrate_tau(cal.case_setup("E", basis), categories, d_l, sigma, n_cal=12)
    assert out["status"] in ("calibrated", "unmet")
    assert (out["status"] == "calibrated") == (not out["unmet_reasons"])
    if out["status"] == "calibrated":
        assert abs(out["median_partial_spearman"] - 0.30) <= 0.03
    taus = [row["tau"] for row in out["evaluated"]]
    assert taus == sorted(taus) and 0.0 in taus and 5.0 in taus
    impossible = cal.calibrate_tau(
        cal.case_setup("E", basis), categories, d_l, sigma, n_cal=12, target=0.999
    )
    assert impossible["status"] == "unmet"


def test_clopper_pearson() -> None:
    low, high = cal.clopper_pearson(0, 500)
    assert low == 0.0 and high == pytest.approx(1 - 0.025 ** (1 / 500), rel=1e-6)
    low, high = cal.clopper_pearson(500, 500)
    assert high == 1.0 and low == pytest.approx(0.025 ** (1 / 500), rel=1e-6)
    low, high = cal.clopper_pearson(5, 500)
    assert 0.003 < low < 0.0035 and 0.0228 < high < 0.0235
    # the defining property: the tails at the limits are 0.025
    n, k = 200, 14
    low, high = cal.clopper_pearson(k, n)

    def pmf(x: int, p: float) -> float:
        return math.comb(n, x) * p**x * (1 - p) ** (n - x)

    assert sum(pmf(x, low) for x in range(k, n + 1)) == pytest.approx(0.025, abs=1e-6)
    assert sum(pmf(x, high) for x in range(0, k + 1)) == pytest.approx(0.025, abs=1e-6)
    with pytest.raises(ValueError):
        cal.clopper_pearson(3, 0)


def record(
    p: float, partial: float, spearman_p: float = 0.5, knn: bool = False, i: int = 0
) -> dict:
    return {
        "i": i,
        "rho_O_L": 0.5,
        "rho_S_L": 0.5,
        "spearman_O_S": 0.3,
        "spearman_p": spearman_p,
        "partial_O_S": partial,
        "partial_p": p,
        "knn_observed": 0.4 if knn else 0.1,
        "knn_null_p95": 0.2,
    }


def test_the_summary_counts_p_and_h_separately() -> None:
    records = [
        record(0.004, 0.25),  # P and H
        record(0.004, 0.10),  # P only
        record(0.03, 0.30),  # alpha 0.05 only
        record(0.5, 0.30),  # none (partial >= 0.2 alone is not H)
    ]
    s = cal.summarize_case(records)
    r = s["rejection"]
    assert r["P_partial_mantel"]["alpha_0.01"]["rejections"] == 2
    assert r["H_partial_ge_0_2_and_p"]["alpha_0.01"]["rejections"] == 1
    assert r["P_partial_mantel"]["alpha_0.05"]["rejections"] == 3
    assert r["H_partial_ge_0_2_and_p"]["alpha_0.05"]["rejections"] == 2
    assert s["partial_ge_0_2_share"] == 0.75 and s["n_simulations"] == 4
    assert (
        r["P_partial_mantel"]["alpha_0.01"]["ci95"][0]
        < 0.5
        < r["P_partial_mantel"]["alpha_0.01"]["ci95"][1]
    )
    assert r["knn_overlap_above_null_p95"]["rejections"] == 0
    assert set(s["realised"]["partial_O_S"]) == {"mean", "median", "p2_5", "p97_5", "min", "max"}


def test_the_boundary_rule() -> None:
    d = cal.boundary_decision
    assert d(0.019, 0.019)["action"] == "acceptable" and d(0.02, 0.02)["action"] == "acceptable"
    assert d(0.021, 0.01)["action"] == "extend_to_2000"
    assert (
        d(0.01, 0.021)["action"] == "extend_to_2000" and d(0.03, 0.03)["action"] == "extend_to_2000"
    )
    over = d(0.031, 0.01)
    assert over["action"] == "not_acceptable_no_extension" and over["not_acceptable_for"] == ["P"]
    both = d(0.05, 0.04)
    assert both["not_acceptable_for"] == ["P", "H"]
    # one over 0.03 and the other inside (0.02, 0.03]: no extension, the first is not acceptable
    mixed = d(0.025, 0.031)
    assert mixed["action"] == "not_acceptable_no_extension" and mixed["not_acceptable_for"] == ["H"]
    assert cal.final_decision(0.02, 0.02)["action"] == "acceptable"
    final = cal.final_decision(0.02, 0.0201)
    assert final["action"] == "not_acceptable" and final["not_acceptable_for"] == ["H"]


def test_the_power_rule() -> None:
    assert cal.power_decision(0.80)["detects_this_scale"] is True
    low = cal.power_decision(0.7999)
    assert low["detects_this_scale"] is False and "検出する力が足りない" in low["reading"]


def test_a_simulation_is_reproducible_and_a_rerun_of_the_same_i_is_identical() -> None:
    d_l, categories = world()
    setup = cal.case_setup("A", cal.mds_basis(d_l))
    one = cal.simulate_one(setup, 0.5, 0.0, d_l, categories, 7, n_permutations=60)
    again = cal.simulate_one(setup, 0.5, 0.0, d_l, categories, 7, n_permutations=60)
    assert one == again and one["i"] == 7
    assert cal.simulate_one(setup, 0.5, 0.0, d_l, categories, 8, n_permutations=60) != one
    assert set(one) >= {
        "rho_O_L",
        "rho_S_L",
        "partial_O_S",
        "partial_p",
        "spearman_p",
        "knn_observed",
    }


def test_the_spec_is_registered_before_the_run() -> None:
    spec = cal.calibration_spec()
    assert spec["run"]["n_simulations"] == 500 and spec["run"]["n_permutations"] == 1000
    assert spec["run"]["n_permutations_reduced_from_10000"] is True
    assert spec["random_numbers"]["case_number"]["A_full"] == 5
    assert spec["calibration_of_sigma"]["targets"] == {
        "A": 0.5,
        "A_64": 0.5,
        "A_full": 0.5,
        "B": 0.7,
        "C": 0.2,
    }
    assert "395828f6709bad0c85aa900a8d100e87a6e55cc56f8c5cbdd450d3bfa4bdcb07" in spec["design"]
    assert spec["false_positive_criterion"]["accept"].endswith("0.02")
    assert "not a proof of cause" in spec["A_A_64_A_full_comparison"]
