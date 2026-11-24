"""RSA calibration of the permutation test of the partial Mantel (design v1.0, 7B; DC-RSA-7).

The registered test (``compare.mantel_suite``: the S residual matrix of the partial Spearman is
permuted by item, rows and columns together) is run on synthetic O and S to see how often it
rejects. No real position is used. ``L`` is the word-overlap distance of the fixed item set.

Generation (fixed here, as registered): the classical MDS of ``L`` gives a basis ``Z`` (the largest
positive eigenvalues, coordinates ``V sqrt(Lambda)``, scaled so that the mean row norm is 1, placed
in the first ``m`` of ``dim`` coordinates, the rest 0). O and S each get independent
``N(0, sigma^2)`` noise in every one of the ``dim`` coordinates and every row is scaled to unit L2
norm.

Cases: A (m = min(10, p_full), dim 64; conditional independence given the shared L structure, not
a pure null), A_64 (m = min(64, p_full), dim 64; diagnostic), A_full (m = p_full, dim p_full;
diagnostic), B (A with a strong dependence on L, Spearman about 0.7), C (a weak one, about 0.2),
D (two independent 64-D Gaussian configurations), E (A plus a shared category centroid, strength
``tau``, added before the noise).

Choices the design leaves to the implementation (registered in ``calibration_spec``):

* a positive eigenvalue is one above 1e-9 (the eigenvalues of the fixed set show a gap from 1e-16
  to 4e-4, so any threshold between them gives the same count);
* the searched value is the geometric (sigma) or arithmetic (tau) midpoint of the last interval,
  and it is evaluated once more (its row in the table); the endpoints of the range are evaluated
  too and all points enter the monotonicity check;
* inside one simulation the draws come from one generator in this order: E's six centroids, the
  noise of O, the noise of S (D: the configuration of O, then that of S); the calibration draws of
  one ``j`` are used for every candidate (the noise is drawn as N(0, 1) and multiplied by the
  candidate).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from nvs_runtime.hidden_trajectory.rsa import compare as rc

Array = NDArray[np.float64]

CASES: Final = ("A", "B", "C", "D", "E", "A_full", "A_64")
CASE_NUMBER: Final = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "A_full": 5, "A_64": 6}
MASTER_SEED: Final = 20261201
CALIBRATION_INDEX: Final = 999999
N_CALIBRATION: Final = 50
SIGMA_RANGE: Final = (0.01, 5.0)
TAU_RANGE: Final = (0.0, 5.0)
MAX_ITERATIONS: Final = 40
WIDTH_TOLERANCE: Final = 1e-3
TARGET_TOLERANCE: Final = 0.03
INVERSION_TOLERANCE: Final = 0.01
SIGMA_TARGET: Final = {"A": 0.5, "A_64": 0.5, "A_full": 0.5, "B": 0.7, "C": 0.2}
TAU_TARGET: Final = 0.30
MDS_DIM_A: Final = 10
EMBED_DIM: Final = 64
POSITIVE_EIGENVALUE_TOLERANCE: Final = 1e-9
N_SIMULATIONS: Final = 500
N_SIMULATIONS_EXTENDED: Final = 2000
N_PERMUTATIONS: Final = 1000
ACCEPT_RATE: Final = 0.02
EXTEND_UP_TO_RATE: Final = 0.03
POWER_MIN: Final = 0.80
ALPHAS: Final = (0.05, 0.01)
DESIGN_SHA256: Final = "395828f6709bad0c85aa900a8d100e87a6e55cc56f8c5cbdd450d3bfa4bdcb07"
COMPARISON_NOTE: Final = (
    "The difference between A and A_64 is the effect of truncating to the top 10 dimensions inside "
    "64 dimensions; the difference between A_64 and A_full is the effect of the eigenvalues beyond "
    "64 and of the noise dimension (p_full instead of 64). Sigma is recalibrated per case and the "
    "distribution of distances after L2 normalisation changes too, so the differences are "
    "observations, not a proof of cause. If A is above the criterion and A_64 or A_full is not, "
    "'truncation or the number of dimensions may be involved' is recorded; the judgment of A stays "
    "'not acceptable' and the criterion is not relaxed."
)


# ---- the registration ------------------------------------------------------------------------


def calibration_spec() -> dict[str, Any]:
    return {
        "calibration_version": "rsa-calibration-spec/1",
        "design": f"design v1.0 (sha256 {DESIGN_SHA256}), 7B (DC-RSA-7)",
        "tested": "the registered partial Mantel (S residual matrix permuted by item, rows and "
        "columns together), the Spearman Mantel and the k-NN overlap, on synthetic data",
        "L": "1 - Jaccard of lower-case [a-z]+ word sets of the fixed 72 sentences",
        "basis_Z": {
            "method": "classical MDS: eigen-decomposition of -1/2 J L^2 J; largest positive "
            "eigenvalues; coordinates V sqrt(Lambda); the whole scaled so the mean row norm is 1; "
            "placed in the first m coordinates, the rest 0",
            "positive_eigenvalue": f"> {POSITIVE_EIGENVALUE_TOLERANCE}",
        },
        "noise": "independent N(0, sigma^2) in every one of dim coordinates for O and for S, then "
        "every row scaled to unit L2 norm; Euclidean distance",
        "cases": {
            "A": f"Z with m = min({MDS_DIM_A}, p_full), dim {EMBED_DIM}; "
            "sigma_A so that the median "
            "Spearman(O, L) and Spearman(S, L) are about 0.5. Not a pure null: structure from L "
            "survives the OLS residualisation, so the partial correlation may not be 0",
            "A_64": f"diagnostic, not used for the judgment: m = min({EMBED_DIM}, p_full), dim "
            f"{EMBED_DIM}, own sigma (target 0.5)",
            "A_full": "diagnostic, not used for the judgment: m = p_full, dim p_full (coordinates "
            "and noise), own sigma (target 0.5)",
            "B": "as A, sigma_B for about 0.7",
            "C": "as A, sigma_C for about 0.2",
            "D": f"two independent {EMBED_DIM}-D standard normal configurations (72 points), each "
            "row scaled to unit L2 norm; no L",
            "E": "as A with sigma_A; before the noise, six category centroids N(0, I) in 64-D (the "
            "same for O and S) times tau are added to Z by category; tau so that the median "
            "partial Spearman (O, S | L) of the calibration draws is 0.30",
        },
        "calibration_of_sigma": {
            "targets": SIGMA_TARGET,
            "target_meaning": "tuning targets of the generator, not an acceptance condition of a "
            "simulation; the realised values are recorded per simulation",
            "data": f"{N_CALIBRATION} draws; the same seeds for every candidate sigma; "
            "median m_O of "
            "Spearman(O, L) and median m_S of Spearman(S, L), both reported",
            "metric": "(m_O + m_S) / 2 (decreasing in sigma)",
            "search": f"bisection in log sigma over {list(SIGMA_RANGE)}, first point the geometric "
            f"mean, stop when the interval is narrower than {WIDTH_TOLERANCE} in log sigma "
            "or after "
            f"{MAX_ITERATIONS} iterations; the result is the geometric midpoint of the last "
            "interval, evaluated once more",
            "verification_after_the_search": {
                "i": "the metric at the two ends of the range brackets the target",
                "ii": "all evaluated sigmas in ascending order: the metric never rises by more "
                f"than {INVERSION_TOLERANCE} (a larger rise is a break of monotonicity)",
                "iii": "m_O and m_S at the result are each within the target "
                f"+- {TARGET_TOLERANCE}",
            },
            "success": "i, ii and iii all hold; otherwise 'unmet': the case is not run, the values "
            "reached are reported, the tolerance is not widened and no other search is used",
        },
        "calibration_of_tau": {
            "target": f"median partial Spearman (O, S | L) = {TAU_TARGET} +- {TARGET_TOLERANCE} at "
            "sigma_A over the calibration draws",
            "search": f"linear bisection over {list(TAU_RANGE)}, first point the midpoint, stop at "
            f"width < {WIDTH_TOLERANCE} or {MAX_ITERATIONS} iterations (increasing in tau)",
            "verification": "the same i, ii (the metric never falls by more than "
            f"{INVERSION_TOLERANCE}), iii; unmet means E is not run",
        },
        "random_numbers": {
            "generator": "numpy PCG64 from SeedSequence",
            "simulation_i_generation": f"SeedSequence([{MASTER_SEED}, case_number, i, 0])",
            "simulation_i_permutations": f"SeedSequence([{MASTER_SEED}, case_number, i, 1])",
            "calibration_j": f"SeedSequence([{MASTER_SEED}, case_number, {CALIBRATION_INDEX}, j])",
            "case_number": CASE_NUMBER,
            "draw_order_in_a_simulation": "E: six centroids, then noise of O, then noise of S; "
            "D: configuration of O, then of S",
        },
        "run": {
            "n_simulations": N_SIMULATIONS,
            "n_permutations": N_PERMUTATIONS,
            "n_permutations_registered_for_the_real_data": rc.N_PERMUTATIONS,
            "n_permutations_reduced_from_10000": True,
            "alphas": list(ALPHAS),
            "interval": "Clopper-Pearson 95 %",
            "order": "calibration of sigma and tau, verification, fix and record the values, then "
            "the simulations",
        },
        "rejection": {
            "partial_mantel_P": "p < alpha",
            "H2_rule_H": f"observed partial Spearman >= {rc.PARTIAL_MIN} and p < alpha",
            "spearman_mantel": "p < alpha",
            "knn_overlap": "observed > null 95th percentile",
        },
        "false_positive_criterion": {
            "cases": ["A", "B", "C", "D"],
            "indicators": [
                "P: partial Mantel rejection rate at alpha 0.01",
                "H: share with partial Spearman >= 0.2 and p < 0.01 (H <= P)",
            ],
            "accept": f"both at most {ACCEPT_RATE}",
            "boundary": {
                "above_0_03": "not acceptable for that indicator; no extension",
                "no_indicator_above_0_03_but_one_in_(0.02, 0.03]": "extend the case to a "
                f"cumulative {N_SIMULATIONS_EXTENDED} simulations (i = {N_SIMULATIONS} to "
                f"{N_SIMULATIONS_EXTENDED - 1}, the first {N_SIMULATIONS} are kept); the final "
                f"judgment needs both indicators at most {ACCEPT_RATE} on the cumulative runs",
                "the_two_indicators_are_treated_independently": True,
            },
            "interval_not_required": "the upper confidence limit is not required to be at most "
            "0.02 (too strict at a true rate of 0.01)",
        },
        "power_criterion_E": f"partial Mantel rejection rate at alpha 0.01 >= {POWER_MIN}",
        "A_A_64_A_full_comparison": COMPARISON_NOTE,
        "limits": "selective extension, synthetic data that only imitate L and a category "
        "structure; acceptable calibration does not guarantee calibration on the real data",
    }


# ---- the basis -------------------------------------------------------------------------------


@dataclass(frozen=True)
class MdsBasis:
    eigenvalues: Array  # all, descending
    eigenvectors: Array  # columns, in the same order
    p_full: int
    n_negative: int
    negative_abs_sum: float
    abs_sum: float
    positive_sum: float


def mds_basis(d_l: Array) -> MdsBasis:
    n = d_l.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d_l**2) @ j
    values, vectors = np.linalg.eigh((b + b.T) / 2.0)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    positive = values > POSITIVE_EIGENVALUE_TOLERANCE
    negative = values < -POSITIVE_EIGENVALUE_TOLERANCE
    return MdsBasis(
        eigenvalues=values,
        eigenvectors=vectors,
        p_full=int(np.sum(positive)),
        n_negative=int(np.sum(negative)),
        negative_abs_sum=float(np.sum(np.abs(values[negative]))),
        abs_sum=float(np.sum(np.abs(values))),
        positive_sum=float(np.sum(values[positive])),
    )


def coordinates(basis: MdsBasis, m: int, dim: int) -> Array:
    """``V[:, :m] sqrt(Lambda[:m])``, scaled so the mean row norm is 1, in the first ``m`` of
    ``dim``
    coordinates."""
    if m > basis.p_full or m > dim:
        raise ValueError("m is above the number of positive eigenvalues or of coordinates")
    z = basis.eigenvectors[:, :m] * np.sqrt(basis.eigenvalues[:m])
    z = z / float(np.mean(np.linalg.norm(z, axis=1)))
    out = np.zeros((z.shape[0], dim))
    out[:, :m] = z
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class CaseSetup:
    case: str
    m: int
    dim: int
    z: Array | None


def case_setup(case: str, basis: MdsBasis) -> CaseSetup:
    if case == "D":
        return CaseSetup("D", 0, EMBED_DIM, None)
    if case in ("A", "B", "C", "E"):
        m, dim = min(MDS_DIM_A, basis.p_full), EMBED_DIM
    elif case == "A_64":
        m, dim = min(EMBED_DIM, basis.p_full), EMBED_DIM
    elif case == "A_full":
        m, dim = basis.p_full, basis.p_full
    else:
        raise ValueError(f"unknown case {case!r}")
    return CaseSetup(case, m, dim, coordinates(basis, m, dim))


def basis_diagnostics(basis: MdsBasis, setup: CaseSetup, d_l: Array) -> dict[str, Any]:
    """What the truncation costs: counts, shares of the eigenvalues, and how well Z reproduces L."""
    out: dict[str, Any] = {
        "p_full": basis.p_full,
        "n_negative_eigenvalues": basis.n_negative,
        "negative_abs_sum": basis.negative_abs_sum,
        "negative_share_of_abs_sum": basis.negative_abs_sum / basis.abs_sum,
        "positive_sum": basis.positive_sum,
        "adopted_dimensions": setup.m,
        "noise_dimensions": setup.dim,
    }
    if setup.z is not None:
        adopted = float(np.sum(basis.eigenvalues[: setup.m]))
        out["adopted_share_of_positive_sum"] = adopted / basis.positive_sum
        out["spearman_Z_distance_vs_L"] = rc.spearman(
            rc.upper(rc.pairwise_euclidean(setup.z)), rc.upper(d_l)
        )
    return out


# ---- generation ------------------------------------------------------------------------------


def rng_for(case: str, i: int, kind: int) -> np.random.Generator:
    """Simulation ``i``: kind 0 is its generation, kind 1 its permutations."""
    seed = np.random.SeedSequence([MASTER_SEED, CASE_NUMBER[case], i, kind])
    return np.random.Generator(np.random.PCG64(seed))


def calibration_rng(case: str, j: int) -> np.random.Generator:
    seed = np.random.SeedSequence([MASTER_SEED, CASE_NUMBER[case], CALIBRATION_INDEX, j])
    return np.random.Generator(np.random.PCG64(seed))


def l2_rows(x: Array) -> Array:
    return np.asarray(x / np.linalg.norm(x, axis=1, keepdims=True), dtype=np.float64)


def make_pair(
    setup: CaseSetup,
    sigma: float,
    tau: float,
    categories: Sequence[str],
    rng: np.random.Generator,
) -> tuple[Array, Array]:
    """One (O, S) of the case. The noise is drawn as N(0, 1) and multiplied by ``sigma`` so that the
    draws of a seed are the same for every candidate sigma."""
    n = len(categories)
    if setup.case == "D":
        o_config = rng.standard_normal((n, setup.dim))
        s_config = rng.standard_normal((n, setup.dim))
        return l2_rows(o_config), l2_rows(s_config)
    assert setup.z is not None
    base = setup.z
    if setup.case == "E":
        names = sorted(set(categories))
        centroids = rng.standard_normal((len(names), setup.dim))
        base = setup.z + tau * centroids[[names.index(c) for c in categories]]
    noise_o = rng.standard_normal((n, setup.dim))
    noise_s = rng.standard_normal((n, setup.dim))
    return l2_rows(base + sigma * noise_o), l2_rows(base + sigma * noise_s)


# ---- the search ------------------------------------------------------------------------------


def bisect_search(
    f: Callable[[float], float],
    low: float,
    high: float,
    target: float,
    *,
    increasing: bool,
    geometric: bool,
) -> dict[str, Any]:
    """Bisection of a monotone ``f``. ``geometric``: in log x (the first point is the geometric
    mean).
    Returns the evaluations in order, the result (the midpoint of the last interval, evaluated once
    more) and the iteration count."""
    a, b = (math.log(low), math.log(high)) if geometric else (low, high)

    def to_x(t: float) -> float:
        return math.exp(t) if geometric else t

    evaluations: list[tuple[float, float]] = []
    iterations = 0
    while (b - a) >= WIDTH_TOLERANCE and iterations < MAX_ITERATIONS:
        mid = (a + b) / 2.0
        x = to_x(mid)
        value = f(x)
        evaluations.append((x, value))
        if increasing:
            a, b = (mid, b) if value < target else (a, mid)
        else:
            a, b = (mid, b) if value > target else (a, mid)
        iterations += 1
    result = to_x((a + b) / 2.0)
    evaluations.append((result, f(result)))
    return {"result": result, "evaluations": evaluations, "iterations": iterations}


def monotone(points: Sequence[tuple[float, float]], *, increasing: bool) -> dict[str, Any]:
    """The values in ascending order of x; the largest move against the direction is the
    inversion."""
    ordered = sorted(points)
    values = [v for _, v in ordered]
    worst = 0.0
    for earlier, later in zip(values, values[1:], strict=False):
        move = (earlier - later) if increasing else (later - earlier)
        worst = max(worst, move)
    return {"largest_inversion": worst, "monotone_within_tolerance": worst <= INVERSION_TOLERANCE}


def _median_rho(
    setup: CaseSetup, categories: Sequence[str], d_l: Array, sigma: float, n_cal: int
) -> dict[str, float]:
    z = rc.upper(d_l)
    rho_o, rho_s = [], []
    for j in range(n_cal):
        o, s = make_pair(setup, sigma, 0.0, categories, calibration_rng(setup.case, j))
        rho_o.append(rc.spearman(rc.upper(rc.pairwise_euclidean(o)), z))
        rho_s.append(rc.spearman(rc.upper(rc.pairwise_euclidean(s)), z))
    m_o, m_s = float(np.median(rho_o)), float(np.median(rho_s))
    return {"m_O": m_o, "m_S": m_s, "metric": (m_o + m_s) / 2.0}


def calibrate_sigma(
    setup: CaseSetup,
    categories: Sequence[str],
    d_l: Array,
    target: float,
    *,
    n_cal: int = N_CALIBRATION,
    sigma_range: tuple[float, float] = SIGMA_RANGE,
) -> dict[str, Any]:
    cache: dict[float, dict[str, float]] = {}

    def measure(sigma: float) -> dict[str, float]:
        if sigma not in cache:
            cache[sigma] = _median_rho(setup, categories, d_l, sigma, n_cal)
        return cache[sigma]

    search = bisect_search(
        lambda s: measure(s)["metric"], *sigma_range, target, increasing=False, geometric=True
    )
    low_end, high_end = measure(sigma_range[0]), measure(sigma_range[1])
    final = measure(search["result"])
    table = sorted(cache.items())
    check_i = low_end["metric"] >= target >= high_end["metric"]
    check_ii = monotone([(s, v["metric"]) for s, v in table], increasing=False)
    check_iii = (
        abs(final["m_O"] - target) <= TARGET_TOLERANCE
        and abs(final["m_S"] - target) <= TARGET_TOLERANCE
    )
    reasons = []
    if not check_i:
        reasons.append("i: the ends of the range do not bracket the target")
    if not check_ii["monotone_within_tolerance"]:
        reasons.append("ii: monotonicity broken beyond the tolerance")
    if not check_iii:
        reasons.append("iii: m_O or m_S is outside the target +- 0.03")
    return {
        "case": setup.case,
        "target": target,
        "sigma": search["result"],
        "m_O": final["m_O"],
        "m_S": final["m_S"],
        "iterations": search["iterations"],
        "ends": {
            "low": {"sigma": sigma_range[0], **low_end},
            "high": {"sigma": sigma_range[1], **high_end},
        },
        "check_i_brackets_target": bool(check_i),
        "check_ii_monotone": check_ii,
        "check_iii_within_tolerance": bool(check_iii),
        "evaluated": [{"sigma": s, **v} for s, v in table],
        "status": "calibrated" if not reasons else "unmet",
        "unmet_reasons": reasons,
        "highest_metric_reached": max(v["metric"] for _, v in table),
    }


def calibrate_tau(
    setup: CaseSetup,
    categories: Sequence[str],
    d_l: Array,
    sigma: float,
    *,
    n_cal: int = N_CALIBRATION,
    target: float = TAU_TARGET,
    tau_range: tuple[float, float] = TAU_RANGE,
) -> dict[str, Any]:
    cache: dict[float, float] = {}
    z = rc.upper(d_l)

    def measure(tau: float) -> float:
        if tau not in cache:
            values = []
            for j in range(n_cal):
                o, s = make_pair(setup, sigma, tau, categories, calibration_rng("E", j))
                values.append(
                    rc.partial_spearman(
                        rc.upper(rc.pairwise_euclidean(o)), rc.upper(rc.pairwise_euclidean(s)), z
                    )
                )
            cache[tau] = float(np.median(values))
        return cache[tau]

    search = bisect_search(measure, *tau_range, target, increasing=True, geometric=False)
    low_end, high_end = measure(tau_range[0]), measure(tau_range[1])
    final = measure(search["result"])
    table = sorted(cache.items())
    check_i = low_end <= target <= high_end
    check_ii = monotone(table, increasing=True)
    check_iii = abs(final - target) <= TARGET_TOLERANCE
    reasons = []
    if not check_i:
        reasons.append("i: the ends of the range do not bracket the target")
    if not check_ii["monotone_within_tolerance"]:
        reasons.append("ii: monotonicity broken beyond the tolerance")
    if not check_iii:
        reasons.append("iii: the median partial Spearman is outside the target +- 0.03")
    return {
        "case": "E",
        "target": target,
        "sigma_used": sigma,
        "tau": search["result"],
        "median_partial_spearman": final,
        "iterations": search["iterations"],
        "ends": {
            "low": {"tau": tau_range[0], "median": low_end},
            "high": {"tau": tau_range[1], "median": high_end},
        },
        "check_i_brackets_target": bool(check_i),
        "check_ii_monotone": check_ii,
        "check_iii_within_tolerance": bool(check_iii),
        "evaluated": [{"tau": t, "median_partial_spearman": v} for t, v in table],
        "status": "calibrated" if not reasons else "unmet",
        "unmet_reasons": reasons,
        "highest_median_reached": max(v for _, v in table),
    }


# ---- the simulations -------------------------------------------------------------------------


def simulate_one(
    setup: CaseSetup,
    sigma: float,
    tau: float,
    d_l: Array,
    categories: Sequence[str],
    i: int,
    *,
    n_permutations: int = N_PERMUTATIONS,
) -> dict[str, Any]:
    """Simulation ``i`` of a case: the numbers of the three tests and the realised Spearman
    values."""
    o, s = make_pair(setup, sigma, tau, categories, rng_for(setup.case, i, 0))
    d_o, d_s = rc.pairwise_euclidean(o), rc.pairwise_euclidean(s)
    perms = rc.permutations(
        d_l.shape[0],
        n_permutations,
        np.random.SeedSequence([MASTER_SEED, CASE_NUMBER[setup.case], i, 1]),
    )
    suite = rc.mantel_suite(d_o, d_s, d_l, perms)
    z = rc.upper(d_l)
    return {
        "i": i,
        "rho_O_L": rc.spearman(rc.upper(d_o), z),
        "rho_S_L": rc.spearman(rc.upper(d_s), z),
        "spearman_O_S": suite["spearman"]["observed"],
        "spearman_p": suite["spearman"]["p"],
        "partial_O_S": suite["partial"]["observed"],
        "partial_p": suite["partial"]["p"],
        "knn_observed": suite["knn_overlap"]["observed"],
        "knn_null_p95": suite["knn_overlap"]["null_p95"],
    }


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """The exact binomial interval (bisection on the binomial tails; no scipy)."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("0 <= k <= n and n > 0")
    tail = (1.0 - level) / 2.0

    def log_pmf(x: int, p: float) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(x + 1)
            - math.lgamma(n - x + 1)
            + x * math.log(p)
            + (n - x) * math.log1p(-p)
        )

    def upper_tail(p: float) -> float:  # P(X >= k)
        return float(sum(math.exp(log_pmf(x, p)) for x in range(k, n + 1)))

    def lower_tail(p: float) -> float:  # P(X <= k)
        return float(sum(math.exp(log_pmf(x, p)) for x in range(0, k + 1)))

    def solve(g: Callable[[float], float], rising: bool) -> float:
        low, high = 1e-12, 1.0 - 1e-12
        for _ in range(100):
            mid = (low + high) / 2.0
            if (g(mid) < tail) == rising:
                low = mid
            else:
                high = mid
        return (low + high) / 2.0

    lower = 0.0 if k == 0 else solve(upper_tail, True)  # P(X>=k|p) rises with p
    upper = 1.0 if k == n else solve(lower_tail, False)  # P(X<=k|p) falls with p
    return lower, upper


def _rate(count: int, n: int) -> dict[str, Any]:
    low, high = clopper_pearson(count, n)
    return {"rejections": count, "n": n, "rate": count / n, "ci95": [low, high]}


def distribution(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    q = np.percentile(v, [2.5, 97.5])
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p2_5": float(q[0]),
        "p97_5": float(q[1]),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def summarize_case(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(records)
    out: dict[str, Any] = {"n_simulations": n, "n_permutations": N_PERMUTATIONS}
    rate: dict[str, Any] = {
        "P_partial_mantel": {},
        "H_partial_ge_0_2_and_p": {},
        "spearman_mantel": {},
    }
    for a in ALPHAS:
        key = f"alpha_{a}"
        rate["P_partial_mantel"][key] = _rate(sum(1 for r in records if r["partial_p"] < a), n)
        rate["H_partial_ge_0_2_and_p"][key] = _rate(
            sum(1 for r in records if r["partial_O_S"] >= rc.PARTIAL_MIN and r["partial_p"] < a), n
        )
        rate["spearman_mantel"][key] = _rate(sum(1 for r in records if r["spearman_p"] < a), n)
    rate["knn_overlap_above_null_p95"] = _rate(
        sum(1 for r in records if r["knn_observed"] > r["knn_null_p95"]), n
    )
    out["rejection"] = rate
    out["partial_ge_0_2_share"] = sum(1 for r in records if r["partial_O_S"] >= rc.PARTIAL_MIN) / n
    out["realised"] = {
        "rho_O_L": distribution([r["rho_O_L"] for r in records]),
        "rho_S_L": distribution([r["rho_S_L"] for r in records]),
        "spearman_O_S": distribution([r["spearman_O_S"] for r in records]),
        "partial_O_S": distribution([r["partial_O_S"] for r in records]),
    }
    return out


def boundary_decision(p_rate: float, h_rate: float) -> dict[str, Any]:
    """The pre-registered handling of the first 500 simulations of a false-positive case."""
    over_hard = {"P": p_rate > EXTEND_UP_TO_RATE, "H": h_rate > EXTEND_UP_TO_RATE}
    if any(over_hard.values()):
        return {
            "action": "not_acceptable_no_extension",
            "not_acceptable_for": [k for k, v in over_hard.items() if v],
        }
    if p_rate > ACCEPT_RATE or h_rate > ACCEPT_RATE:
        return {"action": "extend_to_2000", "not_acceptable_for": []}
    return {"action": "acceptable", "not_acceptable_for": []}


def final_decision(p_rate: float, h_rate: float) -> dict[str, Any]:
    """After the cumulative 2000: both indicators must be at most ``ACCEPT_RATE``."""
    bad = [k for k, v in (("P", p_rate), ("H", h_rate)) if v > ACCEPT_RATE]
    return {"action": "acceptable" if not bad else "not_acceptable", "not_acceptable_for": bad}


def power_decision(rate_at_001: float) -> dict[str, Any]:
    ok = rate_at_001 >= POWER_MIN
    return {
        "criterion": f"partial Mantel rejection rate at alpha 0.01 >= {POWER_MIN}",
        "rate": rate_at_001,
        "detects_this_scale": ok,
        "reading": "この規模の共通構造は検出できる"
        if ok
        else "この方法は、この規模の共通構造を検出する力が足りない。"
        "実データの偏 Mantel が不成立でも、"
        "「共通構造が無い」とは結論づけない（検出力の限界つきの結果）",
    }
