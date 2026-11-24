"""RSA-X3: comparing two spaces of the same items, as pure functions (numpy only).

``O`` is the observed space (the 64-D K positions of the items), ``S`` the string space
(``pipeline.project(text)``). Nothing here reads a store, a kernel or a text; the functions take
arrays and labels and give numbers. The analysis is registered before any real result is seen:
``analysis_spec()`` is that registration (a dict; ``spec_sha256()`` its hash) and every
default below
is a value of it.

Choices that the registration leaves to the implementation are made here and written into the spec:

* the S side is the one that is permuted (rows and columns together, the same permutation);
* the partial statistic works on ranks (a partial Spearman): the three pair vectors are replaced by
  their average ranks, each of O and S is regressed (OLS, intercept) on the word-overlap distance L,
  and the statistic is the Pearson correlation of the two residual vectors (Smouse-Long-Sokal); the
  residual vector of S is the one that is permuted;
* k-NN: the ``k`` smallest distances of a row, the item itself excluded; equal distances are ordered
  by item index (stable); the overlap of an item is ``|N_O ∩ N_S| / k``, the statistic the mean over
  items;
* triples: every (anchor, {b, c}) of the items, no sampling; a triple where either space has equal
  distances is left out of the rate and counted;
* AUC: the chance that a pair of the first kind is nearer than a pair of the other kind (ties 1/2);
* centering: subtracting the mean does not change a Euclidean distance, so the centered version is
  "subtract the mean of the 72 items, then give every vector unit L2 norm again".
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

SEED: Final = 20261117
N_PERMUTATIONS: Final = 10_000
KNN_K: Final = 5
RETRIEVAL_K: Final = 3
SAFE_RADIUS: Final = 0.65
PARTIAL_MIN: Final = 0.2
P_MAX: Final = 0.01
NULL_PERCENTILE: Final = 95
KINDS: Final = ("near", "far", "reversed", "unrelated", "other")
ROLE_CODE: Final = {"near_paraphrase": "n", "far_paraphrase": "f", "reversed_meaning": "r"}
_WORD: Final = re.compile(r"[a-z]+")


# ---- the registration ------------------------------------------------------------------------


#: The registration this one replaced, before any statistic of the real data was computed.
SUPERSEDES: Final = {
    "spec_version": "rsa-analysis-spec/1",
    "spec_hash_sha256": "52f482c4b708bac33fc274fb9c22af30f9c32934bfba13e4b40d903986108f49",
    "spec_file_sha256": "281083e52d88109928a0bdac1ebd6a99af4b76865204a32738e1bd5820182830",
    "replaced": "before any statistic of the real O and S was computed "
    "(design v0.4, DC-RSA-5 to 7)",
    "changed": [
        "criterion_DC_RSA_2 is judged for the raw O only; the centered O is a sensitivity analysis",
        "d_triples is descriptive only: no null distribution, not part of any criterion",
        "the p value of the partial Mantel is provisional until the calibration is confirmed",
        "the eight-outcome decision table of design v1.0 3B is part of the registration, and H1 "
        "(Spearman Mantel p < 0.01) is part of DC-RSA-2",
    ],
    "not_changed": "permutation method, seed, n_permutations, k-NN, stratification, 7.1, 7.2",
    "this_file_replaced": "an earlier draft of spec/2 (hash 556d53e0..., in the incomplete folder "
    "rsa-x2b-20261117-201844, written before the frozen design v1.0)",
}


def analysis_spec() -> dict[str, Any]:
    """The registered method, as data. Its hash is reported before any real value is computed."""
    return {
        "spec_version": "rsa-analysis-spec/2",
        "supersedes": SUPERSEDES,
        "design": "string-projection vs observed-position relation, design v1.0 (sha256 "
        "395828f6709bad0c85aa900a8d100e87a6e55cc56f8c5cbdd450d3bfa4bdcb07), sections 3, 3B, "
        "6, 6A, 7, 7B, 9",
        "spaces": {
            "O": "64-D K position of the item (row 0, last prompt token), Euclidean distance",
            "S": "pipeline.project(text), 64-D, Euclidean distance",
            "O_versions": [
                "raw (primary analysis)",
                "centered (sensitivity analysis): subtract the mean of the 72 items, then unit L2 "
                "norm again (subtracting the mean alone leaves Euclidean distances unchanged)",
            ],
        },
        "items": {"n_items": 72, "n_pairs": 2556, "references": 3, "references_in_pairs": False},
        "pair_kinds": {
            "same_category": {
                "near": "(n,n)",
                "far": "(n,f)",
                "reversed": "(n,r)",
                "other": "(f,f),(f,r),(r,r)",
            },
            "different_category": "unrelated",
            "main_decision_uses": ["near", "far", "reversed", "unrelated"],
        },
        "word_overlap": "1 - Jaccard of the sets of lower-case [a-z]+ words",
        "metrics": {
            "a_spearman": "Spearman (average ranks) of the O and S pair distances",
            "b_partial": "partial Spearman controlling the word-overlap distance "
            "(Smouse-Long-Sokal, "
            "OLS residuals of ranks; the S residual matrix is permuted)",
            "c_knn": {"k": KNN_K, "ties": "by item index", "statistic": "mean of |N_O ∩ N_S| / k"},
            "d_triples": "all (anchor, {b, c}) triples; triples with a tie in either space are "
            "excluded "
            "and counted; agreement = same sign of d(a,b) - d(a,c)",
            "d_triples_status": "descriptive_only: no null distribution, no part of any criterion",
            "e_retrieval": {
                "k": RETRIEVAL_K,
                "definition": "share of the 72 items whose k nearest items (self excluded, within "
                "one "
                "space) hold at least one near-paraphrase item of the same category (not itself)",
                "spaces": ["O", "S"],
            },
        },
        "test": {
            "permutation": "item permutation applied to rows and columns of the S matrix together",
            "rng": "numpy PCG64",
            "seed": SEED,
            "n_permutations": N_PERMUTATIONS,
            "sides": "one-sided, positive",
            "p_value": "(1 + #{null >= observed}) / (1 + n_permutations)",
            "same_permutation_sequence_for": ["a_spearman", "b_partial", "c_knn"],
            "null_percentile_method": f"numpy percentile {NULL_PERCENTILE}, linear",
        },
        "criterion_DC_RSA_2": {
            "partial_min": PARTIAL_MIN,
            "p_max": P_MAX,
            "knn_overlap": "observed > null 95th percentile",
            "H1": "Spearman Mantel, one-sided, p < 0.01",
            "H2": "partial Spearman >= partial_min and p < p_max",
            "H3": "k-NN overlap > null 95th percentile",
            "met_when": "H1, H2 and H3 all hold",
            "judged_for": "raw O only",
            "centered_O": "sensitivity analysis; not used in the judgment",
            "partial_mantel_p_status": "provisional until the calibration of the permutation test "
            "is confirmed (DC-RSA-7); not confirmatory before that",
        },
        "decision_table_DC_RSA_8": [
            {"H1": a, "H2": b, "H3": c, "reading": text} for a, b, c, text in DECISION_TABLE
        ],
        "decision_notes": "a result that includes H2 is exploratory until the calibration "
        "(DC-RSA-7) is confirmed acceptable; a power shortfall of case E is added to a row that "
        "says 'not confirmed'; centered O, triples, 7.1 and 7.2 are not used in the judgment",
        "calibration": "registered in its own file (calibration-spec.json): design v1.0, 7B",
        "stratification": "pairs split by the quartiles of the word-overlap distance; Spearman per "
        "stratum",
        "scale_7_1": {
            "distributions": "n, min, q25, median, q75, max, mean, std per pair kind, O and S",
            "auc": ["near_vs_reversed", "near_vs_far", "near_vs_unrelated"],
        },
        "scale_7_2": {
            "reference_positions": "O positions of ref1..ref3 (raw version)",
            "radius": SAFE_RADIUS,
            "inside": "distance < radius (a distance equal to the radius is outside)",
            "groups": "category C1: near/far/reversed items; all other categories: one group",
            "reported": "per reference and the mean of the three; spread across references",
            "diagnostic_only": True,
        },
    }


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def spec_sha256() -> str:
    return hashlib.sha256(canonical_bytes(analysis_spec())).hexdigest()


# ---- distances -------------------------------------------------------------------------------


def pairwise_euclidean(points: Array) -> Array:
    """Euclidean distance matrix: symmetric, exactly zero on the diagonal."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("points must be a 2-D array of at least two rows")
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return np.asarray(d, dtype=np.float64)


def center_renormalize(points: Array) -> Array:
    """Subtract the mean of the rows, then give every row unit L2 norm."""
    x = np.asarray(points, dtype=np.float64)
    shifted = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(shifted, axis=1, keepdims=True)
    if bool(np.any(norms <= 0.0)):
        raise ValueError("a row equals the mean: it cannot be renormalized")
    return np.asarray(shifted / norms, dtype=np.float64)


def words_of(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def word_overlap_distance(texts: Sequence[str]) -> Array:
    """1 - Jaccard of the sets of lower-case alphabetic words (two empty sets: distance 0)."""
    sets = [words_of(t) for t in texts]
    n = len(sets)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            union = len(sets[i] | sets[j])
            d[i, j] = d[j, i] = 0.0 if union == 0 else 1.0 - len(sets[i] & sets[j]) / union
    return d


def upper(matrix: Array) -> Array:
    n = matrix.shape[0]
    return np.asarray(matrix[np.triu_indices(n, 1)], dtype=np.float64)


def square_from_upper(vector: Array, n: int) -> Array:
    out = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, 1)
    out[iu] = vector
    return out + out.T


def check_distance_matrix(d: Array) -> None:
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("a distance matrix is square")
    if not np.all(np.isfinite(d)):
        raise ValueError("a distance matrix is finite")
    if not np.array_equal(d, d.T) or bool(np.any(np.diag(d) != 0.0)):
        raise ValueError("a distance matrix is symmetric with a zero diagonal")


# ---- ranks and correlations ------------------------------------------------------------------


def rank_average(values: Array) -> Array:
    """Ranks 1..n, equal values share their average rank."""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(1, v.size + 1, dtype=np.float64)
    sorted_v = v[order]
    start = 0
    for end in range(1, v.size + 1):
        if end == v.size or sorted_v[end] != sorted_v[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    return ranks


def pearson(x: Array, y: Array) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    return float("nan") if denominator == 0.0 else float(np.sum(xc * yc) / denominator)


def spearman(x: Array, y: Array) -> float:
    return pearson(
        rank_average(np.asarray(x, dtype=np.float64)), rank_average(np.asarray(y, dtype=np.float64))
    )


def residuals(y: Array, z: Array) -> Array:
    """The residuals of ``y`` regressed on ``z`` (OLS with an intercept)."""
    zc = z - z.mean()
    slope = float(np.sum(zc * (y - y.mean())) / np.sum(zc * zc))
    return np.asarray(y - y.mean() - slope * zc, dtype=np.float64)


def partial_spearman(o: Array, s: Array, control: Array) -> float:
    """The partial Spearman of ``o`` and ``s`` given ``control`` (all pair vectors)."""
    ro, rs, rz = rank_average(o), rank_average(s), rank_average(control)
    return pearson(residuals(ro, rz), residuals(rs, rz))


def partial_by_formula(o: Array, s: Array, control: Array) -> float:
    """The same number from the correlations (a check of the residual form)."""
    r_os, r_oz, r_sz = spearman(o, s), spearman(o, control), spearman(s, control)
    return float((r_os - r_oz * r_sz) / np.sqrt((1 - r_oz**2) * (1 - r_sz**2)))


# ---- permutations ----------------------------------------------------------------------------


def permutations(
    n_items: int,
    n_permutations: int = N_PERMUTATIONS,
    seed: int | np.random.SeedSequence = SEED,
) -> NDArray[np.int64]:
    """The permutation sequence: PCG64 with ``seed`` (an int, or a ``SeedSequence``),
    ``rng.permutation(n_items)`` in a row."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.stack([rng.permutation(n_items) for _ in range(n_permutations)]).astype(np.int64)


def permute_matrix(matrix: Array, perm: NDArray[np.int64]) -> Array:
    """Rows and columns together: ``out[i, j] = matrix[perm[i], perm[j]]``."""
    return np.asarray(matrix[np.ix_(perm, perm)], dtype=np.float64)


def p_value(observed: float, null: Array) -> float:
    return float((1 + int(np.sum(null >= observed))) / (1 + null.size))


# ---- k-NN, triples, retrieval ----------------------------------------------------------------


def knn_indices(d: Array, k: int) -> NDArray[np.int64]:
    """The ``k`` nearest items of every row (self excluded; equal distances by item index)."""
    work = d.copy()
    np.fill_diagonal(work, np.inf)
    return np.asarray(np.argsort(work, axis=1, kind="stable")[:, :k], dtype=np.int64)


def knn_overlap(d_a: Array, d_b: Array, k: int = KNN_K) -> float:
    a, b = knn_indices(d_a, k), knn_indices(d_b, k)
    shared = [len(set(a[i].tolist()) & set(b[i].tolist())) for i in range(a.shape[0])]
    return float(np.mean(shared) / k)


def triple_agreement(d_a: Array, d_b: Array) -> dict[str, Any]:
    """Every (anchor a, {b, c}): do the two spaces order d(a,b) and d(a,c) the same way?"""
    n = d_a.shape[0]
    agree = total = tied = 0
    iu = np.triu_indices(n - 1, 1)
    for a in range(n):
        others = np.array([i for i in range(n) if i != a])
        da, db = d_a[a, others], d_b[a, others]
        diff_a = (da[:, None] - da[None, :])[iu]
        diff_b = (db[:, None] - db[None, :])[iu]
        tie = (diff_a == 0.0) | (diff_b == 0.0)
        tied += int(np.sum(tie))
        keep = ~tie
        total += int(np.sum(keep))
        agree += int(np.sum(np.sign(diff_a[keep]) == np.sign(diff_b[keep])))
    return {
        "method": "all triples, no sampling",
        "n_triples": total + tied,
        "n_tied_excluded": tied,
        "agreement": float(agree / total) if total else float("nan"),
    }


def near_hit_rate(
    d: Array, categories: Sequence[str], roles: Sequence[str], k: int = RETRIEVAL_K
) -> dict[str, Any]:
    """Within one space: the share of items whose ``k`` nearest items hold a near-paraphrase of the
    same category (not the item itself), with the share expected for random neighbours."""
    n = d.shape[0]
    near = np.array([ROLE_CODE.get(r) == "n" for r in roles])
    cat = np.asarray(categories)
    neighbours = knn_indices(d, k)
    hits: list[bool] = []
    expected: list[float] = []
    by_role: dict[str, list[bool]] = {"n": [], "f": [], "r": []}
    for i in range(n):
        relevant = near & (cat == cat[i])
        relevant[i] = False
        m = int(relevant.sum())
        hit = bool(np.any(relevant[neighbours[i]]))
        hits.append(hit)
        by_role.setdefault(ROLE_CODE.get(roles[i], "?"), []).append(hit)
        pool = n - 1
        none = 1.0 if m == 0 else _choose(pool - m, k) / _choose(pool, k)
        expected.append(1.0 - none)
    return {
        "k": k,
        "hit_rate": float(np.mean(hits)),
        "expected_for_random_neighbours": float(np.mean(expected)),
        "hit_rate_by_query_role": {
            r: (float(np.mean(v)) if v else None) for r, v in by_role.items()
        },
    }


def _choose(n: int, k: int) -> float:
    from math import comb

    return float(comb(n, k)) if n >= k else 0.0


# ---- pair kinds, summaries, AUC --------------------------------------------------------------


def pair_kinds(categories: Sequence[str], roles: Sequence[str]) -> NDArray[np.str_]:
    """The kind of every pair (upper triangle order): near (n,n), far (n,f), reversed (n,r) in one
    category; other for (f,f), (f,r), (r,r) of one category; unrelated across categories."""
    n = len(categories)
    codes = [ROLE_CODE[r] for r in roles]
    kinds = []
    rows, cols = np.triu_indices(n, 1)
    for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
        i, j = int(a), int(b)
        if categories[i] != categories[j]:
            kinds.append("unrelated")
            continue
        kinds.append(_same_category_kind("".join(sorted((codes[i], codes[j])))))
    return np.asarray(kinds)


def _same_category_kind(pair: str) -> str:
    return {"nn": "near", "fn": "far", "nr": "reversed"}.get(pair, "other")


def summarize(values: Array) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    q = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "q25": float(q[0]),
        "median": float(q[1]),
        "q75": float(q[2]),
        "max": float(np.max(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def auc_nearer(first: Array, other: Array) -> float:
    """The chance that a distance of ``first`` is smaller than one of ``other`` (ties count 1/2)."""
    n1, n2 = first.size, other.size
    if n1 == 0 or n2 == 0:
        return float("nan")
    ranks = rank_average(np.concatenate([first, other]))
    u_other = float(ranks[n1:].sum() - n2 * (n2 + 1) / 2.0)
    return u_other / (n1 * n2)


def kind_distributions(d: Array, kinds: NDArray[np.str_]) -> dict[str, Any]:
    v = upper(d)
    out: dict[str, Any] = {"distance": {k: summarize(v[kinds == k]) for k in KINDS}}
    near = v[kinds == "near"]
    out["auc_near_is_nearer_than"] = {
        other: auc_nearer(near, v[kinds == other]) for other in ("reversed", "far", "unrelated")
    }
    return out


def stratified_spearman(d_o: Array, d_s: Array, d_l: Array) -> list[dict[str, Any]]:
    """Spearman of O and S within the quartile strata of the word-overlap distance L."""
    o, s, z = upper(d_o), upper(d_s), upper(d_l)
    edges = np.percentile(z, [25, 50, 75])
    index = np.digitize(
        z, edges, right=True
    )  # 0..3, a value equal to an edge goes to the lower stratum
    rows = []
    for b in range(4):
        m = index == b
        rows.append(
            {
                "stratum": b + 1,
                "n_pairs": int(m.sum()),
                "word_distance_range": [float(np.min(z[m])), float(np.max(z[m]))]
                if m.any()
                else None,
                "spearman": spearman(o[m], s[m]) if m.sum() > 2 else None,
            }
        )
    return rows


# ---- the tests of H1-H3 ----------------------------------------------------------------------


def mantel_suite(
    d_o: Array,
    d_s: Array,
    d_l: Array,
    perms: NDArray[np.int64],
    *,
    k: int = KNN_K,
) -> dict[str, Any]:
    """Spearman, partial Spearman and k-NN overlap of O against S, each with the null of the same
    permutation sequence applied to S (rows and columns together). One-sided."""
    for matrix in (d_o, d_s, d_l):
        check_distance_matrix(matrix)
    n = d_o.shape[0]
    if perms.shape[1] != n:
        raise ValueError("the permutations are not of this number of items")
    o, s, z = upper(d_o), upper(d_s), upper(d_l)
    ro, rs, rz = rank_average(o), rank_average(s), rank_average(z)
    res_o, res_s = residuals(ro, rz), residuals(rs, rz)
    rank_s_matrix = square_from_upper(rs, n)
    res_s_matrix = square_from_upper(res_s, n)
    iu = np.triu_indices(n, 1)
    ro_c, res_o_c = ro - ro.mean(), res_o - res_o.mean()
    ro_norm, res_o_norm = float(np.sqrt(np.sum(ro_c**2))), float(np.sqrt(np.sum(res_o_c**2)))

    observed = {
        "spearman": pearson(ro, rs),
        "partial": pearson(res_o, res_s),
        "knn_overlap": knn_overlap(d_o, d_s, k),
    }
    nulls = {name: np.empty(perms.shape[0]) for name in observed}
    for t, perm in enumerate(perms):
        vs = rank_s_matrix[np.ix_(perm, perm)][iu]
        vs_c = vs - vs.mean()
        nulls["spearman"][t] = float(np.sum(ro_c * vs_c) / (ro_norm * np.sqrt(np.sum(vs_c**2))))
        vr = res_s_matrix[np.ix_(perm, perm)][iu]
        vr_c = vr - vr.mean()
        nulls["partial"][t] = float(
            np.sum(res_o_c * vr_c) / (res_o_norm * np.sqrt(np.sum(vr_c**2)))
        )
        nulls["knn_overlap"][t] = knn_overlap(d_o, permute_matrix(d_s, perm), k)
    out: dict[str, Any] = {
        "n_items": n,
        "n_pairs": int(o.size),
        "n_permutations": int(perms.shape[0]),
    }
    for name, value in observed.items():
        null = nulls[name]
        out[name] = {
            "observed": value,
            "p": p_value(value, null),
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "null_p95": float(np.percentile(null, NULL_PERCENTILE)),
            "null_max": float(np.max(null)),
        }
    out["partial_pearson_secondary"] = partial_pearson(o, s, z)
    return out


def partial_pearson(o: Array, s: Array, control: Array) -> float:
    """Secondary (not used by the criterion): the partial correlation of the raw distances."""
    return pearson(residuals(o, control), residuals(s, control))


def criterion_dc_rsa_2(suite: Mapping[str, Any]) -> dict[str, Any]:
    """H1 (Spearman Mantel p < 0.01), H2 (partial Spearman >= 0.2 and p < 0.01), H3 (k-NN overlap
    above the null 95th percentile); DC-RSA-2 is met when all three hold (design v1.0, 3B)."""
    spearman_, partial, knn = suite["spearman"], suite["partial"], suite["knn_overlap"]
    h1 = spearman_["p"] < P_MAX
    h2 = partial["observed"] >= PARTIAL_MIN and partial["p"] < P_MAX
    h3 = knn["observed"] > knn["null_p95"]
    return {
        "H1_spearman_mantel_p_below_0_01": bool(h1),
        "partial_at_least_0_2_and_p_below_0_01": bool(h2),
        "knn_above_null_95th_percentile": bool(h3),
        "met": bool(h1 and h2 and h3),
        "interpretation": decide(bool(h1), bool(h2), bool(h3)),
    }


#: Design v1.0, 3B (DC-RSA-8): the eight outcomes and their reading, as registered.
DECISION_TABLE: Final[tuple[tuple[bool, bool, bool, str], ...]] = (
    (False, False, False, "関係性の保存を確認できない"),
    (
        False,
        False,
        True,
        "距離順位の保存は確認できない。近傍の一致のみ（語の重なりを統制していない）。"
        "部分的で、語の重なりで説明されうる",
    ),
    (
        True,
        False,
        False,
        "距離順位は保たれるが、語の重なりを超える共通性を確認できない。近傍保存も確認できない",
    ),
    (
        True,
        False,
        True,
        "距離順位と近傍は保たれるが、語の重なりを超える共通性を確認できない"
        "（語の重なりで説明されうる）",
    ),
    (
        True,
        True,
        False,
        "距離順位の、語の重なりを超える共通性を支持。近傍保存は確認できない（部分的）",
    ),
    (True, True, True, "関係性と近傍の保存を支持（成立）"),
    (
        False,
        True,
        False,
        "不整合（統制後だけ正）。確認できないと記録し、共変量による抑制の可能性を別に報告する。"
        "支持とは書かない",
    ),
    (False, True, True, "同上（不整合）。近傍の一致は報告するが、支持とは書かない"),
)


def decide(h1: bool, h2: bool, h3: bool) -> dict[str, Any]:
    """The registered reading of (H1, H2, H3). A result that includes H2 is exploratory until the
    calibration (DC-RSA-7) is confirmed acceptable; a power shortfall of case E is added to a row
    that says "not confirmed" (``power_caveat_applies``)."""
    for a, b, c, text in DECISION_TABLE:
        if (a, b, c) == (h1, h2, h3):
            return {
                "H1": h1,
                "H2": h2,
                "H3": h3,
                "reading": text,
                "includes_H2": h2,
                "exploratory_until_calibration_acceptable": bool(h2),
                "power_caveat_applies": "確認できない" in text,
            }
    raise ValueError("unreachable: the table holds all eight outcomes")


# ---- 7.2: distances from reference positions -------------------------------------------------


def reference_distances(
    references: Array,
    items: Array,
    categories: Sequence[str],
    roles: Sequence[str],
    reference_category: str,
    radius: float = SAFE_RADIUS,
) -> dict[str, Any]:
    """Distances of every item to every reference position, by group (the reference's own category:
    near / far / reversed; every other category: one group), and the share inside ``radius``
    (strictly inside). Diagnostic only; nothing is injected anywhere."""
    group = np.array(
        [
            (
                {"n": "near", "f": "far", "r": "reversed"}[ROLE_CODE[r]]
                if c == reference_category
                else "other_category"
            )
            for c, r in zip(categories, roles, strict=True)
        ]
    )
    names = ("near", "far", "reversed", "other_category")
    per_reference = []
    for ref in references:
        dist = np.sqrt(np.sum((items - ref[None, :]) ** 2, axis=1))
        per_reference.append(
            {
                g: {
                    **summarize(dist[group == g]),
                    "inside_share": float(np.mean(dist[group == g] < radius))
                    if np.any(group == g)
                    else None,
                }
                for g in names
            }
        )
    shares = {g: [r[g]["inside_share"] for r in per_reference] for g in names}
    mean_share = {g: (None if None in v else float(np.mean(v))) for g, v in shares.items()}
    return {
        "radius": radius,
        "inside": "distance < radius",
        "n_references": len(per_reference),
        "per_reference": per_reference,
        "mean_inside_share_over_references": mean_share,
        "inside_share_spread_across_references": {
            g: (None if None in v else {"min": min(v), "max": max(v), "std": float(np.std(v))})
            for g, v in shares.items()
        },
    }


# ---- the report ------------------------------------------------------------------------------


def compare_report(
    o_points: Array,
    s_points: Array,
    d_l: Array,
    categories: Sequence[str],
    roles: Sequence[str],
    reference_points: Array,
    reference_category: str,
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Every registered number, for O raw and O centered, as a dict of numbers (no text, no
    vector). ``o_points`` and ``s_points`` are (n_items, 64); ``d_l`` the word-overlap distances."""
    n = o_points.shape[0]
    perms = permutations(n, n_permutations, seed)
    kinds = pair_kinds(categories, roles)
    d_s = pairwise_euclidean(s_points)
    report: dict[str, Any] = {
        "spec_sha256": spec_sha256(),
        "n_items": n,
        "kind_counts": {k: int(np.sum(kinds == k)) for k in KINDS},
        "S": {
            "scale_7_1": kind_distributions(d_s, kinds),
            "retrieval_hit_at_3": near_hit_rate(d_s, categories, roles),
        },
        "O": {},
    }
    for name, points in (("raw", o_points), ("centered", center_renormalize(o_points))):
        d_o = pairwise_euclidean(points)
        suite = mantel_suite(d_o, d_s, d_l, perms)
        report["O"][name] = {
            "mantel": suite,
            "role": "primary" if name == "raw" else "sensitivity",
            "triples": {**triple_agreement(d_o, d_s), "descriptive_only": True},
            "stratified_spearman_by_word_overlap_quartile": stratified_spearman(d_o, d_s, d_l),
            "scale_7_1": kind_distributions(d_o, kinds),
            "retrieval_hit_at_3": near_hit_rate(d_o, categories, roles),
        }
    report["O"]["raw"]["criterion_DC_RSA_2"] = criterion_dc_rsa_2(report["O"]["raw"]["mantel"])
    report["scale_7_2"] = reference_distances(
        reference_points, o_points, categories, roles, reference_category
    )
    return report
