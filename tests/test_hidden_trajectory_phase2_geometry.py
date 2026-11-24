"""Phase 2 (anchor geometry): synthetic tests of the labels, the three methods and M1-M5."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from nvs_runtime.hidden_trajectory.phase2 import evaluate as ev
from nvs_runtime.hidden_trajectory.phase2 import geometry as gm
from nvs_runtime.hidden_trajectory.phase2.labels import (
    LabelError,
    Labels,
    category_of,
    load_labels,
    role_of,
)

REFS = ["ref1", "ref2", "ref3"]


def _line(coords: list[float]) -> np.ndarray:
    d = np.zeros((len(coords), len(coords)))
    for i, a in enumerate(coords):
        for j, b in enumerate(coords):
            d[i, j] = abs(a - b)
    return d


def test_radius_boundary_is_inside() -> None:
    d = _line([0.0, 5.0, 9.0, 0.65, 0.66])
    assert gm.decide_radius(d, 3, (0,)) == (True, 0.65)
    assert gm.decide_radius(d, 4, (0,))[0] is False
    assert gm.decide_radius(d, 3, (0, 1))[1] == 0.65  # the smallest distance of the set


def test_graph_excludes_self_and_keeps_unused_refs_as_candidates() -> None:
    # refs at 0, 1, 2; the item at 1.4: nearest others are ref2 (1), ref3 (2), ref1 (0)
    d = _line([0.0, 1.0, 2.0, 1.4, 10.0])
    x = 3
    assert gm.order_by_distance(d, x)[:3] == [1, 2, 0]
    assert gm.decide_graph(d, x, (2,), 1) is False  # the unused ref1 is the nearest
    assert gm.decide_graph(d, x, (1,), 1) is True
    assert gm.anchor_rank(d, x, (0,)) == 3
    assert gm.anchor_rank(d, x, (0, 2)) == 2


def test_tie_order_and_boundary_tie() -> None:
    d = _line([0.0, 1.0, -1.0, 5.0])  # point 0 has 1 and 2 at the same distance
    assert gm.order_by_distance(d, 0)[:2] == [1, 2]
    assert gm.order_by_distance(d, 0, descending_ties=True)[:2] == [2, 1]
    assert gm.boundary_tie(d, 0, 1) is True
    assert gm.boundary_tie(d, 0, 2) is False
    assert gm.decide_graph(d, 0, (1,), 1) is True
    assert gm.decide_graph(d, 0, (1,), 1, descending_ties=True) is False


def test_labelled_majority() -> None:
    d = _line([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 20.0])
    same = [True, True, True, False, False, False, False]
    assert gm.decide_labelled(d, 6, same, 5) is False  # 3,4,5 (the nearest) are not same: 2 of 5
    assert gm.decide_labelled(d, 3, same, 5) is True  # 2,4,1,5,0 -> 2,1,0 same: 3 of 5


def test_anchor_sets_are_the_seven_subsets() -> None:
    assert len(gm.ANCHOR_SETS) == 7
    assert len(set(gm.ANCHOR_SETS)) == 7


def test_kappa_and_fraction() -> None:
    assert ev.cohen_kappa([True, False], [True, False]) == 1.0
    assert ev.cohen_kappa([True, True], [True, True]) is None  # p_e = 1
    assert ev.cohen_kappa([], []) is None
    assert ev.fraction(0, 0)["share"] is None
    assert ev.fraction(1, 4)["fraction"] == "1/4"


def test_role_and_category() -> None:
    assert (category_of("c3-r2"), role_of("c3-r2")) == ("c3", "reversed")
    assert role_of("c1-n1") == "near" and role_of("c1-f4") == "far"


def _csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        w.writerow(["item_id", "L_anchor_answer", "L_rel_answer", "memo"])
        for r in rows:
            w.writerow([*r, ""])


def test_label_loader_with_bom_and_errors(tmp_path: Path) -> None:
    items = ["c1-n1", "c1-f1"]
    good = [("ref1", "same", ""), ("ref2", "same", ""), ("ref3", "same", ""),
            ("c1-n1", "same", "same"), ("c1-f1", "other", "indeterminate")]  # fmt: skip
    p = tmp_path / "a.csv"
    _csv(p, good)
    labels = load_labels(p, items, REFS)
    assert labels.l_anchor["c1-f1"] == "other" and labels.l_rel["ref1"] is None
    counts = labels.counts(items, REFS)
    assert counts["references_are_all_same"] is True
    assert counts["L_anchor_items"] == {"same": 1, "other": 1}
    _csv(p, good[:-1])
    with pytest.raises(LabelError):
        load_labels(p, items, REFS)
    _csv(p, [*good[:-1], ("c1-f1", "bogus", "same")])
    with pytest.raises(LabelError):
        load_labels(p, items, REFS)
    _csv(p, [("ref1", "other", ""), *good[1:]])
    with pytest.raises(LabelError):
        load_labels(p, items, REFS)


def _toy() -> tuple[np.ndarray, list[str], Labels]:
    items = [f"c1-{r}{i}" for r in "nfr" for i in (1, 2)] + ["c2-n1", "c2-n2", "c2-f1", "c2-r1"]
    ids = REFS + items
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(len(ids), 8))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    anchor = {i: "other" for i in ids}
    rel: dict[str, str | None] = {i: "unrelated" for i in ids}
    for r in REFS:
        anchor[r], rel[r] = "same", None
    for i in ("c1-n1", "c1-n2", "c1-f1", "c1-f2"):
        anchor[i], rel[i] = "same", "same"
    for i in ("c1-r1", "c1-r2"):
        anchor[i], rel[i] = "opposite", "opposite"
    for i in ("c2-n1", "c2-n2"):
        rel[i] = "same"
    return pts, ids, Labels(anchor, rel)


def test_evaluate_shapes_and_records() -> None:
    pts, ids, labels = _toy()
    out = ev.evaluate(pts, ids, labels)
    assert out["n_items"] == 10 and len(out["anchor_sets"]) == 7
    for m in ev.METHODS:
        assert len(out["M1"]["per_method"][m]) == 7
        assert len(out["M2"][m]) == 7
        assert out["M3"][m]["n_pairs"] == 21
        assert 0.0 <= out["M3"][m]["min"] <= out["M3"][m]["max"] <= 1.0
    assert len(out["M4"]) == 7
    first = out["M1"]["per_method"]["R"][0]
    assert first["inside_by_L_anchor"]["same"]["of"] == 4
    assert first["inside_by_L_anchor"]["opposite"]["of"] == 2
    assert set(first["cells"]) >= {"c1-near", "c2-reversed"}
    assert first["cells"]["c2-reversed"]["of"] == 1
    assert out["M1"]["n_ambiguous_left_out"] == 0
    # G2 does not depend on the anchor set
    g2 = [e["inside_by_L_anchor"] for e in out["M1"]["per_method"]["G2_k5"]]
    assert all(g == g2[0] for g in g2)
    # M5: the items with L_rel same are 4 (c1) + 2 (c2); candidates are the other 9 items
    assert out["M5"]["n_items"] == 6
    assert out["M5"]["per_item"][0]["expected_for_random"] == pytest.approx(3 / 9)
    assert "human" in out["M5"]["reading"]
    # nothing in the result holds a vector
    assert "position" not in str(out).lower().replace("positions", "")


def test_auc_values_on_separated_groups() -> None:
    # same items sit next to the anchor, opposite items far: AUC (same nearer) = 1
    items = ["c1-n1", "c1-n2", "c1-r1", "c1-r2"]
    ids = REFS + items
    pts = np.zeros((7, 2))
    pts[:3] = [[0, 0], [0.01, 0], [0, 0.01]]
    pts[3:5] = [[0.1, 0.1], [0.1, 0.2]]
    pts[5:] = [[5, 5], [6, 5]]
    anchor = {i: "same" for i in REFS} | {"c1-n1": "same", "c1-n2": "same"}
    anchor |= {"c1-r1": "opposite", "c1-r2": "opposite"}
    rel: dict[str, str | None] = {i: None for i in REFS} | {i: "same" for i in items[:2]}
    rel |= {i: "opposite" for i in items[2:]}
    out = ev.evaluate(pts, ids, Labels(anchor, rel))
    entry = out["M1"]["per_method"]["R"][0]
    assert entry["auc_same_vs_opposite"] == 1.0
    assert entry["auc_same_vs_other"] is None  # no `other` items: a denominator of 0
    assert entry["inside_by_L_anchor"]["other"]["share"] is None
