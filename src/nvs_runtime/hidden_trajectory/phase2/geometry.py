"""Radius (R), neighbour-graph (G1) and labelled (G2) decisions on the 75 points.

The points are ordered: references ref1..ref3 (indices 0, 1, 2), then the 72 items in the order of
the item set (indices 3..74). Distance: Euclidean on the 64-D unit vectors (``rsa.compare``'s
``pairwise_euclidean``, the one X-4 used). A tie between equal distances is broken by the index in
ascending order; ``descending_ties`` breaks it in descending order, which is how the sensitivity to
the tie rule is measured.
"""

from __future__ import annotations

from itertools import combinations
from typing import Final

import numpy as np
from numpy.typing import NDArray

RADIUS: Final = 0.65
N_REFERENCES: Final = 3
ANCHOR_SETS: Final[tuple[tuple[int, ...], ...]] = tuple(
    c for size in (1, 2, 3) for c in combinations(range(N_REFERENCES), size)
)
K_MAIN: Final = 5
K_SENSITIVITY: Final = (3, 7)

Array = NDArray[np.float64]


def order_by_distance(distances: Array, x: int, *, descending_ties: bool = False) -> list[int]:
    """The other points, nearest first (self excluded); equal distances by index (ascending, or
    descending when ``descending_ties``)."""
    others = [j for j in range(distances.shape[0]) if j != x]
    sign = -1 if descending_ties else 1
    return sorted(others, key=lambda j: (float(distances[x, j]), sign * j))


def decide_radius(distances: Array, x: int, anchors: tuple[int, ...]) -> tuple[bool, float]:
    """(inside, d): d is the smallest distance to an anchor; inside when d <= RADIUS."""
    d = min(float(distances[x, a]) for a in anchors)
    return d <= RADIUS, d


def decide_graph(
    distances: Array, x: int, anchors: tuple[int, ...], k: int, *, descending_ties: bool = False
) -> bool:
    """Inside when one of the ``k`` nearest of the other points is an anchor of the set."""
    nearest = order_by_distance(distances, x, descending_ties=descending_ties)[:k]
    return any(a in nearest for a in anchors)


def anchor_rank(
    distances: Array, x: int, anchors: tuple[int, ...], *, descending_ties: bool = False
) -> int:
    """The 1-based position of the earliest anchor of the set among the other points, nearest
    first (smaller means more inside)."""
    order = order_by_distance(distances, x, descending_ties=descending_ties)
    return 1 + min(order.index(a) for a in anchors)


def decide_labelled(
    distances: Array,
    x: int,
    same: list[bool],
    k: int = K_MAIN,
    *,
    descending_ties: bool = False,
) -> bool:
    """Inside when most of the ``k`` nearest others have L_anchor `same` (the references are `same`,
    the unused ones included)."""
    nearest = order_by_distance(distances, x, descending_ties=descending_ties)[:k]
    return sum(1 for j in nearest if same[j]) * 2 > k


def boundary_tie(distances: Array, x: int, k: int) -> bool:
    """True when the k-th and the (k+1)-th nearest are at exactly the same distance (a tie that
    the tie rule decides)."""
    order = order_by_distance(distances, x)
    if k >= len(order):
        return False
    return float(distances[x, order[k - 1]]) == float(distances[x, order[k]])
