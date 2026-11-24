"""The adopted labels of the Phase 2 evaluation (the CSV of the label adoption record).

Columns used: ``item_id``, ``L_anchor_answer`` (same / opposite / other / ambiguous) and
``L_rel_answer`` (same / opposite / unrelated / indeterminate; blank for a reference). The other
columns are never read into a result. The file is UTF-8, with or without a byte order mark.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

L_ANCHOR_VALUES: Final = ("same", "opposite", "other", "ambiguous")
L_REL_VALUES: Final = ("same", "opposite", "unrelated", "indeterminate")
ROLE_OF_LETTER: Final = {"n": "near", "f": "far", "r": "reversed"}


class LabelError(ValueError):
    """The label file is not what the evaluation needs."""


@dataclass(frozen=True)
class Labels:
    l_anchor: dict[str, str]
    l_rel: dict[str, str | None]  # None for a reference

    def counts(self, items: Sequence[str], references: Sequence[str]) -> dict[str, object]:
        return {
            "n_items": len(items),
            "n_references": len(references),
            "L_anchor_items": dict(Counter(self.l_anchor[i] for i in items)),
            "L_rel_items": dict(Counter(str(self.l_rel[i]) for i in items)),
            "references_L_anchor": dict(Counter(self.l_anchor[i] for i in references)),
            "references_are_all_same": all(self.l_anchor[i] == "same" for i in references),
        }


def category_of(item_id: str) -> str:
    return item_id.split("-")[0]


def role_of(item_id: str) -> str:
    """near / far / reversed, from the letter after the hyphen of the item id (c3-r2: reversed)."""
    return ROLE_OF_LETTER[item_id.split("-")[1][0]]


def load_labels(path: str | Path, items: Sequence[str], references: Sequence[str]) -> Labels:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = set(items) | set(references)
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise LabelError("the item ids of the label file differ from the fixed set")
    l_anchor: dict[str, str] = {}
    l_rel: dict[str, str | None] = {}
    for r in rows:
        item_id, anchor, rel = r["item_id"], r["L_anchor_answer"].strip(), r["L_rel_answer"].strip()
        if anchor not in L_ANCHOR_VALUES:
            raise LabelError(f"{item_id}: L_anchor {anchor!r} is not a registered value")
        if item_id in references:
            if anchor != "same" or rel:
                raise LabelError(f"{item_id}: a reference is `same` by definition, L_rel blank")
            l_anchor[item_id], l_rel[item_id] = "same", None
            continue
        if rel not in L_REL_VALUES:
            raise LabelError(f"{item_id}: L_rel {rel!r} is not a registered value")
        l_anchor[item_id], l_rel[item_id] = anchor, rel
    return Labels(l_anchor, l_rel)
