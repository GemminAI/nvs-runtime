"""RSA-X2: the fixed item set, the position read-out by item_id and the integrity numbers, with fake
captures saved without a K (as in ``test_hidden_trajectory_item_set``). What reads a K measured by
the real kernel is tested in the Developer Tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from nvs_observation import capture_edge
from test_hidden_trajectory_item_set import fake_capture_fn, ids_for, make_saver

from nvs_runtime.hidden_trajectory import item_set as isx
from nvs_runtime.hidden_trajectory.rsa import readout as ro

EDGE, _ = capture_edge.current()
needs_edge = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")


def write_fixed(path: Path, n_items: int = 72, n_refs: int = 3) -> str:
    kinds = {"n": "near_paraphrase", "f": "far_paraphrase", "r": "reversed_meaning"}
    items = [
        {
            "item_id": f"c{c}-{k}{i}",
            "text": f"Item {c} {k} {i} words.",
            "role": role,
            "category": f"C{c}",
        }
        for c in range(1, 7)
        for k, role in kinds.items()
        for i in range(1, 5)
    ][:n_items]
    refs = [
        {
            "item_id": f"ref{i}",
            "text": f"Reference sentence {i}.",
            "role": "reference",
            "category": "C1",
        }
        for i in range(1, n_refs + 1)
    ]
    path.write_text(
        json.dumps({"set_id": "t", "items": items, "references": refs}), encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_registered_hash_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "fixed.json"
    sha = write_fixed(path)
    got = ro.load_fixed_set(path, expected_sha256=sha)
    assert len(got.item_ids) == 72 and len(got.reference_ids) == 3 and got.sha256 == sha
    assert got.all_ids == got.item_ids + got.reference_ids
    with pytest.raises(ro.ReadoutError, match="hashes to"):
        ro.load_fixed_set(path)  # the registered hash of the real set is not this file's
    path.write_text(path.read_text().replace("Item 1 n 1", "Item X"), encoding="utf-8")
    with pytest.raises(ro.ReadoutError):
        ro.load_fixed_set(path, expected_sha256=sha)


def test_the_counts_are_checked(tmp_path: Path) -> None:
    path = tmp_path / "fixed.json"
    sha = write_fixed(path, n_items=70)
    with pytest.raises(ro.ReadoutError, match="expected 72 items"):
        ro.load_fixed_set(path, expected_sha256=sha)


def captured(tmp_path: Path, texts: dict[str, str]) -> tuple[Path, list[isx.ItemRun]]:
    items = tuple(isx.Item(i, t, "smoke") for i, t in texts.items())
    item_set = isx.ItemSet("fake", items, "0" * 64)
    saver, ports = make_saver(tmp_path)
    rows = isx.run_item_set(item_set, saver, fake_capture_fn(), ids_for(ports))
    return tmp_path / "store", rows


TEXTS = {"a1": "alpha beta.", "a2": "gamma delta.", "r1": "reference one."}


@needs_edge
def test_a_missing_and_an_extra_item_id_are_found(tmp_path: Path) -> None:
    store, _ = captured(tmp_path, TEXTS)
    positions, notes = ro.read_positions(store)
    report = ro.integrity_report(positions, ["a1", "a2", "zz"], notes=notes)
    assert report["missing_item_ids"] == ["zz"] and report["extra_item_ids"] == ["r1"]
    assert not report["item_ids_match_exactly"] and not report["ok"]
    with pytest.raises(ro.ReadoutError):
        ro.position_array(positions, ["zz"])


@needs_edge
def test_a_position_that_was_not_measured_is_counted_not_hidden(tmp_path: Path) -> None:
    from test_hidden_trajectory_item_set import fake_capture_fn as fc

    items = (isx.Item("q1", "text one.", "smoke"),)
    saver, ports = make_saver(tmp_path)
    isx.run_item_set(
        isx.ItemSet("f", items, "0" * 64), saver, fc("qwen", dimension=8), ids_for(ports)
    )
    positions, notes = ro.read_positions(tmp_path / "store")
    report = ro.integrity_report(positions, ["q1"], notes=notes)
    assert (
        report["status_counts"] == {isx.KPOS_NOT_K: 1}
        and report["n_measured"] == 0
        and not report["ok"]
    )


def test_the_integrity_report_sees_a_bad_vector() -> None:
    good = isx.KPosition("o1", "measured", position=tuple([1.0] + [0.0] * 63), n_cycles=1)
    bad_nan = isx.KPosition(
        "o2", "measured", position=tuple([float("nan")] + [0.0] * 63), n_cycles=1
    )
    short = isx.KPosition("o3", "measured", position=(1.0, 0.0), n_cycles=1)
    scaled = isx.KPosition("o4", "measured", position=tuple([2.0] + [0.0] * 63), n_cycles=1)
    ids = ["g", "n", "s", "x"]
    out = ro.integrity_report({"g": good, "n": bad_nan, "s": short, "x": scaled}, ids)
    assert not out["all_finite"] and out["dimensions"] == [2, 64] and not out["ok"]
    assert out["norm_max"] == 2.0 and not out["unit_norm_within_tolerance"]


def test_string_positions_call_the_projector_and_keep_the_order() -> None:
    calls: list[str] = []

    def project(text: str) -> Any:
        calls.append(text)
        v = np.zeros(64)
        v[len(text) % 64] = 1.0
        return v

    out = ro.string_positions({"a": "x", "b": "yy"}, ["b", "a"], project)
    assert calls == ["yy", "x"] and out.shape == (2, 64) and out[0, 2] == 1.0 and out[1, 1] == 1.0
