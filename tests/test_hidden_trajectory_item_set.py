"""RSA-X1: the item-set entry and the read-back of K positions, with fake captures, the real
in-process kernel of ``nvs_edge`` (a 4096-D Llama-shaped capture goes through ``tf-1a``), a fake
Formal Store and a stub Gate. The real model is not used here (see
``test_hidden_trajectory_real_machine_item_set.py``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from nvs_observation import capture_edge
from test_hidden_trajectory_envelope import Ports
from test_hidden_trajectory_run import clock, pair
from test_hidden_trajectory_save import BUNDLE, FakeStore, Gate
from u65_support import make_capture

from nvs_runtime.hidden_trajectory import item_set as isx
from nvs_runtime.hidden_trajectory import k_measure, record
from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.envelope import EnvelopeIds
from nvs_runtime.hidden_trajectory.formal_save import _CaptureSaver

EDGE, _ = capture_edge.current()
needs_edge = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")

TEXTS = {
    "a1": ("The cat sat on the mat.", "plain"),
    "a2": ("A feline rested upon the rug.", "plain"),
    "a3": ("Quarterly revenue rose by four percent.", "other"),
}


def write_set(path: Path, items: dict[str, tuple[str, str]] | None = None, **extra: Any) -> Path:
    items = items or TEXTS
    doc = {
        "version": isx.SET_VERSION,
        "set_id": "fake-set",
        "items": [{"item_id": k, "text": t, "role": r} for k, (t, r) in items.items()],
        **extra,
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def make_saver(tmp_path: Path, *, k_path: bool = False) -> tuple[_CaptureSaver, Ports]:
    """A saver of item captures. By default it makes no K (``k_path`` off): what these tests are
    about does not need one; K measured by the real kernel is tested in the Developer Tools."""
    ports = Ports()
    saver = _CaptureSaver(
        store=FakeStore(),
        archive=CaptureArchive(tmp_path / "store"),
        gate=Gate(),
        space_id_of=ports.space_id_of,
        seal=ports.seal,
        checker_bundle=BUNDLE,
        clock=clock,
        k_path=k_path,
    )
    return saver, ports


def ids_for(ports: Ports) -> Any:
    def make(item: isx.Item) -> EnvelopeIds:
        session = "ses:" + hashlib.sha256(item.item_id.encode()).hexdigest()[:16]
        observation = "obs:" + hashlib.sha256(session.encode()).hexdigest()[:16]
        return EnvelopeIds(session, observation, "2026-11-18T00:00:00Z", ports.next_seq)

    return make


def fake_capture_fn(kind: str = "llama", n_gen: int = 1, dimension: int = 4096) -> Any:
    """Each call is a capture whose single row depends on the text (so positions differ)."""

    def capture(messages: list[dict[str, str]]) -> Any:
        text = messages[-1]["content"]
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        base = make_capture(kind=kind, n_gen=n_gen, dimension=dimension)
        rng = np.random.default_rng(seed)
        states = {
            p: np.asarray(rng.standard_normal((n_gen, dimension)), dtype=np.float32)
            for p in base.points
        }
        for value in states.values():
            value.setflags(write=False)
        return pair(
            make_capture(
                kind=kind, n_gen=n_gen, dimension=dimension, states=MappingProxyType(states)
            )
        )

    return capture


# ---- the set file ----------------------------------------------------------------------------


def test_the_set_file_is_read_and_its_hash_is_that_of_its_bytes(tmp_path: Path) -> None:
    path = write_set(tmp_path / "set.json")
    loaded = isx.load_item_set(path)
    assert [i.item_id for i in loaded.items] == ["a1", "a2", "a3"]
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.items[0].text_sha256 == hashlib.sha256(TEXTS["a1"][0].encode()).hexdigest()


def test_a_changed_set_file_is_found_by_its_hash(tmp_path: Path) -> None:
    path = write_set(tmp_path / "set.json")
    recorded = isx.load_item_set(path).sha256
    write_set(path, {**TEXTS, "a3": ("Quarterly revenue fell.", "other")})
    with pytest.raises(isx.ItemSetError, match="hashes to"):
        isx.load_item_set(path, expected_sha256=recorded)
    assert isx.load_item_set(path).sha256 != recorded


@pytest.mark.parametrize(
    "doc",
    [
        {"version": "x", "set_id": "s", "items": []},
        {"version": isx.SET_VERSION, "set_id": "s", "items": []},
        {"version": isx.SET_VERSION, "set_id": "s", "items": [{"item_id": "a", "text": "t"}]},
        {
            "version": isx.SET_VERSION,
            "set_id": "s",
            "items": [
                {"item_id": "a", "text": "t", "role": "r"},
                {"item_id": "a", "text": "u", "role": "r"},
            ],
        },
        {
            "version": isx.SET_VERSION,
            "set_id": "s",
            "items": [{"item_id": "a", "text": "", "role": "r"}],
        },
        {"version": isx.SET_VERSION, "set_id": "s", "items": [], "extra": 1},
    ],
)
def test_a_bad_set_file_is_refused(tmp_path: Path, doc: dict[str, Any]) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(isx.ItemSetError):
        isx.load_item_set(path)


def test_every_item_gets_the_same_message_shape(tmp_path: Path) -> None:
    loaded = isx.load_item_set(write_set(tmp_path / "s.json", system="Be brief."))
    shapes = {tuple(m["role"] for m in isx.messages_for(i, loaded.system)) for i in loaded.items}
    assert shapes == {("system", "user")}
    plain = isx.load_item_set(write_set(tmp_path / "t.json"))
    assert isx.messages_for(plain.items[0], plain.system) == [
        {"role": "user", "content": TEXTS["a1"][0]}
    ]


# ---- captures and the mapping table ----------------------------------------------------------


@needs_edge
def test_the_text_of_an_item_is_in_no_file_of_the_store(tmp_path: Path) -> None:
    loaded = isx.load_item_set(write_set(tmp_path / "set.json"))
    saver, ports = make_saver(tmp_path)
    isx.run_item_set(loaded, saver, fake_capture_fn(), ids_for(ports))
    for path in (tmp_path / "store").rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert all(text.encode() not in data for text, _ in TEXTS.values())


@needs_edge
def test_a_failing_item_is_a_row_and_the_next_item_goes_on(tmp_path: Path) -> None:
    loaded = isx.load_item_set(write_set(tmp_path / "set.json"))
    saver, ports = make_saver(tmp_path)
    good = fake_capture_fn()

    def flaky(messages: list[dict[str, str]]) -> Any:
        if messages[-1]["content"] == TEXTS["a2"][0]:
            raise RuntimeError("secret detail that must not be kept")
        return good(messages)

    rows = isx.run_item_set(loaded, saver, flaky, ids_for(ports))
    assert [r.outcome for r in rows] == [isx.OUT_SAVED, isx.OUT_CAPTURE_ERROR, isx.OUT_SAVED]
    assert rows[1].error == "RuntimeError"
    assert "secret" not in json.dumps([r.as_row() for r in rows])


@needs_edge
def test_an_empty_generation_is_a_row_without_a_position(tmp_path: Path) -> None:
    loaded = isx.load_item_set(write_set(tmp_path / "set.json", {"a1": TEXTS["a1"]}))
    saver, ports = make_saver(tmp_path)

    def empty(messages: list[dict[str, str]]) -> Any:
        return pair(make_capture(kind="llama", n_gen=0, dimension=4096))

    rows = isx.run_item_set(loaded, saver, empty, ids_for(ports))
    assert rows[0].record_outcome == "empty_generation" and rows[0].k_status is None
    table = isx.mapping_table(loaded, rows, tmp_path / "store")
    assert table["items"][0]["k_position"]["status"] in (isx.KPOS_NO_POSITION, isx.KPOS_NOT_K)


@needs_edge
def test_without_a_k_measurer_a_position_cannot_be_given_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no Developer Tools installed there is no K: every row says ``not_measured`` and the
    table says why (the position of a measured K is tested in the Developer Tools)."""
    monkeypatch.setattr(k_measure.metadata, "entry_points", lambda *, group: [])
    loaded = isx.load_item_set(write_set(tmp_path / "set.json"))
    saver, ports = make_saver(tmp_path, k_path=True)
    assert isx.k_position(tmp_path / "store", "obs:none").status == isx.KPOS_NO_RECORD
    rows = isx.run_item_set(loaded, saver, fake_capture_fn(), ids_for(ports))
    assert all(r.outcome == isx.OUT_SAVED and r.record_outcome == "enveloped" for r in rows)
    assert all(r.k_status == record.K_NOT_MEASURED for r in rows)
    table = isx.mapping_table(loaded, rows, tmp_path / "store")
    assert table["n_items"] == 3
    for entry in table["items"]:
        k = entry["k_position"]
        assert k["status"] == isx.KPOS_NOT_K and k["position"] is None
        assert k["reason"] == record.REASON_K_EDGE_UNAVAILABLE


# ---- the template condition ------------------------------------------------------------------


def test_template_condition_from_token_ids() -> None:
    same = isx.template_condition({"a": [1, 2, 5, 6, 9], "b": [1, 2, 7, 8, 6, 9]})
    assert same["all_items_end_with_the_same_token"] and same["last_token_ids"] == [9]
    assert same["common_prefix_tokens"] == 2 and same["common_suffix_tokens"] == 2
    other = isx.template_condition({"a": [1, 2, 9], "b": [1, 2, 8]})
    assert not other["all_items_end_with_the_same_token"] and other["common_suffix_tokens"] == 0
    with pytest.raises(ValueError):
        isx.template_condition({})
