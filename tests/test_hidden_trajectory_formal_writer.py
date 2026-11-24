# ruff: noqa: F811  (``door`` is the fixture of the integration test file, imported)
"""U6T-2b-b: ``FormalWriter`` and the guarded archive chains (design U6T-2b v0.1, T-T2b-01 to 08).

The unit tests use a fake Formal Store (``FakeStore`` of the save tests, with ``select``) and need
``nvs_edge`` contract v2 for the fake captures. The tests that need hekb-vnext are opt-in
(``HEKB_FORMAL_ROOT``; the real Gate also ``GATE_PY`` and ``SENSOS_TOOLS``).
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from nvs_observation import capture_edge
from test_hidden_trajectory_envelope import Ports
from test_hidden_trajectory_run import clock, pair
from test_hidden_trajectory_save import BUNDLE, FakeStore, Gate, StoreRefusal
from u65_support import make_capture

from nvs_runtime.hidden_trajectory import formal_writer
from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.capture_chain import (
    ChainChangedExternally,
    ChainError,
    ChainFile,
    TornTailError,
)
from nvs_runtime.hidden_trajectory.formal_writer import FormalWriter, WriterError
from nvs_runtime.hidden_trajectory.record_schema import validate_record
from nvs_runtime.integrations import hekb_formal

EDGE, _ = capture_edge.current()
SRC = Path(formal_writer.__file__).resolve().parents[2]
RECORDS = Path("capture") / "records.jsonl"
INDEX = Path("capture") / "raw_index.jsonl"
QDIR = Path("capture") / "quarantine"


class ListStore(FakeStore):
    def select(self, query: Any, limit: int) -> tuple[list[str], bool]:
        return list(self.held), False

    def gate_status(self, artifact_id: str) -> dict[str, Any] | None:
        return {"self": "PASS"} if artifact_id in self.held else None


class World:
    """The fake stores by folder, so that a second open sees what the first wrote."""

    def __init__(self) -> None:
        self.stores: dict[str, ListStore] = {}

    def open(self, root: str, data_dir: str) -> ListStore:
        return self.stores[data_dir]

    def create(self, root: str, data_dir: str, declaration: Any) -> ListStore:
        self.stores[data_dir] = ListStore()
        return self.stores[data_dir]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    made = World()
    monkeypatch.setattr(formal_writer, "_open_store", made.open)
    monkeypatch.setattr(formal_writer, "_create_store", made.create)
    monkeypatch.setattr(hekb_formal, "_claim_checkout", lambda root: str(root))
    monkeypatch.setattr(hekb_formal, "_release_checkout", lambda: None)
    return made


def make(tmp_path: Path, ports: Ports | None = None, **kw: Any) -> FormalWriter:
    ports = ports or Ports()
    return FormalWriter.create(
        tmp_path / "s",
        {"declaration": "formal-store/1"},
        hekb_vnext_root=SRC,
        gate=kw.pop("gate", None) or Gate(),
        space_id_of=ports.space_id_of,
        seal=ports.seal,
        checker_bundle=BUNDLE,
        clock=clock,
    )


def reopen(tmp_path: Path, ports: Ports | None = None, **kw: Any) -> FormalWriter:
    ports = ports or Ports()
    return FormalWriter.open(
        tmp_path / "s",
        hekb_vnext_root=SRC,
        gate=Gate(),
        space_id_of=ports.space_id_of,
        seal=ports.seal,
        checker_bundle=BUNDLE,
        clock=clock,
    )


def save(writer: FormalWriter, ports: Ports, kind: str = "llama", tag: str = "") -> Any:
    return writer.save(
        pair(make_capture(kind=kind)),
        contract_session_id=f"u6t2b-{kind}{tag}",
        synthetic=False,
        ids=ports.ids,
    )


needs_edge = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")


def tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture(autouse=True)
def _own_process_for_hekb(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A checkout of hekb-vnext loaded by an earlier test (the older ``hekb_vnext.py`` path) is
    refused by the writer on purpose; here each test starts as a fresh process would, and the
    earlier modules are put back afterwards."""
    saved = {
        n: m for n, m in sys.modules.items() if n.split(".")[0] in ("graph", "vendor", "storage")
    }
    for name in saved:
        del sys.modules[name]
    monkeypatch.setattr(hekb_formal, "_claims", 0)
    monkeypatch.setattr(hekb_formal, "_claimed_root", None)
    yield
    for name in [n for n in sys.modules if n.split(".")[0] in ("graph", "vendor", "storage")]:
        del sys.modules[name]
    sys.modules.update(saved)


@pytest.fixture(autouse=True)
def _no_writer_left() -> Any:
    yield
    for writer in list(formal_writer._registry.values()):
        writer.close()


# ------------------------------------------------------------------------------------ T-T2b-01, 02
def test_t_t2b_01_a_second_open_of_the_same_folder_is_refused_until_close(
    world: World, tmp_path: Path
) -> None:
    first = make(tmp_path)
    with pytest.raises(WriterError) as raised:
        reopen(tmp_path)
    assert raised.value.code == "writer_already_open"
    with pytest.raises(WriterError) as again:
        make(tmp_path)
    assert again.value.code == "writer_already_open"
    first.close()
    first.close()  # twice is fine
    with reopen(tmp_path) as second:
        assert not second.closed
    with pytest.raises(WriterError) as closed:
        second.get("art:x")
    assert closed.value.code == "writer_closed"


def test_the_writer_is_not_made_by_calling_the_class() -> None:
    with pytest.raises(TypeError):
        FormalWriter()


def test_t_t2b_02_a_symlink_to_the_folder_is_the_same_folder(world: World, tmp_path: Path) -> None:
    writer = make(tmp_path)
    link = tmp_path / "alias"
    link.symlink_to(tmp_path / "s", target_is_directory=True)
    with pytest.raises(WriterError) as raised:
        FormalWriter.open(link, hekb_vnext_root=SRC)
    assert raised.value.code == "writer_already_open"
    writer.close()


def test_a_failed_open_leaves_nothing_registered(world: World, tmp_path: Path) -> None:
    with pytest.raises(KeyError):  # the fake world has no store there: opening fails
        reopen(tmp_path)
    assert not formal_writer._registry
    make(tmp_path).close()


# ------------------------------------------------------------------------------------ T-T2b-03
@needs_edge
def test_t_t2b_03_concurrent_saves_are_one_queue_and_the_chains_do_not_fork(
    world: World, tmp_path: Path
) -> None:
    ports = Ports()

    class SlowGate(Gate):
        def __call__(self, envelope: Any, **kw: Any) -> Any:
            time.sleep(0.002)  # a place for another thread to get in, if the lock did not hold
            return super().__call__(envelope, **kw)

    writer = make(tmp_path, ports, gate=SlowGate())
    results: list[Any] = []
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            results.append(save(writer, ports, "qwen", f"-t{index}"))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(results) == 6
    store = world.stores[str((tmp_path / "s").resolve())]
    # each capture's envelopes are stored in a row: no other capture's put in between
    for result in results:
        seqs = sorted(e["stored"]["hekb_seq"] for e in result.record["positions"].values())
        assert seqs == list(range(seqs[0], seqs[0] + 5)), "puts of two captures were interleaved"
    assert len(store.held) == 30
    writer.close()
    with reopen(tmp_path) as again:  # the records chain verifies and is whole
        records = again.records()
        assert len(records) == 6 and all(validate_record(r) == [] for r in records)
        assert again.unrecorded_envelopes == ()


# ------------------------------------------------------------------------------------ T-T2b-04
@needs_edge
def test_t_t2b_04_a_line_written_by_somebody_else_stops_the_next_save(
    world: World, tmp_path: Path
) -> None:
    ports = Ports()
    writer = make(tmp_path, ports)
    save(writer, ports, "llama", "-a")
    other = ChainFile(tmp_path / "s" / RECORDS)  # a rule-breaker: no writer, no check
    other.append({"intruder": True})
    before = tree(tmp_path / "s")
    held_before = dict(world.stores[str((tmp_path / "s").resolve())].held)
    with pytest.raises(WriterError) as raised:
        save(writer, ports, "llama", "-b")
    assert raised.value.code == "store_changed_externally"
    assert tree(tmp_path / "s") == before, "not a contract, not a RAW file, not a line"
    assert world.stores[str((tmp_path / "s").resolve())].held == held_before
    assert writer.dead
    with pytest.raises(WriterError) as again:  # and it never writes again
        save(writer, ports, "llama", "-c")
    assert again.value.code == "store_changed_externally"
    assert writer.get(next(iter(held_before))) is not None  # reads still work
    writer.close()
    with reopen(tmp_path) as reopened:  # opening again shows what the file holds: the intruder
        assert reopened.records()[-1] == {"intruder": True} and len(reopened.records()) == 2


@needs_edge
def test_the_store_saying_it_was_changed_stops_the_writer_and_the_record_is_still_written(
    world: World, tmp_path: Path
) -> None:
    ports = Ports()
    writer = make(tmp_path, ports)
    store = world.stores[str((tmp_path / "s").resolve())]
    store.refuse = {0: StoreRefusal("store_changed_externally", "another writer")}
    result = save(writer, ports, "llama")
    assert result.record["positions"]["final_norm"]["store_refusal"]["code"] == (
        "store_changed_externally"
    )
    assert len(writer.records()) == 1 and writer.dead
    with pytest.raises(WriterError) as raised:
        save(writer, ports, "llama", "-2")
    assert raised.value.code == "store_changed_externally"


@needs_edge
def test_a_raw_index_changed_from_outside_is_seen_too(world: World, tmp_path: Path) -> None:
    ports = Ports()
    writer = make(tmp_path, ports)
    save(writer, ports, "llama", "-a")
    ChainFile(tmp_path / "s" / INDEX).append({"raw_sha256": "0" * 64, "trajectory_id": "x"})
    with pytest.raises(WriterError) as raised:
        save(writer, ports, "llama", "-b")
    assert raised.value.code == "store_changed_externally"


# ------------------------------------------------------------------------------------ T-T2b-06
def chain_with_tail(path: Path, keep: int = 11) -> tuple[bytes, bytes]:
    chain = ChainFile(path)
    chain.append({"n": 1})
    chain.append({"n": 2})
    data = path.read_bytes()
    cut = data[:-1].rfind(b"\n") + 1
    path.write_bytes(data[: cut + keep])
    return data[:cut], data[cut : cut + keep]


@pytest.mark.parametrize("name", ["records.jsonl", "raw_index.jsonl"])
def test_t_t2b_06_a_torn_last_line_of_an_archive_chain_is_set_aside(
    tmp_path: Path, name: str
) -> None:
    path = tmp_path / "capture" / name
    good, torn = chain_with_tail(path)
    with pytest.raises(ChainError):  # as before without the guard
        ChainFile(path)
    chain = ChainFile(path, guarded=True)
    assert chain.torn is not None and path.read_bytes() == good + torn
    with pytest.raises(TornTailError):
        chain.append({"n": 3})  # not before it is set aside
    record = chain.set_aside_torn_tail()
    assert record is not None and path.read_bytes() == good and len(chain) == 1
    saved = path.parent / record["torn_file"]
    assert saved.read_bytes() == torn and saved.parent == path.parent / "quarantine"
    assert saved.name == f"{name}.{len(good) + len(torn)}.torn"
    assert record["torn_sha256"] == hashlib.sha256(torn).hexdigest()
    assert record["torn_length"] == len(torn) and record["offset"] == len(good)
    assert record["original_length"] == len(good) + len(torn) and record["file"] == name
    assert list(ChainFile(path.parent / "quarantine" / "torn_tails.jsonl").records()) == [record]
    chain.append({"n": 3})
    assert len(ChainFile(path)) == 2


def test_a_line_with_a_newline_that_is_broken_or_a_fork_is_not_touched(tmp_path: Path) -> None:
    path = tmp_path / "capture" / "records.jsonl"
    good, torn = chain_with_tail(path)
    path.write_bytes(good + torn + b"\n")
    before = tree(tmp_path)
    with pytest.raises(ChainError, match="unreadable"):
        ChainFile(path, guarded=True)
    assert tree(tmp_path) == before
    path.write_bytes((good).replace(b'"n":1', b'"n":9') + torn)  # a chain that does not add up
    with pytest.raises(ChainError, match="broken"):
        ChainFile(path, guarded=True)
    # a fork: two lines with the same seq
    one, two = ChainFile(tmp_path / "f1.jsonl"), ChainFile(tmp_path / "f1.jsonl")
    one.append({"a": 1})
    two.append({"b": 2})
    with pytest.raises(ChainError, match="broken"):
        ChainFile(tmp_path / "f1.jsonl", guarded=True)


@pytest.mark.parametrize("step", ["_step_write_evidence", "_step_write_record"])
def test_a_failure_in_step_1_or_2_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    path = tmp_path / "capture" / "records.jsonl"
    good, torn = chain_with_tail(path)
    chain = ChainFile(path, guarded=True)

    def boom(self: ChainFile, *a: Any) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(ChainFile, step, boom)
        with pytest.raises(TornTailError):
            chain.set_aside_torn_tail()
    assert path.read_bytes() == good + torn, "not one byte of the original changed"
    assert chain.set_aside_torn_tail() is not None  # the retry finishes it
    assert path.read_bytes() == good


def test_a_failure_in_step_3_keeps_the_evidence_and_the_retry_writes_one_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture" / "records.jsonl"
    good, torn = chain_with_tail(path)

    def boom(self: ChainFile, *a: Any) -> None:
        raise OSError("read-only")

    with monkeypatch.context() as patch:
        patch.setattr(ChainFile, "_step_cut", boom)
        with pytest.raises(TornTailError):
            ChainFile(path, guarded=True).set_aside_torn_tail()
    assert path.read_bytes() == good + torn
    assert len(list((path.parent / "quarantine").glob("*.torn"))) == 1
    ChainFile(path, guarded=True).set_aside_torn_tail()
    assert path.read_bytes() == good
    assert len(ChainFile(path.parent / "quarantine" / "torn_tails.jsonl")) == 1


def test_the_order_is_evidence_then_record_then_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture" / "records.jsonl"
    good, torn = chain_with_tail(path)
    seen: list[str] = []
    for name in ("_step_write_evidence", "_step_write_record", "_step_cut"):
        original = getattr(ChainFile, name)

        def wrap(self: ChainFile, *a: Any, _o: Any = original, _n: str = name) -> Any:
            if _n == "_step_cut":
                assert path.read_bytes() == good + torn, "whole until the cut"
                assert list((path.parent / "quarantine").glob("*.torn"))
                assert (path.parent / "quarantine" / "torn_tails.jsonl").exists()
            seen.append(_n)
            return _o(self, *a)

        monkeypatch.setattr(ChainFile, name, wrap)
    ChainFile(path, guarded=True).set_aside_torn_tail()
    assert seen == ["_step_write_evidence", "_step_write_record", "_step_cut"]


@needs_edge
def test_the_writer_sets_aside_torn_tails_at_open_and_lists_them(
    world: World, tmp_path: Path
) -> None:
    ports = Ports()
    writer = make(tmp_path, ports)
    save(writer, ports, "llama")
    writer.close()
    path = tmp_path / "s" / RECORDS
    data = path.read_bytes()
    path.write_bytes(data + b'{"seq":1,"prev":"ab')
    with reopen(tmp_path) as again:
        assert path.read_bytes() == data
        [tail] = again.torn_tails()
        assert tail["file"] == "records.jsonl" and tail["torn_length"] == 19
        assert len(again.records()) == 1


def test_the_unguarded_chain_and_archive_are_as_they_were(tmp_path: Path) -> None:
    path = tmp_path / "capture" / "records.jsonl"
    good, torn = chain_with_tail(path)
    with pytest.raises(ChainError):
        CaptureArchive(tmp_path)
    assert path.read_bytes() == good + torn and not (path.parent / "quarantine").exists()
    ChainFile(tmp_path / "x.jsonl").append({"a": 1})
    a, b = ChainFile(tmp_path / "y.jsonl"), ChainFile(tmp_path / "y.jsonl")
    a.append({"a": 1})
    b.append({"b": 1})  # no check without the guard: it forks, as it always could
    with pytest.raises(ChainError):
        ChainFile(tmp_path / "y.jsonl")
    with pytest.raises(ChainChangedExternally):
        g1, g2 = (
            ChainFile(tmp_path / "z.jsonl", guarded=True),
            ChainFile(tmp_path / "z.jsonl", guarded=True),
        )
        g1.append({"a": 1})
        g2.append({"b": 1})


# ------------------------------------------------------------------------------------ T-T2b-07
@needs_edge
def test_t_t2b_07_a_stored_envelope_with_no_record_is_listed_not_recorded(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ports = Ports()
    writer = make(tmp_path, ports)

    def stop(self: CaptureArchive, record: Any) -> Any:
        raise OSError("stopped before the record")

    with monkeypatch.context() as patch:
        patch.setattr(CaptureArchive, "append_record", stop)
        with pytest.raises(OSError, match="stopped"):
            save(writer, ports, "qwen", "-lost")
    save(writer, ports, "llama", "-ok")
    store = world.stores[str((tmp_path / "s").resolve())]
    writer.close()
    records_before = (tmp_path / "s" / RECORDS).read_bytes()
    with reopen(tmp_path) as again:
        lost = again.unrecorded_envelopes
        assert len(lost) == 5 and set(lost) <= set(store.held)
        assert again.find_unrecorded() == list(lost)
        assert len(again.records()) == 1
    assert (tmp_path / "s" / RECORDS).read_bytes() == records_before, "no record was written"


# ------------------------------------------------------------------------------------ T-T2b-08
def test_t_t2b_08_the_save_entries_are_not_public() -> None:
    from nvs_runtime.hidden_trajectory import formal_save

    assert not hasattr(formal_save, "CaptureSaver")
    assert not hasattr(hekb_formal, "create_formal_store")
    assert not hasattr(hekb_formal, "open_formal_store")
    assert not hasattr(hekb_formal, "load_hekb_module")
    allowed = {"formal_writer.py", "formal_save.py", "hekb_formal.py"}
    names = ("_CaptureSaver", "_open_formal_store", "_create_formal_store", "_load_hekb_module")
    for path in SRC.rglob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        assert not [n for n in names if n in text], f"{path} reaches an internal entry"
    # the Formal Store's folder is opened by hekb's IC04Store only by require_formal=True; and
    # only this writer opens it (the others are in test_t_t2b_08_with_hekb below)


# ------------------------------------------------------------------------------------ checkout
def test_a_second_writer_with_another_checkout_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    made = World()
    monkeypatch.setattr(formal_writer, "_open_store", made.open)
    monkeypatch.setattr(formal_writer, "_create_store", made.create)
    monkeypatch.setattr(hekb_formal, "_claims", 0)
    monkeypatch.setattr(hekb_formal, "_claimed_root", None)
    for name in ("graph", "vendor", "storage"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    first = FormalWriter.create(tmp_path / "a", {}, hekb_vnext_root=one)
    with pytest.raises(WriterError) as raised:
        FormalWriter.create(tmp_path / "b", {}, hekb_vnext_root=two)
    assert raised.value.code == "hekb_checkout_conflict"
    third = FormalWriter.create(tmp_path / "c", {}, hekb_vnext_root=one)  # the same one is fine
    third.close()
    first.close()
    FormalWriter.create(tmp_path / "d", {}, hekb_vnext_root=two).close()  # all closed: free again


def test_a_graph_loaded_from_another_checkout_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import types

    made = World()
    monkeypatch.setattr(formal_writer, "_open_store", made.open)
    monkeypatch.setattr(formal_writer, "_create_store", made.create)
    monkeypatch.setattr(hekb_formal, "_claims", 0)
    monkeypatch.setattr(hekb_formal, "_claimed_root", None)
    other, mine = tmp_path / "other", tmp_path / "mine"
    other.mkdir()
    mine.mkdir()
    for name in ("vendor", "storage"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    fake = types.ModuleType("storage")
    fake.__file__ = str(other / "storage" / "__init__.py")
    monkeypatch.setitem(sys.modules, "storage", fake)  # what the older hekb_vnext.py leaves behind
    with pytest.raises(WriterError) as raised:
        FormalWriter.create(tmp_path / "a", {}, hekb_vnext_root=mine)
    assert raised.value.code == "hekb_checkout_conflict"
    assert not formal_writer._registry


# ------------------------------------------------------------------------------------ with hekb
HEKB = __import__("os").environ.get("HEKB_FORMAL_ROOT")
needs_hekb = pytest.mark.skipif(
    not (HEKB and (Path(HEKB) / "graph" / "formal.py").is_file()),
    reason="HEKB_FORMAL_ROOT is not set (a hekb-vnext checkout with the Formal Store)",
)


@needs_hekb
def test_t_t2b_08_with_hekb_the_folder_is_not_opened_around_the_writer(
    tmp_path: Path,
) -> None:
    assert HEKB is not None
    formal = hekb_formal._load_hekb_module(HEKB, "graph.formal")
    declaration = formal.formal_declaration(
        catalog_rows=[["msr.trajectory-window", "window_rows", "raw_not_in_artifact"]],
        catalog_version="test",
        created_by="u6t2b-b",
        allow_unverified=[],
    )
    with FormalWriter.create(tmp_path / "s", declaration, hekb_vnext_root=HEKB):
        with pytest.raises(WriterError) as raised:
            FormalWriter.open(tmp_path / "s", hekb_vnext_root=HEKB)
        assert raised.value.code == "writer_already_open"
    module = hekb_formal._load_hekb_module(HEKB, "graph.ic04")
    with pytest.raises(module.IC04Error) as old:  # an older way in: no require_formal
        module.IC04Store(tmp_path / "s")
    assert old.value.code == "formal_store_needs_require_formal"
    with FormalWriter.open(tmp_path / "s", hekb_vnext_root=HEKB) as writer:
        assert writer.stored_artifact_ids() == [] and writer.unrecorded_envelopes == ()


@needs_hekb
def test_a_second_checkout_is_refused_with_the_real_checkout(tmp_path: Path) -> None:
    assert HEKB is not None
    other = tmp_path / "other_checkout"
    other.mkdir()
    formal = hekb_formal._load_hekb_module(HEKB, "graph.formal")
    declaration = formal.formal_declaration(
        catalog_rows=[["msr.trajectory-window", "window_rows", "raw_not_in_artifact"]],
        catalog_version="test",
        created_by="u6t2b-b",
        allow_unverified=[],
    )
    with FormalWriter.create(tmp_path / "s", declaration, hekb_vnext_root=HEKB):
        with pytest.raises(WriterError) as raised:
            FormalWriter.create(tmp_path / "t", declaration, hekb_vnext_root=other)
        assert raised.value.code == "hekb_checkout_conflict"
        assert not (tmp_path / "t").exists(), "nothing was made for the refused writer"


# the real Gate and a real Formal Store (also GATE_PY and SENSOS_TOOLS)
from test_hidden_trajectory_save_integration import (  # noqa: E402
    _declaration,
    _ids,
    door,  # noqa: F401 (a fixture)
    needs_gate,
)


def _real_writer(tmp_path: Path, door: Any, *, create: bool) -> FormalWriter:
    assert HEKB is not None
    common: dict[str, Any] = {
        "hekb_vnext_root": HEKB,
        "gate": door.verdict,
        "space_id_of": door.space_id_of,
        "seal": door.seal,
        "checker_bundle": door.client.checker_bundle,
        "clock": clock,
    }
    if create:
        return FormalWriter.create(tmp_path / "s", _declaration(door), **common)
    return FormalWriter.open(tmp_path / "s", **common)


def _real_save(writer: FormalWriter, door: Any, kind: str) -> Any:
    return writer.save(
        pair(make_capture(kind=kind)),
        contract_session_id=f"u6t2b-b-{kind}",
        synthetic=False,
        ids=_ids(door, f"w-{kind}"),
    )


@needs_gate
@pytest.mark.parametrize(("kind", "count"), [("llama", 1), ("qwen", 5)])
def test_saved_through_the_writer_and_read_back_after_opening_again(
    tmp_path: Path, door: Any, kind: str, count: int
) -> None:
    with _real_writer(tmp_path, door, create=True) as writer:
        result = _real_save(writer, door, kind)
        assert result.record["outcome"] == "enveloped" and len(result.receipts) == count
        assert writer.unrecorded_envelopes == () and writer.find_unrecorded() == []
    with _real_writer(tmp_path, door, create=False) as again:
        assert again.records() == [result.record]
        assert again.unrecorded_envelopes == () and again.torn_tails() == []
        assert sorted(again.stored_artifact_ids()) == sorted(
            e["artifact_id"] for e in result.envelopes.values()
        )
        for point, entry in result.record["positions"].items():
            got = again.get(entry["artifact_id"])
            assert got is not None and got["envelope"] == result.envelopes[point]
            assert entry["stored"]["hekb_seq"] == got["hekb_seq"]
            assert entry["stored"]["hekb_hash"] == got["hekb_hash"]
            status = again.gate_status(entry["artifact_id"])
            assert status is not None and status["self"] in ("PASS", "PASS_UNVERIFIED")


@needs_gate
def test_an_outside_line_in_the_formal_store_chain_stops_the_writer(
    tmp_path: Path, door: Any
) -> None:
    assert HEKB is not None
    with _real_writer(tmp_path, door, create=True) as writer:
        _real_save(writer, door, "llama")
        chainlog = hekb_formal._load_hekb_module(HEKB, "graph.chainlog")
        path = tmp_path / "s" / "graph" / "ic00_envelopes.jsonl"
        chainlog.ChainLog(path).append({"intruder": True})
        before = tree(tmp_path / "s")
        result = _real_save(
            writer, door, "qwen"
        )  # the Gate and the archive are fine, the put is not
        codes = {e["store_refusal"]["code"] for e in result.record["positions"].values()}
        assert codes == {"store_changed_externally"} and not result.receipts
        assert len(writer.records()) == 2, "the record of that capture was still written"
        assert writer.dead
        assert (tmp_path / "s" / "graph" / "ic00_envelopes.jsonl").read_bytes() == before[
            "graph/ic00_envelopes.jsonl"
        ]
        with pytest.raises(WriterError):
            _real_save(writer, door, "llama")
