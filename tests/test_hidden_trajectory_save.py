"""U6T-2b': the capture archive, the save order and R-9 (design U6T-2 v0.1, 6-8; T-T2-14, T-T2-15).

The Formal Store is a fake here (``FakeStore``: the ``put`` / ``get`` the save calls); the Gate is a
stub. The real Gate and a real
Formal Store of hekb-vnext are in ``test_hidden_trajectory_save_integration.py`` (opt-in). The
captures are the fake ones of U6-5d
(``make_capture``), which need ``nvs_edge`` contract v2 for the RAW files: without it every test
here is skipped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from nvs_observation import capture_edge
from test_hidden_trajectory_envelope import Ports
from test_hidden_trajectory_run import clock, pair, stub_verdict
from u65_support import make_capture

from nvs_runtime.hidden_trajectory import record as rec
from nvs_runtime.hidden_trajectory.capture_archive import ArchiveError, CaptureArchive
from nvs_runtime.hidden_trajectory.capture_chain import ChainError, ChainFile
from nvs_runtime.hidden_trajectory.envelope import ParentRef
from nvs_runtime.hidden_trajectory.formal_save import (
    SaveResult,
)
from nvs_runtime.hidden_trajectory.formal_save import (
    _CaptureSaver as CaptureSaver,
)
from nvs_runtime.hidden_trajectory.gate_client import (
    GateUnavailable,
    ParentEnvelopeMissing,
    ParentMismatch,
)
from nvs_runtime.hidden_trajectory.record_schema import validate_record

EDGE, _ = capture_edge.current()
pytestmark = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")

BUNDLE = b"a checker bundle"
PARENT_ARTIFACT = "art:" + "7" * 64


class StoreRefusal(Exception):
    """What hekb-vnext's ``IC04Error`` looks like to the save: it has a ``code``."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


class FakeStore:
    """``put`` and ``get`` of a Formal Store, in memory. ``refuse`` maps an index (0-based, in the
    order of ``put``) to what to raise."""

    def __init__(self, refuse: dict[int, BaseException] | None = None) -> None:
        self.refuse = refuse or {}
        self.calls: list[dict[str, Any]] = []
        self.held: dict[str, dict[str, Any]] = {}

    def put(
        self,
        envelope: dict[str, Any],
        blobs: Any = None,
        *,
        gate_verdict: Any,
        checker_bundle: Any = None,
    ) -> dict[str, Any]:
        index = len(self.calls)
        self.calls.append(
            {
                "envelope": envelope,
                "blobs": dict(blobs or {}),
                "verdict": gate_verdict,
                "bundle": checker_bundle,
            }
        )
        if index in self.refuse:
            raise self.refuse[index]
        seq = len(self.held) + 1
        self.held[envelope["artifact_id"]] = {"envelope": envelope}
        return {
            "artifact_id": envelope["artifact_id"],
            "content_id": envelope["content_id"],
            "hekb_seq": seq,
            "hekb_hash": f"{seq:064x}",
            "already_stored": False,
            "held": [],
        }

    def get(self, artifact_id: str) -> dict[str, Any] | None:
        return self.held.get(artifact_id)


def verdict_for(envelope: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return stub_verdict(
        verdict_sha256=hashlib.sha256(envelope["artifact_id"].encode()).hexdigest(), **changes
    )


class Gate:
    def __init__(self, fail: BaseException | None = None) -> None:
        self.fail, self.calls = fail, []

    def __call__(
        self, envelope: Any, *, lookup: Any, stored: Any, blobs: Any, contract_facts: Any
    ) -> dict[str, Any]:
        self.calls.append({"lookup": lookup, "stored": stored})
        if self.fail is not None:
            raise self.fail
        return verdict_for(envelope)


def saver(tmp_path: Path, **kw: Any) -> tuple[CaptureSaver, FakeStore, CaptureArchive, Ports, Gate]:
    ports = Ports()
    store, archive = kw.pop("store", None) or FakeStore(), CaptureArchive(tmp_path)
    gate = kw.pop("gate", None) or Gate()
    made = CaptureSaver(
        store=store,
        archive=archive,
        gate=gate,
        space_id_of=kw.pop("space_id_of", ports.space_id_of),
        seal=kw.pop("seal", ports.seal),
        checker_bundle=kw.pop("checker_bundle", BUNDLE),
        clock=clock,
    )
    assert not kw
    return made, store, archive, ports, gate


def save(made: CaptureSaver, ports: Ports, kind: str = "llama", **kw: Any) -> SaveResult:
    kw.setdefault("contract_session_id", f"u6t2b-{kind}")
    return made.save(pair(make_capture(kind=kind)), synthetic=False, ids=ports.ids, **kw)


def assert_valid(result: SaveResult) -> None:
    assert validate_record(result.record) == []
    assert (
        list(result.line["record"]) == list(result.record) or result.line["record"] == result.record
    )


# ---- the order: archive, envelopes, Gate, store, record last -------------------------------------


@pytest.mark.parametrize(("kind", "count"), [("llama", 1), ("qwen", 5)])
def test_a_capture_is_archived_gated_stored_and_its_record_is_last(
    tmp_path: Path, kind: str, count: int
) -> None:
    made, store, archive, ports, _gate = saver(tmp_path)
    result = save(made, ports, kind)
    assert result.stage == "built" and len(result.receipts) == count and len(store.calls) == count
    assert_valid(result)
    record = result.record
    assert record["outcome"] == "enveloped" and len(archive) == 1
    for point, entry in record["positions"].items():
        receipt = result.receipts[point]
        assert entry["stored"] == {
            "artifact_id": receipt["artifact_id"],
            "hekb_seq": receipt["hekb_seq"],
            "hekb_hash": receipt["hekb_hash"],
        }
        assert entry["verdict_sha256"] == result.verdicts[point]["verdict_sha256"]
        assert entry["store_refusal"] is None and entry["artifact_id"] == receipt["artifact_id"]
        assert entry["gate"]["status"] == "PASS"
    # what the record names is there: the contract and every RAW file, by hash and by trajectory_id
    digest = result.archived["contract_sha256"]
    assert archive.read_contract(digest) == result.contract
    for point, info in result.archived["raw"].items():
        trajectory = next(
            p["trajectory_id"] for p in result.contract["positions"] if p["point"] == point
        )  # type: ignore[index]
        raw = archive.read_raw(info["sha256"])
        assert raw is not None and raw == archive.raw_for_trajectory(trajectory) and info["written"]
    for (
        call
    ) in store.calls:  # the store got the verdict, the bundle, and only the Blobs of that envelope
        assert call["bundle"] == BUNDLE and call["verdict"]["result"] == "PASS"
        payload = call["envelope"]["payload"]
        assert set(call["blobs"]) == {payload["points"]["uri"], payload["token_index"]["uri"]}
    assert list(archive.records()) == [record]


def test_the_record_is_the_last_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    made, store, archive, ports, gate = saver(tmp_path)
    original_put, original_append = store.put, archive.append_record
    store.put = lambda *a, **k: (order.append("put"), original_put(*a, **k))[1]  # type: ignore[method-assign]
    archive.append_record = lambda r: (order.append("record"), original_append(r))[1]  # type: ignore[method-assign]
    original_write = archive.write_raw
    archive.write_raw = lambda *a, **k: (order.append("archive"), original_write(*a, **k))[1]  # type: ignore[method-assign]
    original_gate = made._gate
    made._gate = lambda *a, **k: (order.append("gate"), original_gate(*a, **k))[1]
    save(made, ports, "qwen")
    assert order == ["archive", *["gate", "put"] * 5, "record"]


# ---- T-T2-14: the archive ------------------------------------------------------------------------


def test_t_t2_14_the_same_contract_and_raw_are_not_written_again(tmp_path: Path) -> None:
    made, _store, archive, ports, _gate = saver(tmp_path)
    first = save(made, ports, "qwen")
    files = {
        p: p.stat().st_mtime_ns
        for p in (tmp_path / "capture").rglob("*")
        if p.is_file() and "records" not in p.name
    }
    index_before = (tmp_path / "capture" / "raw_index.jsonl").read_bytes()
    second = save(made, ports, "qwen")
    assert first.archived["contract_written"] and all(
        i["written"] for i in first.archived["raw"].values()
    )
    assert not second.archived["contract_written"] and not any(
        i["written"] for i in second.archived["raw"].values()
    )
    assert first.archived["contract_sha256"] == second.archived["contract_sha256"]
    assert {
        p: p.stat().st_mtime_ns
        for p in (tmp_path / "capture").rglob("*")
        if p.is_file() and "records" not in p.name
    } == files
    assert (
        tmp_path / "capture" / "raw_index.jsonl"
    ).read_bytes() == index_before  # nothing indexed twice
    assert (
        len(archive) == 2 and len(list(CaptureArchive(tmp_path).records())) == 2
    )  # but every capture has its record


def test_t_t2_14_a_broken_chain_is_found_when_the_archive_is_opened(tmp_path: Path) -> None:
    made, _store, _archive, ports, _gate = saver(tmp_path)
    save(made, ports)
    save(made, ports, "qwen")
    CaptureArchive(tmp_path)  # fine as it is
    for name in ("records.jsonl", "raw_index.jsonl"):
        path = tmp_path / "capture" / name
        good = path.read_text(encoding="utf-8")
        path.write_text(good.replace('"seq":0', '"seq":5', 1), encoding="utf-8")  # a line changed
        with pytest.raises(ChainError):
            CaptureArchive(tmp_path)
        path.write_text(
            "".join(good.splitlines(keepends=True)[1:]), encoding="utf-8"
        )  # the first line removed
        with pytest.raises(ChainError):
            CaptureArchive(tmp_path)
        path.write_text(good[:-20], encoding="utf-8")  # the last line cut
        with pytest.raises(ChainError):
            CaptureArchive(tmp_path)
        path.write_text(good, encoding="utf-8")
    CaptureArchive(tmp_path)


def test_t_t2_14_the_files_are_by_hash_and_a_wrong_file_is_found(tmp_path: Path) -> None:
    made, _store, archive, ports, _gate = saver(tmp_path)
    result = save(made, ports)
    digest = result.archived["contract_sha256"]
    path = tmp_path / "capture" / "contracts" / digest[:2] / digest[2:4] / digest
    assert path.is_file()
    raw_sha = next(iter(result.archived["raw"].values()))["sha256"]
    with pytest.raises(ArchiveError):  # a name that is not the bytes' hash
        archive._put("raw", "0" * 64, b"x")
    with pytest.raises(ArchiveError):  # a contract that does not hash to the name
        archive.write_contract({"a": 1}, digest)
    with pytest.raises(ArchiveError):  # floats have no stable text
        archive.write_contract({"a": 1.5}, "0" * 64)
    path.write_bytes(b"changed")
    with pytest.raises(ArchiveError):
        archive.read_contract(digest)
    assert archive.read_contract("f" * 64) is None and archive.read_raw("f" * 64) is None
    assert raw_sha in {i["sha256"] for i in result.archived["raw"].values()}


def test_the_chain_format_is_the_hekb_chainlog_format(tmp_path: Path) -> None:
    chain = ChainFile(tmp_path / "c.jsonl")
    line = chain.append({"b": 2, "a": "é"})
    text = (tmp_path / "c.jsonl").read_text(encoding="utf-8")
    assert json.loads(text) == line and set(line) == {"seq", "prev", "record", "hash"}
    assert (
        text == json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n" and "\\u00e9" in text
    )
    assert line["prev"] == "0" * 64 and ChainFile(tmp_path / "c.jsonl").head == line["hash"]


def test_an_invalid_record_is_not_appended(tmp_path: Path) -> None:
    archive = CaptureArchive(tmp_path)
    with pytest.raises(ArchiveError):
        archive.append_record({"record_version": "x"})
    assert len(archive) == 0 and not (tmp_path / "capture" / "records.jsonl").exists()


# ---- T-T2-15: the store refuses one envelope -----------------------------------------------------


def test_t_t2_15_one_envelope_refused_by_the_store(tmp_path: Path) -> None:
    store = FakeStore(refuse={2: StoreRefusal("gate_rejected", "result FAIL")})
    made, _s, archive, ports, _gate = saver(tmp_path, store=store)
    result = save(made, ports, "qwen")
    assert_valid(result)
    points = list(result.record["positions"])
    refused_point = points[2]
    assert set(result.receipts) == set(points) - {refused_point} and len(store.calls) == 5
    entry = result.record["positions"][refused_point]
    assert entry["status"] == "enveloped" and entry["stored"] is None
    assert entry["store_refusal"] == {
        "code": "gate_rejected",
        "detail": "StoreRefusal: gate_rejected: result FAIL",
    }
    assert entry["verdict_sha256"] == result.verdicts[refused_point]["verdict_sha256"]
    for point in set(points) - {refused_point}:
        assert (
            result.record["positions"][point]["stored"]["artifact_id"]
            == result.record["positions"][point]["artifact_id"]
        )
        assert result.record["positions"][point]["store_refusal"] is None
    assert result.store_refusals == {refused_point: entry["store_refusal"]}
    # the contract and the RAW files are in the archive, and the record is the last line
    assert archive.read_contract(result.archived["contract_sha256"]) == result.contract
    assert all(archive.read_raw(i["sha256"]) for i in result.archived["raw"].values())
    assert list(CaptureArchive(tmp_path).records()) == [result.record]


# ---- R-9: a record is always written -------------------------------------------------------------


def _boom(name: str) -> Any:
    def raiser(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"{name} went wrong")

    return raiser


@pytest.mark.parametrize(
    ("stage", "code"),
    [("space_id_of", "space_id_failed"), ("seal", "seal_failed")],
)
def test_r9_space_id_of_and_seal_failing_leave_a_record_and_the_archive(
    tmp_path: Path, stage: str, code: str
) -> None:
    made, store, archive, ports, gate = saver(tmp_path, **{stage: _boom(stage)})
    result = save(made, ports, "qwen")
    assert_valid(result)
    assert result.stage == f"failed:{code}" and result.record["outcome"] == "refused"
    assert result.record["refusals"] == [f"{code}: RuntimeError: {stage} went wrong"]
    entries = result.record["positions"]
    assert len(entries) == 5 and all(
        e["status"] == "not_built" and e["reasons"] == [code] for e in entries.values()
    )
    assert result.record["contract"]["status"] == "VERIFIED"
    assert gate.calls == [] and store.calls == [] and len(archive) == 1
    assert (
        archive.read_contract(result.archived["contract_sha256"]) is not None
    )  # what was made is kept
    assert all(archive.read_raw(i["sha256"]) for i in result.archived["raw"].values())


def test_r9_the_gate_failing_leaves_a_record_and_the_store_is_not_asked(tmp_path: Path) -> None:
    made, store, archive, ports, _gate = saver(
        tmp_path, gate=Gate(GateUnavailable("the child exited 1"))
    )
    result = save(made, ports)
    assert_valid(result)
    entry = result.record["positions"]["final_norm"]
    assert (
        entry["gate"]["status"] == "gate_error"
        and entry["stored"] is None
        and entry["verdict_sha256"] is None
    )
    assert (
        entry["store_refusal"]["code"] == "gate_error"
        and "GateUnavailable" in entry["store_refusal"]["detail"]
    )
    assert store.calls == [] and len(archive) == 1 and result.gate_errors


def test_r9_the_store_raising_without_a_code_is_put_failed(tmp_path: Path) -> None:
    made, _store, archive, ports, _gate = saver(
        tmp_path, store=FakeStore(refuse={0: OSError("disk full")})
    )
    result = save(made, ports)
    assert_valid(result)
    entry = result.record["positions"]["final_norm"]
    assert (
        entry["store_refusal"] == {"code": "put_failed", "detail": "OSError: disk full"}
        and entry["stored"] is None
    )
    assert entry["verdict_sha256"] and len(archive) == 1


def test_r9_the_archive_write_failing_leaves_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made, store, archive, ports, gate = saver(tmp_path)
    monkeypatch.setattr(archive, "write_raw", _boom("write_raw"))
    result = save(made, ports)
    assert_valid(result)
    assert result.stage == "failed:archive_write_failed" and result.record["outcome"] == "refused"
    assert result.record["positions"]["final_norm"]["reasons"] == ["archive_write_failed"]
    assert gate.calls == [] and store.calls == [] and len(archive) == 1


def test_r9_the_contract_stage_and_the_check_failing_leave_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nvs_runtime.hidden_trajectory import formal_save

    made, _store, archive, ports, _gate = saver(tmp_path)
    monkeypatch.setattr(formal_save, "check_contract_v2", _boom("check"))
    result = save(made, ports)
    assert_valid(result)
    assert (
        result.stage == "failed:contract_check_failed"
        and result.record["contract"]["status"] == "none"
    )
    assert (
        result.record["positions"]["final_norm"]["reasons"] == ["contract_check_failed"]
        and result.contract is not None
    )
    monkeypatch.setattr(formal_save, "contract_from_captured", _boom("contract"))
    result = save(made, ports)
    assert_valid(result)
    assert result.stage == "failed:contract_stage_failed" and result.record["positions"] == {}
    assert result.record["capture"]["points"] == [] and result.contract is None
    assert len(archive) == 2


def test_r9_only_a_failure_of_the_records_own_append_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made, _store, archive, ports, _gate = saver(tmp_path)
    monkeypatch.setattr(ChainFile, "_write", _boom("append"))
    with pytest.raises(RuntimeError, match="append went wrong"):
        save(made, ports)
    assert len(archive) == 0


# ---- the other paths of the record, and the caller's mistakes ------------------------------------


def test_a_refused_capture_and_an_empty_generation_have_their_records(tmp_path: Path) -> None:
    from u65_support import make_facts

    made, store, archive, ports, gate = saver(tmp_path)
    capture = make_capture()
    refused = made.save(
        pair(capture, make_facts(capture, **{"origin.revision": "f" * 40})),
        contract_session_id="refused",
        synthetic=False,
        ids=ports.ids,
    )
    assert refused.stage == "refused" and refused.record["refusals"] == [
        "origin_mismatch: origin.revision"
    ]
    empty = made.save(
        pair(make_capture(n_gen=0)), contract_session_id="empty", synthetic=False, ids=ports.ids
    )
    assert empty.stage == "empty_generation" and empty.record["outcome"] == "empty_generation"
    assert len(archive) == 2 and store.calls == [] and gate.calls == []
    assert (
        not (tmp_path / "capture" / "contracts").exists()
        and not (tmp_path / "capture" / "raw").exists()
    )


def parent_ref(ports: Ports) -> ParentRef:
    return ParentRef(PARENT_ARTIFACT, "cnt:" + "8" * 64, 0, ports.ids.observation_id)


def held_parent(parent: ParentRef, **changes: Any) -> dict[str, Any]:
    return {
        "artifact_id": parent.artifact_id,
        "content_id": parent.content_id,
        "seq": parent.seq,
        "observation_id": parent.observation_id,
        **changes,
    }


def test_the_parent_is_read_from_the_formal_store(tmp_path: Path) -> None:
    store = FakeStore()
    made, _s, archive, ports, gate = saver(tmp_path, store=store)
    parent = parent_ref(ports)
    store.held[parent.artifact_id] = {"envelope": held_parent(parent)}
    result = save(made, ports, parent=parent)
    assert result.stage == "built" and gate.calls[0]["stored"] == {
        PARENT_ARTIFACT: {"envelope": held_parent(parent)}
    }
    assert PARENT_ARTIFACT in gate.calls[0]["lookup"] and len(archive) == 1
    assert_valid(result)


def test_a_parent_the_store_does_not_have_is_a_refusal_record_and_nothing_is_gated(
    tmp_path: Path,
) -> None:
    made, store, archive, ports, gate = saver(tmp_path)
    result = save(made, ports, parent=parent_ref(ports))
    assert_valid(result)
    assert result.stage == "refused" and result.record["refusals"] == [
        f"parent_not_stored: {PARENT_ARTIFACT}"
    ]
    assert all(e["reasons"] == ["parent_not_stored"] for e in result.record["positions"].values())
    assert gate.calls == [] and store.calls == [] and len(archive) == 1
    assert archive.read_contract(result.archived["contract_sha256"]) is not None


def test_a_mismatched_or_unfetched_parent_is_the_callers_mistake_and_writes_nothing(
    tmp_path: Path,
) -> None:
    store = FakeStore()
    made, _s, archive, ports, gate = saver(tmp_path, store=store)
    parent = parent_ref(ports)
    store.held[parent.artifact_id] = {"envelope": held_parent(parent, seq=9)}
    with pytest.raises(ParentMismatch):
        save(made, ports, parent=parent)
    with pytest.raises(ParentEnvelopeMissing):
        save(made, ports, parent=parent, parent_envelope=None)
    assert gate.calls == [] and store.calls == [] and len(archive) == 0
    assert not (tmp_path / "capture").exists() or not any((tmp_path / "capture").rglob("*.jsonl"))


# ---- the record's three new keys -----------------------------------------------------------------


def test_the_schema_of_the_stored_verdict_and_refusal_keys(tmp_path: Path) -> None:
    made, _store, _archive, ports, _gate = saver(tmp_path)
    good = save(made, ports).record
    assert validate_record(good) == []
    point = "final_norm"

    def changed(**entry: Any) -> list[str]:
        import copy

        record = copy.deepcopy(good)
        record["positions"][point].update(entry)
        return validate_record(record)

    assert changed(stored=None, store_refusal={"code": "x", "detail": "y"}) == []
    assert changed(
        stored={"artifact_id": "art:other", "hekb_seq": 1, "hekb_hash": "a" * 64}
    )  # not the position's own
    assert changed(
        stored={
            "artifact_id": good["positions"][point]["artifact_id"],
            "hekb_seq": -1,
            "hekb_hash": "a" * 64,
        }
    )
    assert changed(
        stored={
            "artifact_id": good["positions"][point]["artifact_id"],
            "hekb_seq": 1,
            "hekb_hash": "A" * 64,
        }
    )
    assert changed(stored={"artifact_id": good["positions"][point]["artifact_id"], "hekb_seq": 1})
    assert changed(store_refusal={"code": "x", "detail": "y"})  # stored and refusal both
    assert changed(stored=None, store_refusal={"code": "", "detail": "y"})
    assert changed(stored=None, store_refusal={"code": "x", "detail": "y" * 513})
    assert changed(stored=None, store_refusal={"code": "x"})
    assert changed(verdict_sha256="xyz") and changed(verdict_sha256=5)
    assert changed(extra=1)
    assert changed(verdict_sha256=None, stored=None, store_refusal=None) == []
    old = rec.position_entry(rec.POSITION_NOT_BUILT, ("x",))
    assert set(old) == {
        "status",
        "reasons",
        "artifact_id",
        "gate",
    }  # a record without them is as before
