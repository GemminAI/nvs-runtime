"""U6T-2b': a made-up capture through the real Gate (U6T-1) into a real Formal Store of hekb-vnext
(U6T-2a), and the chain format.

Opt-in. Skipped, with the reason, unless ``HEKB_FORMAL_ROOT`` (a hekb-vnext checkout with the Formal
Store) is set; the Gate tests also need ``GATE_PY`` and ``SENSOS_TOOLS``
(as ``test_hidden_trajectory_gate.py``). hekb-vnext is read, never written (its ``data_dir`` is a
temporary folder here).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from nvs_observation import capture_edge
from test_hidden_trajectory_gate import EXAMPLE, _skip_reason
from test_hidden_trajectory_run import clock, pair
from u65_support import make_capture

from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.capture_chain import ChainFile
from nvs_runtime.hidden_trajectory.envelope import EnvelopeIds
from nvs_runtime.hidden_trajectory.formal_save import _CaptureSaver as CaptureSaver
from nvs_runtime.hidden_trajectory.gate_client import git_head
from nvs_runtime.hidden_trajectory.record_schema import validate_record
from nvs_runtime.integrations.hekb_formal import (
    _create_formal_store as create_formal_store,
)
from nvs_runtime.integrations.hekb_formal import (
    _load_hekb_module as load_hekb_module,
)
from nvs_runtime.integrations.hekb_formal import (
    _open_formal_store as open_formal_store,
)

EDGE, _ = capture_edge.current()
HEKB = os.environ.get("HEKB_FORMAL_ROOT")


def _hekb_reason() -> str | None:
    if not HEKB:
        return "HEKB_FORMAL_ROOT is not set (a hekb-vnext checkout with the Formal Store)"
    if not (Path(HEKB) / "graph" / "formal.py").is_file():
        return "HEKB_FORMAL_ROOT has no graph/formal.py (a checkout with the Formal Store)"
    return None


HEKB_REASON = _hekb_reason()
needs_hekb = pytest.mark.skipif(HEKB_REASON is not None, reason=HEKB_REASON or "")


def _gate_reason() -> str | None:
    return (
        HEKB_REASON
        or _skip_reason()
        or (None if EDGE is not None else "nvs_edge contract v2 is not usable")
    )


GATE_REASON = _gate_reason()
needs_gate = pytest.mark.skipif(GATE_REASON is not None, reason=GATE_REASON or "")


@needs_hekb
def test_the_chain_file_is_byte_for_byte_hekb_vnexts_chainlog(tmp_path: Path) -> None:
    assert HEKB is not None
    chainlog = load_hekb_module(HEKB, "graph.chainlog")
    ours, theirs = ChainFile(tmp_path / "ours.jsonl"), chainlog.ChainLog(tmp_path / "theirs.jsonl")
    for record in ({"a": 1, "b": [1, 2, {"z": None}]}, {"é": "ü", "n": 0}, {"nested": {"k": "v"}}):
        assert ours.append(record) == theirs.append(record)
    assert (tmp_path / "ours.jsonl").read_bytes() == (tmp_path / "theirs.jsonl").read_bytes()
    assert ours.head == theirs.head
    assert [x["record"] for x in chainlog.ChainLog(tmp_path / "ours.jsonl").verify()] == list(
        ChainFile(tmp_path / "theirs.jsonl").records()
    )


@pytest.fixture(scope="module")
def door() -> Any:
    sys.path.insert(0, str(EXAMPLE))
    try:
        from sensos_door import SensosDoor
    finally:
        sys.path.remove(str(EXAMPLE))
    return SensosDoor.from_environment()


def _ids(door: Any, item: str) -> EnvelopeIds:
    session = door.session_id(spec="0" * 64, item_id=item, condition="run", replicate=0, rng_seed=0)
    sequence = iter(range(1, 10**6))
    return EnvelopeIds(
        session, door.observation_id(session, 0), "2026-11-17T00:00:00Z", lambda: next(sequence)
    )


def _declaration(door: Any) -> dict[str, Any]:
    assert HEKB is not None
    formal = load_hekb_module(HEKB, "graph.formal")
    catalog = door.client.process.call({"op": "catalog"})
    version = git_head(door.sensos_tools.resolve().parent) or "unknown"
    declaration: dict[str, Any] = formal.formal_declaration(
        catalog_rows=catalog["rows"], catalog_version=version, created_by="u6t2b-integration"
    )
    return declaration


@needs_gate
@pytest.mark.parametrize(("kind", "count"), [("llama", 1), ("qwen", 5)])
def test_a_capture_goes_through_the_real_gate_into_a_real_formal_store(
    tmp_path: Path, door: Any, kind: str, count: int
) -> None:
    assert HEKB is not None
    store = create_formal_store(HEKB, tmp_path, _declaration(door))
    archive = CaptureArchive(tmp_path)
    saver = CaptureSaver(
        store=store,
        archive=archive,
        gate=door.verdict,
        space_id_of=door.space_id_of,
        seal=door.seal,
        checker_bundle=door.client.checker_bundle,
        clock=clock,
    )
    result = saver.save(
        pair(make_capture(kind=kind)),
        contract_session_id=f"u6t2b-{kind}",
        synthetic=False,
        ids=_ids(door, f"it-{kind}"),
    )
    assert result.stage == "built" and validate_record(result.record) == []
    assert len(result.receipts) == count and not result.store_refusals and not result.gate_errors
    assert result.record["outcome"] == "enveloped"
    for point, entry in result.record["positions"].items():
        got = store.get(entry["artifact_id"])
        assert got is not None and got["envelope"] == result.envelopes[point]
        assert entry["stored"] == {
            "artifact_id": got["envelope"]["artifact_id"],
            "hekb_seq": got["hekb_seq"],
            "hekb_hash": got["hekb_hash"],
        }
        assert entry["verdict_sha256"] == got["gate_verdict"]["verdict_sha256"]
        assert got["gate_verdict"] == result.verdicts[point]
        assert store.gate_status(entry["artifact_id"])["self"] in ("PASS", "PASS_UNVERIFIED")
        assert entry["gate"]["status"] == got["gate_verdict"]["result"]
    assert store.checker_bundle(
        result.record["positions"][next(iter(result.record["positions"]))]["artifact_id"]
    ) == (door.client.checker_bundle())
    # what the record names is in the archive, the store opens again as a Formal Store, the
    # archive's chains verify
    assert archive.read_contract(result.archived["contract_sha256"]) == result.contract
    reopened = open_formal_store(HEKB, tmp_path)
    assert not reopened.quarantined and len(reopened) == count
    assert list(CaptureArchive(tmp_path).records()) == [result.record]


@needs_gate
def test_a_second_save_of_the_same_capture_stores_nothing_new_and_writes_no_file_again(
    tmp_path: Path, door: Any
) -> None:
    assert HEKB is not None
    store = create_formal_store(HEKB, tmp_path, _declaration(door))
    saver = CaptureSaver(
        store=store,
        archive=CaptureArchive(tmp_path),
        gate=door.verdict,
        space_id_of=door.space_id_of,
        seal=door.seal,
        checker_bundle=door.client.checker_bundle,
        clock=clock,
    )
    kwargs: dict[str, Any] = {"contract_session_id": "u6t2b-twice", "synthetic": False}
    first = saver.save(pair(make_capture(kind="qwen")), ids=_ids(door, "twice"), **kwargs)
    second = saver.save(pair(make_capture(kind="qwen")), ids=_ids(door, "twice"), **kwargs)
    assert len(store) == 5 and all(r["already_stored"] for r in second.receipts.values())
    assert [r["hekb_seq"] for r in first.receipts.values()] == [
        r["hekb_seq"] for r in second.receipts.values()
    ]
    assert not second.archived["contract_written"] and not any(
        i["written"] for i in second.archived["raw"].values()
    )
