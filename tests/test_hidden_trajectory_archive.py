"""The capture archive and its chain file, by themselves (U6T-2b'; T-T2-14). No Adapter, no Gate, no
``nvs_edge``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nvs_runtime.hidden_trajectory import record as rec
from nvs_runtime.hidden_trajectory.capture_archive import (
    ArchiveError,
    CaptureArchive,
    contract_bytes,
)
from nvs_runtime.hidden_trajectory.capture_chain import ChainError, ChainFile


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def a_record(session: str = "ses:1") -> dict:
    capture = rec.capture_block(
        n_in=1,
        n_gen=0,
        finish_reason=None,
        points=[],
        model_build_id=None,
        compute_environment=None,
        adapter=None,
        prompt_ids_sha256=None,
        gen_token_ids_sha256=None,
    )
    return rec.empty_generation_record(session_id=session, observation_id="obs:1", capture=capture)


def test_the_chain_format_and_its_hash() -> None:
    from nvs_runtime.hidden_trajectory.capture_chain import GENESIS, canonical, line_hash

    assert canonical({"b": 1, "a": "é"}) == '{"a":"\\u00e9","b":1}'
    assert line_hash(GENESIS, {"a": 1}) == sha((GENESIS + '{"a":1}').encode())


def test_append_verify_reopen_and_the_head(tmp_path: Path) -> None:
    chain = ChainFile(tmp_path / "c.jsonl")
    first, second = chain.append({"n": 1}), chain.append({"n": 2})
    text = (tmp_path / "c.jsonl").read_text(encoding="utf-8")
    assert text == "".join(
        json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in (first, second)
    )
    assert (first["seq"], first["prev"], second["seq"], second["prev"]) == (
        0,
        "0" * 64,
        1,
        first["hash"],
    )
    again = ChainFile(tmp_path / "c.jsonl")
    assert (
        len(again) == 2
        and again.head == second["hash"]
        and list(again.records()) == [{"n": 1}, {"n": 2}]
    )
    assert again.append({"n": 3})["prev"] == second["hash"]


def test_a_broken_chain_is_found_on_open(tmp_path: Path) -> None:
    chain = ChainFile(tmp_path / "c.jsonl")
    for n in range(3):
        chain.append({"n": n})
    path = tmp_path / "c.jsonl"
    good = path.read_text(encoding="utf-8")
    lines = good.splitlines(keepends=True)
    for broken in (
        good.replace('"n":1', '"n":9', 1),  # a record changed
        "".join(lines[1:]),  # the first line removed
        lines[0] + lines[2],  # a middle line removed
        lines[0] + lines[1] + lines[1],  # a line repeated
        good[:-15],  # the last line cut
        good + "not json\n",  # a line that is not JSON
    ):
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(ChainError):
            ChainFile(path)
    path.write_text(good, encoding="utf-8")
    assert len(ChainFile(path)) == 3


def test_contract_bytes_are_canonical_and_floats_are_refused() -> None:
    assert (
        contract_bytes({"b": [1, {"z": "é"}], "a": None}) == '{"a":null,"b":[1,{"z":"é"}]}'.encode()
    )
    with pytest.raises(ArchiveError):
        contract_bytes({"a": [1.5]})


def test_a_contract_is_kept_by_its_hash_once(tmp_path: Path) -> None:
    archive = CaptureArchive(tmp_path)
    document = {"session_id": "s", "positions": [{"point": "p", "trajectory_id": "t" * 64}]}
    digest = sha(contract_bytes(document))
    assert archive.write_contract(document, digest) is True
    path = tmp_path / "capture" / "contracts" / digest[:2] / digest[2:4] / digest
    stamp = path.stat().st_mtime_ns
    assert archive.write_contract(document, digest) is False and path.stat().st_mtime_ns == stamp
    assert archive.read_contract(digest) == document and archive.read_contract("0" * 64) is None
    with pytest.raises(ArchiveError):
        archive.write_contract(document, "0" * 64)  # the bytes are not those of the name
    path.write_bytes(b"{}")
    with pytest.raises(ArchiveError):
        archive.read_contract(digest)  # a file that does not match its name


def test_raw_files_are_kept_by_hash_and_indexed_by_trajectory_id(tmp_path: Path) -> None:
    archive = CaptureArchive(tmp_path)
    raw = {"a": b"raw-a", "b": b"raw-b"}
    done = archive.write_raw(raw, {"a": "ta", "b": "tb"})
    assert done == {
        "a": {"sha256": sha(b"raw-a"), "written": True},
        "b": {"sha256": sha(b"raw-b"), "written": True},
    }
    index = (tmp_path / "capture" / "raw_index.jsonl").read_bytes()
    again = archive.write_raw(raw, {"a": "ta", "b": "tb"})
    assert not any(v["written"] for v in again.values())
    assert (tmp_path / "capture" / "raw_index.jsonl").read_bytes() == index  # not indexed twice
    assert (
        archive.read_raw(sha(b"raw-a")) == b"raw-a" and archive.raw_for_trajectory("tb") == b"raw-b"
    )
    assert archive.raw_for_trajectory("unknown") is None
    with pytest.raises(ArchiveError):
        archive.write_raw({"a": b"other"}, {"a": "ta"})  # one trajectory_id, two files
    reopened = CaptureArchive(tmp_path)  # the index is read back
    assert reopened.raw_for_trajectory("ta") == b"raw-a"


def test_records_are_chained_and_an_invalid_one_is_not_appended(tmp_path: Path) -> None:
    archive = CaptureArchive(tmp_path)
    with pytest.raises(ArchiveError):
        archive.append_record({"record_version": "x"})
    assert len(archive) == 0 and not (tmp_path / "capture" / "records.jsonl").exists()
    one, two = a_record("ses:1"), a_record("ses:2")
    archive.append_record(one)
    archive.append_record(two)
    assert list(CaptureArchive(tmp_path).records()) == [one, two]
    path = tmp_path / "capture" / "records.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("ses:1", "ses:9"), encoding="utf-8")
    with pytest.raises(ChainError):
        CaptureArchive(tmp_path)


def test_a_broken_index_chain_is_found_on_open(tmp_path: Path) -> None:
    archive = CaptureArchive(tmp_path)
    archive.write_raw({"a": b"x"}, {"a": "ta"})
    path = tmp_path / "capture" / "raw_index.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace('"ta"', '"tz"'), encoding="utf-8")
    with pytest.raises(ChainError):
        CaptureArchive(tmp_path)
