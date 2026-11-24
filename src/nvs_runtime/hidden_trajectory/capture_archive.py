"""The capture archive (U6T-2 design 6): where the contract document, the RAW files and the capture
record of every capture are kept.

It lives beside a Formal Store, under ``<data_dir>/capture/``, and is never mixed with the
store's chain, declaration or index::

    capture/contracts/aa/bb/<contract_sha256>   the canonical JSON bytes of the contract document
    capture/raw/aa/bb/<sha256 of the file>      a RAW file (raw_store format), as encoded
    capture/raw_index.jsonl                     chain: {raw_sha256, trajectory_id}, every RAW file
    capture/records.jsonl                       chain: the capture records (nvs.capture-record/1)

Content is found by its hash; the same content is never written twice; nothing is overwritten or
deleted. No Gate is run here: a capture record is not an IC-00 artifact (DC-U6-3 (a)), so this is
not "formal storage". Only the save (``formal_save.py``) writes
here, and it writes through one object. Standard library plus this package.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from nvs_runtime.hidden_trajectory.capture_chain import ChainError, ChainFile
from nvs_runtime.hidden_trajectory.record_schema import validate_record

__all__ = ["ArchiveError", "CaptureArchive", "ChainError", "contract_bytes"]


class ArchiveError(Exception):
    """An archive write or read that cannot be done as asked (a name that does not match its bytes,
    a file that differs...)."""


def _reject_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise ArchiveError(f"the contract has a float at {path}: it has no stable text form")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_floats(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for i, item in enumerate(value):
            _reject_floats(item, f"{path}[{i}]")


def contract_bytes(document: Mapping[str, Any]) -> bytes:
    """The canonical bytes of a contract document (sorted keys, no spaces, UTF-8), the same bytes
    the contract hash is taken over."""
    _reject_floats(document)
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CaptureArchive:
    """``CaptureArchive(data_dir)``: opens (and verifies) both chains; ``ChainError`` if one is
    broken."""

    def __init__(self, data_dir: Path | str, *, guarded: bool = False) -> None:
        """``guarded`` (the ``FormalWriter`` only): both chains check the file before an append and
        read a half-written last line as "not written" until ``set_aside_torn_tails`` moves it
        (see ``capture_chain``)."""
        self.root = Path(data_dir) / "capture"
        self._index = ChainFile(self.root / "raw_index.jsonl", guarded=guarded)
        self._records = ChainFile(self.root / "records.jsonl", guarded=guarded)
        self._trajectory: dict[str, str] = {}
        for entry in self._index.records():
            self._trajectory[entry["trajectory_id"]] = entry["raw_sha256"]

    # ---- the guards of the chains (guarded archive) ---------------------------------------------
    def set_aside_torn_tails(self) -> list[dict[str, Any]]:
        """Set aside the half-written last line of each chain, if there is one. Returns what was
        recorded."""
        done = (chain.set_aside_torn_tail() for chain in (self._index, self._records))
        return [record for record in done if record is not None]

    def torn_tails(self) -> list[dict[str, Any]]:
        """Every last line set aside in this archive (``quarantine/torn_tails.jsonl``)."""
        return self._records.torn_records()

    def check_unchanged(self) -> None:
        """``ChainChangedExternally`` if either chain was written by somebody else."""
        self._index.check_unchanged()
        self._records.check_unchanged()

    @property
    def dead(self) -> str | None:
        """The reason a chain of this archive saw another writer (and will not be written), or
        ``None``."""
        return self._index.dead or self._records.dead

    # ---- files by hash -----------------------------------------------------------------------
    def _path(self, kind: str, sha: str) -> Path:
        return self.root / kind / sha[:2] / sha[2:4] / sha

    def _put(self, kind: str, sha: str, data: bytes) -> bool:
        """Write ``data`` under its hash; ``False`` (nothing written) if it is already there. A file
        with the name and other bytes is an error."""
        if _sha(data) != sha:
            raise ArchiveError(f"{kind} {sha}: the bytes are not those of the name")
        path = self._path(kind, sha)
        if path.exists():
            if path.read_bytes() != data:
                raise ArchiveError(f"{kind} {sha}: a file of this name has other bytes")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True

    def _get(self, kind: str, sha: str) -> bytes | None:
        path = self._path(kind, sha)
        if not path.is_file():
            return None
        data = path.read_bytes()
        if _sha(data) != sha:
            raise ArchiveError(f"{kind} {sha}: the file does not match its hash")
        return data

    def write_contract(self, document: Mapping[str, Any], contract_sha256: str) -> bool:
        """Keep the contract document under ``contract_sha256``. The canonical bytes must hash to it
        (else ``ArchiveError``)."""
        return self._put("contracts", contract_sha256, contract_bytes(document))

    def read_contract(self, contract_sha256: str) -> dict[str, Any] | None:
        data = self._get("contracts", contract_sha256)
        if data is None:
            return None
        document: dict[str, Any] = json.loads(data)
        return document

    def write_raw(
        self, raw_bytes: Mapping[str, bytes], trajectory_ids: Mapping[str, str]
    ) -> dict[str, dict[str, Any]]:
        """Keep every RAW file under the SHA-256 of its bytes and index its ``trajectory_id``.
        ``{point: {sha256, written}}``."""
        out: dict[str, dict[str, Any]] = {}
        for point, data in raw_bytes.items():
            digest = _sha(data)
            tid = trajectory_ids[point]
            written = self._put("raw", digest, data)
            known = self._trajectory.get(tid)
            if known is None:
                self._index.append({"raw_sha256": digest, "trajectory_id": tid})
                self._trajectory[tid] = digest
            elif known != digest:
                raise ArchiveError(f"trajectory_id {tid} is indexed with another RAW file")
            out[point] = {"sha256": digest, "written": written}
        return out

    def read_raw(self, raw_sha256: str) -> bytes | None:
        return self._get("raw", raw_sha256)

    def raw_for_trajectory(self, trajectory_id: str) -> bytes | None:
        digest = self._trajectory.get(trajectory_id)
        return None if digest is None else self.read_raw(digest)

    # ---- the K path and the id lists (U6-6a) ----------------------------------------------------
    def write_k(self, data: bytes) -> dict[str, Any]:
        """Keep a K file (the board records, or the causal log) under the SHA-256 of its bytes.
        ``{sha256, written}``."""
        digest = _sha(data)
        return {"sha256": digest, "written": self._put("k", digest, data)}

    def read_k(self, sha256: str) -> bytes | None:
        return self._get("k", sha256)

    def write_ids(self, data: bytes, expected_sha256: str) -> bool:
        """Keep an id list (little-endian int64, as the contract's ``prompt_ids_sha256`` and
        ``gen_token_ids_sha256`` hash it) under ``expected_sha256``, the contract's value. Bytes
        that
        do not hash to it are refused and nothing is written (``ArchiveError``)."""
        return self._put("ids", expected_sha256, data)

    def read_ids(self, sha256: str) -> bytes | None:
        return self._get("ids", sha256)

    # ---- the capture records -----------------------------------------------------------------
    def append_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Chain one capture record (it must fit ``nvs.capture-record/1``). Returns its line. Raises
        if the record is invalid or the file cannot be written."""
        problems = validate_record(record)
        if problems:
            raise ArchiveError(
                f"the record does not fit nvs.capture-record/1: {'; '.join(problems[:3])}"
            )
        return self._records.append(record)

    def records(self) -> Iterator[dict[str, Any]]:
        return self._records.records()

    def __len__(self) -> int:
        return len(self._records)
