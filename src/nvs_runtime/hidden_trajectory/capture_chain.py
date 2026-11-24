"""An append-only hash chain of JSON lines, in the format of hekb-vnext's ``graph/chainlog.py``
(U6T-2b').

Each line: ``{"seq": n, "prev": <hash of the previous line>, "record": {...}, "hash": <sha256 of
prev + canonical(record)>}``,
``canonical`` being sorted keys, no spaces, ASCII escapes. Lines are only appended. The format is
written here and not imported
(nvs-runtime does not import hekb-vnext for this); a test compares the bytes with hekb-vnext's own
``ChainLog`` when its
checkout is given.

``ChainFile._write`` is the one place that touches the file: a check before the write (U6T-2b-a:
someone else wrote since the
last append) and the handling of a half-written last line belong there, and nowhere else.

A chain opened with ``guarded=True`` (the archive of the ``FormalWriter``, U6T-2b-b; the same
ideas as hekb-vnext's
``GuardedChainLog``, and the same files and records for a torn line):

1. before every append the file is compared with the memory (its size, and the ``hash`` of its last
   line against ``head``). A
   difference means somebody else wrote: nothing is written, the handle never writes again
   (``dead``), reads still work.
   This is detection, not a lock: a writer between the check and the write is not seen (U6T-8).
2. a last line that does not end in a newline was being written when the process stopped. It is
   read as "not written"; the
   bytes are saved first (``quarantine/<file>.<length>.torn``, fsync), described in
   ``quarantine/torn_tails.jsonl`` (fsync),
   and only then is the file cut back to the start of that line (fsync). A failure before the cut
   leaves the file as it was
   and the chain does not open. A line that ends in a newline and cannot be read, a chain that does
   not add up and a fork
   are not touched: they fail as before.

A chain opened without ``guarded`` behaves as it always did.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

GENESIS: Final = "0" * 64


class ChainError(RuntimeError):
    """The chain does not verify (a line was changed, removed, inserted or cut)."""


class ChainChangedExternally(ChainError):
    """The file is not what this handle last wrote (size or last line differ from memory). Nothing
    was written."""

    code = "store_changed_externally"


class TornTailError(ChainError):
    """The half-written last line could not be set aside safely; the file was not changed."""


QUARANTINE_DIR: Final = "quarantine"
TORN_RECORD_FILE: Final = "torn_tails.jsonl"


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def line_hash(prev: str, record: Any) -> str:
    return hashlib.sha256((prev + canonical(record)).encode("utf-8")).hexdigest()


class ChainFile:
    """One chain file. Opening verifies the whole file (``ChainError`` if it is broken)."""

    def __init__(self, path: Path | str, *, guarded: bool = False) -> None:
        self.path = Path(path)
        self._seq = 0
        self._head = GENESIS
        self._size = 0
        self.guarded = guarded
        self.dead: str | None = None
        self.torn: dict[str, Any] | None = None
        if guarded:
            self._open_guarded()
        elif self.path.exists():
            for line in self.verify():
                self._seq = line["seq"] + 1
                self._head = line["hash"]

    def _open_guarded(self) -> None:
        data = self.path.read_bytes() if self.path.exists() else b""
        cut = data.rfind(b"\n") + 1
        complete, tail = data[:cut], data[cut:]
        for line in self._verify_bytes(complete):
            self._seq = line["seq"] + 1
            self._head = line["hash"]
        self._size = cut
        if tail:
            self.torn = {"offset": cut, "bytes": tail, "original_length": len(data)}

    def _verify_bytes(self, data: bytes) -> Iterator[dict[str, Any]]:
        prev, seq = GENESIS, 0
        for raw in data.split(b"\n")[:-1]:
            try:
                line = json.loads(raw.decode("utf-8"))
                valid = (
                    isinstance(line, dict)
                    and line["seq"] == seq
                    and line["prev"] == prev
                    and line["hash"] == line_hash(prev, line["record"])
                )
            except (ValueError, KeyError, TypeError) as error:
                raise ChainError(f"{self.path.name}: unreadable line at seq {seq}") from error
            if not valid:
                raise ChainError(f"{self.path.name}: chain broken at seq {seq}")
            prev, seq = str(line["hash"]), seq + 1
            yield line

    def __len__(self) -> int:
        return self._seq

    @property
    def head(self) -> str:
        return self._head

    def verify(self) -> Iterator[dict[str, Any]]:
        prev, seq = GENESIS, 0
        if not self.path.exists():
            return
        if self.guarded:
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):
                raise ChainError(f"{self.path.name}: the last line does not end in a newline")
            yield from self._verify_bytes(data)
            return
        with self.path.open(encoding="utf-8") as handle:
            for raw in handle:
                try:
                    line = json.loads(raw)
                    valid = (
                        isinstance(line, dict)
                        and line["seq"] == seq
                        and line["prev"] == prev
                        and line["hash"] == line_hash(prev, line["record"])
                    )
                except (ValueError, KeyError, TypeError) as error:
                    raise ChainError(f"{self.path.name}: unreadable line at seq {seq}") from error
                if not valid:
                    raise ChainError(f"{self.path.name}: chain broken at seq {seq}")
                prev, seq = str(line["hash"]), seq + 1
                yield line

    def records(self) -> Iterator[dict[str, Any]]:
        for line in self.verify():
            record: dict[str, Any] = line["record"]
            yield record

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append one record and return its line. The file is flushed and synced before this
        returns."""
        if self.torn is not None:
            raise TornTailError(f"{self.path.name}: a torn last line is not set aside yet")
        if self.guarded:
            self.check_unchanged()
        body = dict(record)
        line = {
            "seq": self._seq,
            "prev": self._head,
            "record": body,
            "hash": line_hash(self._head, body),
        }
        self._write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
        self._seq += 1
        self._head = str(line["hash"])
        return line

    def _write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        self._size += len(text.encode("utf-8"))

    # ---- the check before an append (guarded) ---------------------------------------------------
    def _last_line_hash(self) -> str:
        if not self._size:
            return GENESIS
        with self.path.open("rb") as handle:
            handle.seek(max(0, self._size - 1 - 65536))
            block = handle.read(min(self._size, 1 + 65536))
        body = block[:-1] if block.endswith(b"\n") else block
        last = body[body.rfind(b"\n") + 1 :]
        try:
            return str(json.loads(last)["hash"])
        except (ValueError, KeyError, TypeError):
            return ""

    def check_unchanged(self) -> None:
        """Raise ``ChainChangedExternally`` (and kill the handle for writing) unless the file is as
        this handle left it."""
        if self.dead:
            raise ChainChangedExternally(self.dead)
        try:
            size = self.path.stat().st_size if self.path.exists() else 0
            same = size == self._size and (self._size == 0 or self._last_line_hash() == self._head)
        except OSError as error:
            self.dead = f"{self.path.name}: cannot be read before an append: {type(error).__name__}"
            raise ChainChangedExternally(self.dead) from error
        if not same:
            self.dead = (
                f"{self.path.name}: the file is not what this handle wrote "
                f"(size {size}, expected {self._size})"
            )
            raise ChainChangedExternally(self.dead)

    # ---- the torn last line: the evidence first, the cut last (guarded) --------------------------
    @property
    def quarantine_dir(self) -> Path:
        return self.path.parent / QUARANTINE_DIR

    @property
    def record_path(self) -> Path:
        return self.quarantine_dir / TORN_RECORD_FILE

    def torn_records(self) -> list[dict[str, Any]]:
        """Every tail that was set aside in this folder (the quarantine record, verified)."""
        if not self.record_path.exists():
            return []
        return list(ChainFile(self.record_path).records())

    def _step_write_evidence(self, target: Path, tail: bytes) -> None:  # step 1
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != tail:
                raise TornTailError(f"{target.name} exists with other bytes")
            return
        part = target.with_name(target.name + ".part")
        with part.open("wb") as handle:
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, target)
        _fsync_dir(target.parent)

    def _step_write_record(self, record: dict[str, Any]) -> None:  # step 2
        log = ChainFile(self.record_path)
        for earlier in log.records():
            if all(earlier.get(k) == record[k] for k in ("file", "original_length", "torn_sha256")):
                return  # an earlier try got this far
        log.append(record)  # synced by append
        _fsync_dir(self.record_path.parent)

    def _step_cut(self, offset: int) -> None:  # step 3
        with self.path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())

    def set_aside_torn_tail(self) -> dict[str, Any] | None:
        """1) the bytes to ``quarantine/<file>.<original length>.torn``, fsync; 2) their SHA-256,
        length, offset and time to the
        quarantine record, fsync; 3) only then cut the file back to the start of the torn line,
        fsync. A failure in 1) or 2) leaves
        the file exactly as it was and raises ``TornTailError``. ``None`` if there is no torn
        tail."""
        if self.torn is None:
            return None
        tail, offset = self.torn["bytes"], self.torn["offset"]
        original = self.torn["original_length"]
        name = f"{self.path.name}.{original}.torn"
        record = {
            "kind": "torn_tail",
            "file": self.path.name,
            "original_length": original,
            "offset": offset,
            "torn_length": len(tail),
            "torn_sha256": hashlib.sha256(tail).hexdigest(),
            "torn_file": f"{QUARANTINE_DIR}/{name}",
            "quarantined_at": _utc_now(),
        }
        try:
            self._step_write_evidence(self.quarantine_dir / name, tail)
            self._step_write_record(record)
        except Exception as error:  # noqa: BLE001
            raise TornTailError(
                f"{self.path.name}: the torn last line was not set aside "
                f"({type(error).__name__}: {error}); the file is unchanged"
            ) from error
        try:
            self._step_cut(offset)
        except Exception as error:  # noqa: BLE001
            raise TornTailError(
                f"{self.path.name}: the torn last line is kept but the file could not be cut "
                f"({type(error).__name__}: {error})"
            ) from error
        self.torn = None
        self._size = offset
        return record
