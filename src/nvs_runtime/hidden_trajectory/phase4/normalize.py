"""The comparison of a repeated capture (pre-registration v0.8, section 2): three items.

(1) the sha256 of the generated token id sequence;
(2) the sha256 of the normalised K causal log (every ``hmac`` key removed at every depth, the
    order of arrays kept, ``json.dumps`` with ``sort_keys=True``, ``separators=(",", ":")``,
    ``ensure_ascii=False``, ``allow_nan=False``, UTF-8);
(3) the board records: the chain (``prev_hash``), the CRC and the ``record_hash`` of every record
    one by one, and the ``key_independent_sha256`` of AC-66-3 (the body without ``hmac``). The
    ``hmac`` is not compared (the key differs on every run).

A NaN or an Infinity is a ``NormalizationError``: nothing is replaced silently and no verdict is
made. The identity and the version of the normaliser, the Python version, the ``json`` module and
the sha256 of the source of the functions are recorded (``normalizer_record``); two runs are
compared with the same implementation only.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from nvs_runtime.hidden_trajectory.canonical import canonical_bytes, key_independent

NORMALIZER_ID: Final = "causal-log-normalizer/1"
HMAC_KEY: Final = "hmac"


class NormalizationError(ValueError):
    """A value the normalisation refuses to write (NaN, Infinity) or a file that is not JSON."""


@dataclass(frozen=True, slots=True)
class Normalized:
    data: bytes
    sha256: str
    removed_hmac_keys: int


def strip_hmac(node: Any) -> tuple[Any, int]:
    """``node`` without any ``hmac`` key, at every depth (dictionaries inside lists included), and
    the
    number of keys removed. The order of every list is kept."""
    if isinstance(node, dict):
        removed = 0
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == HMAC_KEY:
                removed += 1
                continue
            cleaned, count = strip_hmac(value)
            out[key] = cleaned
            removed += count
        return out, removed
    if isinstance(node, list):
        items: list[Any] = []
        removed = 0
        for value in node:
            cleaned, count = strip_hmac(value)
            items.append(cleaned)
            removed += count
        return items, removed
    return node, 0


def normalize_causal_log(data: bytes) -> Normalized:
    try:
        document = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise NormalizationError(f"not JSON: {type(error).__name__}") from error
    stripped, removed = strip_hmac(document)
    try:
        text = json.dumps(
            stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except ValueError as error:  # NaN or Infinity
        raise NormalizationError(f"a value cannot be written: {error}") from error
    encoded = text.encode("utf-8")
    return Normalized(encoded, hashlib.sha256(encoded).hexdigest(), removed)


def normalizer_record() -> dict[str, Any]:
    source = inspect.getsource(strip_hmac) + inspect.getsource(normalize_causal_log)
    return {
        "normalizer_id": NORMALIZER_ID,
        "python": platform.python_version(),
        "json_module": "standard library json",
        "json_module_version": getattr(json, "__version__", platform.python_version()),
        "functions": ["strip_hmac", "normalize_causal_log"],
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def token_ids_sha256(ids: Sequence[int]) -> str:
    """sha256 of the ids as little-endian int64 (the way the archive stores an id list)."""
    return hashlib.sha256(struct.pack(f"<{len(ids)}q", *[int(i) for i in ids])).hexdigest()


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """What one run leaves for the comparison."""

    generated_ids: Sequence[int]
    causal_log: bytes
    board_records: Sequence[Mapping[str, Any]]


def compare_repeat(first: RunArtifacts, again: RunArtifacts) -> dict[str, Any]:
    """The three comparisons, each reported whatever the others say."""
    out: dict[str, Any] = {"normalizer": normalizer_record()}
    a, b = token_ids_sha256(first.generated_ids), token_ids_sha256(again.generated_ids)
    out["item_1_generated_ids"] = {
        "n_first": len(first.generated_ids),
        "n_again": len(again.generated_ids),
        "sha256_first": a,
        "sha256_again": b,
        "same": a == b,
    }
    try:
        n_first, n_again = (
            normalize_causal_log(first.causal_log),
            normalize_causal_log(again.causal_log),
        )
        out["item_2_causal_log"] = {
            "sha256_first": n_first.sha256,
            "sha256_again": n_again.sha256,
            "removed_hmac_keys_first": n_first.removed_hmac_keys,
            "removed_hmac_keys_again": n_again.removed_hmac_keys,
            "same": n_first.sha256 == n_again.sha256,
        }
    except NormalizationError as error:
        out["item_2_causal_log"] = {"normalization_error": str(error), "same": None}
    wires_first, wires_again = list(first.board_records), list(again.board_records)
    per_record = []
    for index, (x, y) in enumerate(zip(wires_first, wires_again, strict=False)):
        per_record.append(
            {
                "seq": index,
                "prev_hash_same": x.get("prev_hash") == y.get("prev_hash"),
                "crc_same": x.get("crc") == y.get("crc"),
                "record_hash_same": x.get("record_hash") == y.get("record_hash"),
                "hmac_same_not_compared": x.get("hmac") == y.get("hmac"),
            }
        )
    key_a = hashlib.sha256(canonical_bytes(key_independent(wires_first))).hexdigest()
    key_b = hashlib.sha256(canonical_bytes(key_independent(wires_again))).hexdigest()
    out["item_3_board"] = {
        "n_records_first": len(wires_first),
        "n_records_again": len(wires_again),
        "per_record": per_record,
        "all_chain_crc_record_hash_same": len(wires_first) == len(wires_again)
        and all(
            r["prev_hash_same"] and r["crc_same"] and r["record_hash_same"] for r in per_record
        ),
        "key_independent_sha256_first": key_a,
        "key_independent_sha256_again": key_b,
        "key_independent_same": key_a == key_b,
    }
    item2 = out["item_2_causal_log"]["same"]
    out["all_same"] = bool(
        out["item_1_generated_ids"]["same"]
        and item2 is True
        and out["item_3_board"]["all_chain_crc_record_hash_same"]
        and out["item_3_board"]["key_independent_same"]
    )
    return out
