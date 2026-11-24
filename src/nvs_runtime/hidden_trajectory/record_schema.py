"""The strict schema of the capture record ``nvs.capture-record/1`` and its checker
(design U6-5 v0.5, 5.2 "version and readers", DC-U65-7 = (a)).

The record is a Python dictionary made by ``record.py``. This module holds its schema as a
dictionary (JSON Schema draft 2020-12, so that a reader outside this package can use it with any
validator after ``json.dumps``) and ``validate_record``, which checks the same rules without a
validator library. The rules:

* No key outside the schema is allowed anywhere (``additionalProperties: false``).
* ``capture_extra`` is the only optional key of a record. A record without it is valid.
* The values inside ``capture_extra.values`` are any JSON; only the key set is fixed.

The constants below are copies, not imports: this module is meant to be read on its own, and a
test compares them with ``record.py`` and ``capture_extra.py``.

This module imports the standard library only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

RECORD_VERSION: Final = "nvs.capture-record/1"
CAPTURE_EXTRA_VERSION: Final = "nvs.capture-extra/1"

OUTCOMES: Final = ("enveloped", "partly_enveloped", "empty_generation", "refused", "not_enveloped")
POSITION_STATUSES: Final = ("enveloped", "not_built")
CONTRACT_STATUSES: Final = ("VERIFIED", "NOT_VERIFIED", "REJECTED", "none")
HASH_SOURCES: Final = (
    "recomputed_matches_recorded",
    "recomputed_no_recorded",
    "recorded_only",
    "none",
)
CAPTURE_KEYS: Final = (
    "n_in",
    "n_gen",
    "finish_reason",
    "points",
    "model_build_id",
    "compute_environment",
    "adapter",
    "prompt_ids_sha256",
    "gen_token_ids_sha256",
)
EXTRA_VALUE_KEYS: Final = (
    "sampler",
    "max_tokens",
    "chat_template_kwargs",
    "capture_unknowns",
    "adapter_notes",
    "origin_facts_unknowns",
    "raw_encoding",
    "recorded_at",
    "origin",
)
#: ``capture_extra.unknowns[].reason``: the four of ``capture_extra.py`` and ``failed`` (the reason
#: of a caller's own line, ``origin.edge_version`` when ``source_build_id`` raised).
EXTRA_UNKNOWN_REASONS: Final = (
    "not_json_value",
    "not_json_key",
    "too_deep",
    "too_large",
    "failed",
)
MAX_EXTRA_UNKNOWNS: Final = 64
MAX_FIELD_CHARS: Final = 256
MAX_DETAIL_CHARS: Final = 128
HEX64_PATTERN: Final = "^[0-9a-f]{64}$"
_HEX64: Final = re.compile(r"[0-9a-f]{64}")

_STRING_OR_NULL: Final = {"type": ["string", "null"]}

#: The longest ``store_refusal.detail`` (the saver cuts a longer one).
MAX_REFUSAL_DETAIL_CHARS: Final = 512

#: U6T-2b': what became of the envelope of a position in the Formal Store. Optional keys of a
#: position; a record made before the save (``run_capture_session`` alone) has none of them.
_STORED: Final[dict[str, Any]] = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "required": ["artifact_id", "hekb_seq", "hekb_hash"],
    "properties": {
        "artifact_id": {"type": "string", "minLength": 1},
        "hekb_seq": {"type": "integer", "minimum": 0},
        "hekb_hash": {"type": "string", "pattern": HEX64_PATTERN},
    },
}
_STORE_REFUSAL: Final[dict[str, Any]] = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "required": ["code", "detail"],
    "properties": {
        "code": {"type": "string", "minLength": 1},
        "detail": {"type": "string", "maxLength": MAX_REFUSAL_DETAIL_CHARS},
    },
}

#: U6-6a: the K path of a position (design U6-6 v0.1, 4 and 14). Optional key of a position.
K_STATUSES: Final = ("measured", "not_measured", "not_eligible")
K_VERIFY_CHECKS: Final = ("structure", "session_id", "crc", "record_hash", "hmac", "chain")
K_KEYS: Final = (
    "status",
    "reason",
    "reason_detail",
    "input_dimension",
    "trajectory_id",
    "compute_environment",
    "edge_session_id",
    "projection",
    "policies",
    "space_id",
    "space",
    "n_records",
    "token_index_range",
    "board_records_sha256",
    "causal_log_sha256",
    "chain_head",
    "edge_input_cast_exact",
    "integrity",
    "verify_failure",
)
#: The verifier a K record names (the published value of ``integrity.verifier``).
VERIFIER: Final = "nvs_edge.verify.verify_record"
K_INTEGRITY_KEYS: Final = ("chain_and_crc", "hmac", "hmac_key", "verifier", "verifier_build")
K_INTEGRITY_FIXED: Final = {
    "chain_and_crc": "recomputable",
    "hmac": "verified_at_run_time",
    "hmac_key": "ephemeral_not_kept",
    "verifier": VERIFIER,
}
_HEX_OR_NULL: Final = {"anyOf": [{"type": "string", "pattern": HEX64_PATTERN}, {"type": "null"}]}
_PRE_TOKEN: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["token_index", "status", "reason", "anchors"],
    "properties": {
        "token_index": {"type": "integer", "minimum": 0},
        "status": {"enum": ["measured", "not_measured"]},
        "reason": {"type": ["string", "null"]},
        "anchors": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["config_bootstrap", "dynamic", "manual", "total"],
            "properties": {
                k: {"type": "integer", "minimum": 0}
                for k in ("config_bootstrap", "dynamic", "manual", "total")
            },
        },
    },
}
_K_PATH: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(K_KEYS),
    "properties": {
        "status": {"enum": list(K_STATUSES)},
        "reason": {"type": ["string", "null"]},
        "reason_detail": {"type": ["string", "null"], "maxLength": MAX_DETAIL_CHARS},
        "input_dimension": {"type": ["integer", "null"]},
        "trajectory_id": _HEX_OR_NULL,
        "compute_environment": {"type": ["string", "null"]},
        "edge_session_id": {"type": ["string", "null"]},
        "projection": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["id", "p_sha256", "w_sha256"],
            "properties": {
                "id": {"type": "string"},
                "p_sha256": {"type": "string", "pattern": HEX64_PATTERN},
                "w_sha256": {"type": "string", "pattern": HEX64_PATTERN},
            },
        },
        "policies": {"type": ["object", "null"]},
        "space_id": {"type": ["string", "null"]},
        "space": {"type": ["object", "null"]},
        "n_records": {"type": ["integer", "null"], "minimum": 0},
        "token_index_range": {
            "type": ["array", "null"],
            "items": {"type": "integer", "minimum": 0},
            "minItems": 2,
            "maxItems": 2,
        },
        "board_records_sha256": _HEX_OR_NULL,
        "causal_log_sha256": _HEX_OR_NULL,
        "chain_head": _HEX_OR_NULL,
        "edge_input_cast_exact": {"type": ["boolean", "null"]},
        "integrity": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": list(K_INTEGRITY_KEYS),
            "properties": {
                **{k: {"const": v} for k, v in K_INTEGRITY_FIXED.items()},
                "verifier_build": {"type": "string", "minLength": 1},
            },
        },
        "verify_failure": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["seq", "check", "detail"],
            "properties": {
                "seq": {"type": ["integer", "null"], "minimum": 0},
                "check": {"enum": list(K_VERIFY_CHECKS)},
                "detail": {"type": "string", "maxLength": MAX_DETAIL_CHARS},
            },
        },
        "pre_cycle_anchors": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source", "configured_bootstrap_anchors", "per_token"],
            "properties": {
                "source": {"type": "string", "minLength": 1},
                "configured_bootstrap_anchors": {"type": ["integer", "null"], "minimum": 0},
                "per_token": {"type": "array", "items": _PRE_TOKEN},
            },
        },
    },
}

_GATE: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "violations", "unverifiable"],
    "properties": {
        "status": {"type": "string", "minLength": 1},
        "violations": {"type": "array"},
        "unverifiable": {"type": "array"},
    },
}

_POSITION: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "reasons", "artifact_id", "gate"],
    "properties": {
        "status": {"enum": list(POSITION_STATUSES)},
        "reasons": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "artifact_id": {"type": ["string", "null"]},
        "gate": {"$ref": "#/$defs/gate"},
        "stored": _STORED,
        "verdict_sha256": {
            "anyOf": [{"type": "string", "pattern": HEX64_PATTERN}, {"type": "null"}]
        },
        "store_refusal": _STORE_REFUSAL,
        "k_path": _K_PATH,
    },
    "if": {"properties": {"status": {"const": "enveloped"}}, "required": ["status"]},
    "then": {"properties": {"artifact_id": {"type": "string", "minLength": 1}}},
    "else": {"properties": {"artifact_id": {"type": "null"}}},
}

_CAPTURE: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(CAPTURE_KEYS),
    "properties": {
        "n_in": {"type": ["integer", "string", "null"]},
        "n_gen": {"type": ["integer", "string", "null"]},
        "finish_reason": _STRING_OR_NULL,
        "points": {"type": "array", "items": {"type": "string"}},
        "model_build_id": _STRING_OR_NULL,
        "compute_environment": _STRING_OR_NULL,
        "adapter": _STRING_OR_NULL,
        "prompt_ids_sha256": _STRING_OR_NULL,
        "gen_token_ids_sha256": _STRING_OR_NULL,
    },
}

_CONTRACT: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "contract_sha256", "hash_source"],
    "properties": {
        "status": {"enum": list(CONTRACT_STATUSES)},
        "contract_sha256": {
            "anyOf": [{"type": "string", "pattern": HEX64_PATTERN}, {"type": "null"}]
        },
        "hash_source": {"enum": list(HASH_SOURCES)},
    },
}

_EXTRA_UNKNOWN: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["field", "reason", "detail"],
    "properties": {
        "field": {"type": "string", "minLength": 1, "maxLength": MAX_FIELD_CHARS},
        "reason": {"enum": list(EXTRA_UNKNOWN_REASONS)},
        "detail": {"type": "string", "maxLength": MAX_DETAIL_CHARS},
        "sha256": {"type": "string", "pattern": HEX64_PATTERN},
    },
    "if": {"properties": {"reason": {"const": "too_large"}}, "required": ["reason"]},
    "then": {"required": ["sha256"]},
    "else": {"not": {"required": ["sha256"]}},
}

_CAPTURE_EXTRA: Final[dict[str, Any]] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "values", "unknowns"],
    "properties": {
        "version": {"const": CAPTURE_EXTRA_VERSION},
        "values": {
            "type": "object",
            "additionalProperties": False,
            "properties": {key: True for key in EXTRA_VALUE_KEYS},
        },
        "unknowns": {
            "type": "array",
            "maxItems": MAX_EXTRA_UNKNOWNS,
            "items": {"$ref": "#/$defs/extra_unknown"},
        },
    },
}

#: The schema of ``nvs.capture-record/1``. ``capture_extra`` is not in ``required``.
RECORD_SCHEMA: Final[dict[str, Any]] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://nvs.invalid/schemas/nvs.capture-record/1",
    "title": RECORD_VERSION,
    "type": "object",
    "additionalProperties": False,
    "required": [
        "record_version",
        "outcome",
        "session_id",
        "observation_id",
        "contract_session_id",
        "capture",
        "contract",
        "refusals",
        "positions",
    ],
    "properties": {
        "record_version": {"const": RECORD_VERSION},
        "outcome": {"enum": list(OUTCOMES)},
        "session_id": {"type": "string"},
        "observation_id": {"type": "string"},
        "contract_session_id": _STRING_OR_NULL,
        "capture": {"$ref": "#/$defs/capture"},
        "contract": {"$ref": "#/$defs/contract"},
        "refusals": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "positions": {"type": "object", "additionalProperties": {"$ref": "#/$defs/position"}},
        "capture_extra": {"$ref": "#/$defs/capture_extra"},
    },
    "$defs": {
        "capture": _CAPTURE,
        "contract": _CONTRACT,
        "gate": _GATE,
        "position": _POSITION,
        "extra_unknown": _EXTRA_UNKNOWN,
        "capture_extra": _CAPTURE_EXTRA,
    },
}


# ---- the checker ------------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_json(value: object, depth: int = 0) -> bool:
    """Whether ``value`` is JSON as ``json.dumps`` would take it (str keys, finite numbers)."""
    if depth > 64:
        return False
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, (list, tuple)):
        return all(_is_json(item, depth + 1) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _is_json(v, depth + 1) for k, v in value.items())
    return False


def _keys(
    path: str,
    node: Mapping[str, Any],
    required: tuple[str, ...],
    optional: tuple[str, ...],
    out: list[str],
) -> bool:
    ok = True
    for key in required:
        if key not in node:
            out.append(f"{path}: missing key {key!r}")
            ok = False
    for key in node:
        if key not in required and key not in optional:
            out.append(f"{path}: unknown key {key!r}")
            ok = False
    return ok


def _string(
    path: str, value: object, out: list[str], *, nullable: bool = False, min_length: int = 0
) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        out.append(f"{path}: must be a string{' or null' if nullable else ''}")
    elif len(value) < min_length:
        out.append(f"{path}: must not be empty")


def _array(path: str, value: object, out: list[str]) -> list[Any] | None:
    if not isinstance(value, list):
        out.append(f"{path}: must be an array")
        return None
    return value


def _check_capture(node: object, out: list[str]) -> None:
    if not isinstance(node, Mapping):
        out.append("capture: must be an object")
        return
    _keys("capture", node, CAPTURE_KEYS, (), out)
    for key in ("n_in", "n_gen"):
        if key in node and not (
            node[key] is None or isinstance(node[key], str) or _is_int(node[key])
        ):
            out.append(f"capture.{key}: must be an integer, a string or null")
    for key in (
        "finish_reason",
        "model_build_id",
        "compute_environment",
        "adapter",
        "prompt_ids_sha256",
        "gen_token_ids_sha256",
    ):
        if key in node:
            _string(f"capture.{key}", node[key], out, nullable=True)
    if "points" in node:
        points = _array("capture.points", node["points"], out)
        for index, point in enumerate(points or ()):
            _string(f"capture.points[{index}]", point, out)


def _check_contract(node: object, out: list[str]) -> None:
    if not isinstance(node, Mapping):
        out.append("contract: must be an object")
        return
    _keys("contract", node, ("status", "contract_sha256", "hash_source"), (), out)
    if "status" in node and node["status"] not in CONTRACT_STATUSES:
        out.append(f"contract.status: must be one of {list(CONTRACT_STATUSES)}")
    if "hash_source" in node and node["hash_source"] not in HASH_SOURCES:
        out.append(f"contract.hash_source: must be one of {list(HASH_SOURCES)}")
    digest = node.get("contract_sha256")
    if "contract_sha256" in node and digest is not None:
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            out.append("contract.contract_sha256: must be 64 lowercase hex digits or null")


def _check_gate(path: str, node: object, out: list[str]) -> None:
    if not isinstance(node, Mapping):
        out.append(f"{path}: must be an object")
        return
    _keys(path, node, ("status", "violations", "unverifiable"), (), out)
    if "status" in node:
        _string(f"{path}.status", node["status"], out, min_length=1)
    for key in ("violations", "unverifiable"):
        if key in node:
            _array(f"{path}.{key}", node[key], out)


def _check_position(path: str, node: object, out: list[str]) -> None:
    if not isinstance(node, Mapping):
        out.append(f"{path}: must be an object")
        return
    _keys(
        path,
        node,
        ("status", "reasons", "artifact_id", "gate"),
        ("stored", "verdict_sha256", "store_refusal", "k_path"),
        out,
    )
    status = node.get("status")
    if "status" in node and status not in POSITION_STATUSES:
        out.append(f"{path}.status: must be one of {list(POSITION_STATUSES)}")
    if "reasons" in node:
        reasons = _array(f"{path}.reasons", node["reasons"], out)
        for index, reason in enumerate(reasons or ()):
            _string(f"{path}.reasons[{index}]", reason, out, min_length=1)
    if "artifact_id" in node:
        if status == "enveloped":
            _string(f"{path}.artifact_id", node["artifact_id"], out, min_length=1)
        elif node["artifact_id"] is not None:
            out.append(f"{path}.artifact_id: must be null unless the position is enveloped")
    if "gate" in node:
        _check_gate(f"{path}.gate", node["gate"], out)
    _check_store_fields(path, node, out)
    if "k_path" in node:
        _check_k_path(f"{path}.k_path", node["k_path"], out)


def _hex_or_null(path: str, value: object, out: list[str]) -> None:
    if value is not None and (not isinstance(value, str) or _HEX64.fullmatch(value) is None):
        out.append(f"{path}: must be 64 lowercase hex digits or null")


def _int_or_null(path: str, value: object, out: list[str]) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        out.append(f"{path}: must be a non-negative integer or null")


def _check_pre_cycle_anchors(path: str, node: object, out: list[str]) -> None:
    """The anchors the kernel session held before each cycle (optional; absent in older records).
    A token either has its counts or says why it has none: no count stands in for a missing one."""
    if not isinstance(node, Mapping):
        out.append(f"{path}: must be an object")
        return
    if not _keys(path, node, ("source", "configured_bootstrap_anchors", "per_token"), (), out):
        return
    _string(f"{path}.source", node["source"], out, min_length=1)
    _int_or_null(f"{path}.configured_bootstrap_anchors", node["configured_bootstrap_anchors"], out)
    tokens = node["per_token"]
    if not isinstance(tokens, list):
        out.append(f"{path}.per_token: must be an array")
        return
    for position, item in enumerate(tokens):
        here = f"{path}.per_token[{position}]"
        if not isinstance(item, Mapping):
            out.append(f"{here}: must be an object")
            continue
        if not _keys(here, item, ("token_index", "status", "reason", "anchors"), (), out):
            continue
        index = item["token_index"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            out.append(f"{here}.token_index: must be a non-negative integer")
        anchors = item["anchors"]
        if item["status"] == "measured":
            if item["reason"] is not None:
                out.append(f"{here}.reason: must be null when measured")
            if not isinstance(anchors, Mapping):
                out.append(f"{here}.anchors: a measured token has its counts")
            elif _keys(
                here + ".anchors",
                anchors,
                ("config_bootstrap", "dynamic", "manual", "total"),
                (),
                out,
            ):
                for name, value in anchors.items():
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        out.append(f"{here}.anchors.{name}: must be a non-negative integer")
        elif item["status"] == "not_measured":
            if anchors is not None:
                out.append(f"{here}.anchors: must be null when not measured (no stand-in value)")
            if not (isinstance(item["reason"], str) and item["reason"]):
                out.append(f"{here}.reason: a token without counts says why")
        else:
            out.append(f"{here}.status: must be 'measured' or 'not_measured'")


def _check_k_path(path: str, node: object, out: list[str]) -> None:
    """The K path of a position (U6-6a): the keys, the status rules, and what ``integrity`` says."""
    if not isinstance(node, Mapping):
        out.append(f"{path}: must be an object")
        return
    if not _keys(path, node, K_KEYS, ("pre_cycle_anchors",), out):
        return
    if "pre_cycle_anchors" in node:
        _check_pre_cycle_anchors(f"{path}.pre_cycle_anchors", node["pre_cycle_anchors"], out)
    status = node["status"]
    if status not in K_STATUSES:
        out.append(f"{path}.status: must be one of {list(K_STATUSES)}")
    reason = node["reason"]
    if status == "measured" and reason is not None:
        out.append(f"{path}.reason: must be null when the K path is measured")
    if status != "measured" and not (isinstance(reason, str) and reason):
        out.append(f"{path}.reason: a K path that is not measured says why")
    detail = node["reason_detail"]
    if detail is not None and (not isinstance(detail, str) or len(detail) > MAX_DETAIL_CHARS):
        out.append(
            f"{path}.reason_detail: must be a string of at most {MAX_DETAIL_CHARS} characters"
        )
    _int_or_null(f"{path}.input_dimension", node["input_dimension"], out)
    _int_or_null(f"{path}.n_records", node["n_records"], out)
    for key in ("trajectory_id", "board_records_sha256", "causal_log_sha256", "chain_head"):
        _hex_or_null(f"{path}.{key}", node[key], out)
    for key in ("compute_environment", "edge_session_id", "space_id"):
        if node[key] is not None and not isinstance(node[key], str):
            out.append(f"{path}.{key}: must be a string or null")
    cast = node["edge_input_cast_exact"]
    if cast is not None and not isinstance(cast, bool):
        out.append(f"{path}.edge_input_cast_exact: must be a boolean or null")
    span = node["token_index_range"]
    if span is not None and not (
        isinstance(span, list)
        and len(span) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in span)
    ):
        out.append(f"{path}.token_index_range: must be [first, last] or null")
    projection = node["projection"]
    if projection is not None:
        if not isinstance(projection, Mapping):
            out.append(f"{path}.projection: must be an object or null")
        elif _keys(f"{path}.projection", projection, ("id", "p_sha256", "w_sha256"), (), out):
            _string(f"{path}.projection.id", projection["id"], out, min_length=1)
            _hex_or_null(f"{path}.projection.p_sha256", projection["p_sha256"], out)
            _hex_or_null(f"{path}.projection.w_sha256", projection["w_sha256"], out)
    for key in ("policies", "space"):
        if node[key] is not None and not isinstance(node[key], Mapping):
            out.append(f"{path}.{key}: must be an object or null")
    integrity = node["integrity"]
    if integrity is not None:
        if not isinstance(integrity, Mapping):
            out.append(f"{path}.integrity: must be an object or null")
        elif _keys(f"{path}.integrity", integrity, K_INTEGRITY_KEYS, (), out):
            for key, wanted in K_INTEGRITY_FIXED.items():
                if integrity[key] != wanted:
                    out.append(f"{path}.integrity.{key}: must be {wanted!r}")
            _string(
                f"{path}.integrity.verifier_build", integrity["verifier_build"], out, min_length=1
            )
    if status == "measured":
        if integrity is None:
            out.append(f"{path}.integrity: a measured K path says how it was checked")
        if node["board_records_sha256"] is None or node["chain_head"] is None:
            out.append(f"{path}: a measured K path names its board records and chain head")
        if node["verify_failure"] is not None:
            out.append(f"{path}.verify_failure: must be null when the K path is measured")
    failure = node["verify_failure"]
    if failure is not None:
        if not isinstance(failure, Mapping):
            out.append(f"{path}.verify_failure: must be an object or null")
        elif _keys(f"{path}.verify_failure", failure, ("seq", "check", "detail"), (), out):
            _int_or_null(f"{path}.verify_failure.seq", failure["seq"], out)
            if failure["check"] not in K_VERIFY_CHECKS:
                out.append(f"{path}.verify_failure.check: must be one of {list(K_VERIFY_CHECKS)}")
            _string(f"{path}.verify_failure.detail", failure["detail"], out)
            detail = failure["detail"]
            if isinstance(detail, str) and len(detail) > MAX_DETAIL_CHARS:
                out.append(
                    f"{path}.verify_failure.detail: longer than {MAX_DETAIL_CHARS} characters"
                )


def _check_store_fields(path: str, node: Mapping[str, Any], out: list[str]) -> None:
    """``stored``, ``verdict_sha256`` and ``store_refusal`` (all optional)."""
    stored = node.get("stored")
    if stored is not None:
        if not isinstance(stored, Mapping):
            out.append(f"{path}.stored: must be an object or null")
        else:
            _keys(f"{path}.stored", stored, ("artifact_id", "hekb_seq", "hekb_hash"), (), out)
            if "artifact_id" in stored:
                _string(f"{path}.stored.artifact_id", stored["artifact_id"], out, min_length=1)
                if node.get("status") != "enveloped" or stored["artifact_id"] != node.get(
                    "artifact_id"
                ):
                    out.append(f"{path}.stored.artifact_id: must be the position's own artifact_id")
            seq = stored.get("hekb_seq")
            if "hekb_seq" in stored and (
                not isinstance(seq, int) or isinstance(seq, bool) or seq < 0
            ):
                out.append(f"{path}.stored.hekb_seq: must be a non-negative integer")
            digest = stored.get("hekb_hash")
            if "hekb_hash" in stored and (
                not isinstance(digest, str) or _HEX64.fullmatch(digest) is None
            ):
                out.append(f"{path}.stored.hekb_hash: must be 64 lowercase hex digits")
    verdict = node.get("verdict_sha256")
    if verdict is not None and (not isinstance(verdict, str) or _HEX64.fullmatch(verdict) is None):
        out.append(f"{path}.verdict_sha256: must be 64 lowercase hex digits or null")
    refusal = node.get("store_refusal")
    if refusal is not None:
        if not isinstance(refusal, Mapping):
            out.append(f"{path}.store_refusal: must be an object or null")
        else:
            _keys(f"{path}.store_refusal", refusal, ("code", "detail"), (), out)
            if "code" in refusal:
                _string(f"{path}.store_refusal.code", refusal["code"], out, min_length=1)
            if "detail" in refusal:
                _string(f"{path}.store_refusal.detail", refusal["detail"], out)
                detail = refusal["detail"]
                if isinstance(detail, str) and len(detail) > MAX_REFUSAL_DETAIL_CHARS:
                    limit = MAX_REFUSAL_DETAIL_CHARS
                    out.append(f"{path}.store_refusal.detail: longer than {limit} characters")
    if stored is not None and refusal is not None:
        out.append(f"{path}: stored and store_refusal cannot both be set")


def _check_extra_unknown(path: str, node: object, out: list[str]) -> None:
    if not isinstance(node, Mapping):
        out.append(f"{path}: must be an object")
        return
    _keys(path, node, ("field", "reason", "detail"), ("sha256",), out)
    if "field" in node:
        _string(f"{path}.field", node["field"], out, min_length=1)
        if isinstance(node["field"], str) and len(node["field"]) > MAX_FIELD_CHARS:
            out.append(f"{path}.field: longer than {MAX_FIELD_CHARS} characters")
    reason = node.get("reason")
    if "reason" in node and reason not in EXTRA_UNKNOWN_REASONS:
        out.append(f"{path}.reason: must be one of {list(EXTRA_UNKNOWN_REASONS)}")
    if "detail" in node:
        _string(f"{path}.detail", node["detail"], out)
        if isinstance(node["detail"], str) and len(node["detail"]) > MAX_DETAIL_CHARS:
            out.append(f"{path}.detail: longer than {MAX_DETAIL_CHARS} characters")
    if "sha256" in node:
        digest = node["sha256"]
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            out.append(f"{path}.sha256: must be 64 lowercase hex digits")
        if "reason" in node and reason != "too_large":
            out.append(f"{path}.sha256: only a too_large line has a sha256")
    elif reason == "too_large":
        out.append(f"{path}: a too_large line needs a sha256")


def _check_capture_extra(node: object, out: list[str]) -> None:
    if not isinstance(node, Mapping):
        out.append("capture_extra: must be an object")
        return
    _keys("capture_extra", node, ("version", "values", "unknowns"), (), out)
    if "version" in node and node["version"] != CAPTURE_EXTRA_VERSION:
        out.append(f"capture_extra.version: must be {CAPTURE_EXTRA_VERSION!r}")
    values = node.get("values")
    if "values" in node:
        if not isinstance(values, Mapping):
            out.append("capture_extra.values: must be an object")
        else:
            for key, item in values.items():
                if key not in EXTRA_VALUE_KEYS:
                    out.append(f"capture_extra.values: unknown key {key!r}")
                elif not _is_json(item):
                    out.append(f"capture_extra.values.{key}: must be JSON")
    if "unknowns" in node:
        unknowns = _array("capture_extra.unknowns", node["unknowns"], out)
        if unknowns is not None:
            if len(unknowns) > MAX_EXTRA_UNKNOWNS:
                out.append(f"capture_extra.unknowns: more than {MAX_EXTRA_UNKNOWNS} items")
            for index, item in enumerate(unknowns):
                _check_extra_unknown(f"capture_extra.unknowns[{index}]", item, out)


def validate_record(record: object) -> list[str]:
    """The violations of ``RECORD_SCHEMA`` in ``record``; an empty list means it is valid.

    The same rules as the schema dictionary, without a validator library. The order of the
    messages is the order of the checks, so the same record gives the same list.
    """
    out: list[str] = []
    if not isinstance(record, Mapping):
        return ["record: must be an object"]
    required = tuple(RECORD_SCHEMA["required"])
    _keys("record", record, required, ("capture_extra",), out)
    if "record_version" in record and record["record_version"] != RECORD_VERSION:
        out.append(f"record.record_version: must be {RECORD_VERSION!r}")
    if "outcome" in record and record["outcome"] not in OUTCOMES:
        out.append(f"record.outcome: must be one of {list(OUTCOMES)}")
    for key in ("session_id", "observation_id"):
        if key in record:
            _string(f"record.{key}", record[key], out)
    if "contract_session_id" in record:
        _string("record.contract_session_id", record["contract_session_id"], out, nullable=True)
    if "capture" in record:
        _check_capture(record["capture"], out)
    if "contract" in record:
        _check_contract(record["contract"], out)
    if "refusals" in record:
        refusals = _array("record.refusals", record["refusals"], out)
        for index, reason in enumerate(refusals or ()):
            _string(f"record.refusals[{index}]", reason, out, min_length=1)
    if "positions" in record:
        positions = record["positions"]
        if not isinstance(positions, Mapping):
            out.append("record.positions: must be an object")
        else:
            for point, entry in positions.items():
                if not isinstance(point, str):
                    out.append(f"record.positions: key {point!r} must be a string")
                else:
                    _check_position(f"record.positions[{point!r}]", entry, out)
    if "capture_extra" in record:
        _check_capture_extra(record["capture_extra"], out)
    return out
