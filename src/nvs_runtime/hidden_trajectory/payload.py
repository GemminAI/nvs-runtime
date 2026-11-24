"""The payload ``nvs.hidden-trajectory`` 0.1.0 of one position of a RawObservationContract v2.

Every value is copied from the contract document (and, where the contract does not know it, from
the rows themselves); none is made up. The payload schema cannot hold some values a contract may
hold -- ``"unknown"`` for ``binding.n_in``, for ``final_norm_applied``, a runtime that is not a
mapping. Such a payload is not built: ``PayloadNotBuilt`` carries the reasons, and the two
``*_obstacles`` functions give the same reasons without raising.

Imports the standard library and ``nvs_runtime.core.errors`` only: no numpy, no nvs_edge, no
sensos contracts. The rows are taken as bytes (float32,
little-endian, C order -- the bytes whose SHA-256 is the RAW's ``trajectory_id``).
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Mapping
from typing import Any, Final

from nvs_runtime.core.errors import RuntimeError as NvsRuntimeError

PAYLOAD_SCHEMA_ID: Final = "nvs.hidden-trajectory"
PAYLOAD_SCHEMA_VERSION: Final = "0.1.0"
SOURCE: Final = "T-hid"
POLICY_ALL_ROWS: Final = "capture-all-rows/1"
MEDIA_TYPE: Final = "application/vnd.sensos.ndarray-raw"
UNKNOWN: Final = "unknown"
ALIGNMENT_DECIDES_NEXT: Final = "decides-next/1"
FINISH_REASONS: Final = ("eos", "length")
MAX_DIMENSION: Final = 65535
MAX_SAFE_INTEGER: Final = 2**53 - 1

CANNOT_EXPRESS: Final = "payload_cannot_express"
REASON_POINT_NOT_IN_CONTRACT: Final = "point_not_in_contract"
REASON_TRAJECTORY_ID_UNKNOWN: Final = "trajectory_id_unknown"
REASON_LENGTH_UNKNOWN: Final = "length_unknown"
REASON_ROWS_LENGTH: Final = "rows_bytes:length"
REASON_ROWS_TRAJECTORY_ID: Final = "rows_bytes:trajectory_id"
REASON_ROWS_STEPS: Final = "rows_bytes:steps"

_POINT: Final = re.compile(r"final_norm|(block_out|mlp_out):(0|[1-9][0-9]*)")
_HEX64: Final = re.compile(r"[0-9a-f]{64}")
_UNKNOWN_REASONS: Final = ("unknown", "not_recorded", "unavailable", "failed")
_HASH_FIELDS: Final = (
    "tokenizer_sha256",
    "chat_template_sha256",
    "prompt_ids_sha256",
    "gen_token_ids_sha256",
)
_CAPTURE_KEYS: Final = ("package", "version", "adapter")
_MISSING: Final = object()


class PayloadNotBuilt(NvsRuntimeError):
    """The payload of a position cannot be written from this contract; ``reasons`` says why."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        super().__init__("; ".join(reasons), code="PAYLOAD_NOT_BUILT")
        self.reasons = reasons


def _cannot(field: str) -> str:
    return f"{CANNOT_EXPRESS}:{field}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_in(value: Any, low: int, high: int) -> int | None:
    """``value`` if it is an integer (not a bool) in ``low..high``, else ``None``."""
    if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
        return value
    return None


def _section(document: Any, name: str) -> Mapping[str, Any]:
    section = document.get(name) if isinstance(document, Mapping) else None
    return section if isinstance(section, Mapping) else {}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pointer(*parts: str) -> str:
    return "".join("/" + part.replace("~", "~0").replace("/", "~1") for part in parts)


# ---- names ----------------------------------------------------------------------------------


def kind_of(point: str) -> str:
    """``block_out`` / ``mlp_out`` / ``final_norm`` of a canonical point name."""
    _require_point(point)
    return point.partition(":")[0]


def legacy_position(point: str) -> str:
    """The earlier name of a point: ``block_out:15`` -> ``L15.block_out``; ``final_norm`` as is."""
    _require_point(point)
    kind, _, layer = point.partition(":")
    return kind if kind == "final_norm" else f"L{layer}.{kind}"


def _require_point(point: Any) -> None:
    if not isinstance(point, str) or _POINT.fullmatch(point) is None:
        raise ValueError(f"not a canonical point name: {point!r}")


# ---- blobs ----------------------------------------------------------------------------------


def token_index_bytes(length: int) -> bytes:
    """``0, 1, .., length - 1`` as little-endian int64 (rule ``decides-next/1``)."""
    if not _is_int(length) or length < 0:
        raise ValueError(f"length must be an integer >= 0, not {length!r}")
    return struct.pack(f"<{length}q", *range(length))


def blob_uri(sha256: str) -> str:
    return f"blob:sha256:{sha256}"


def _points_ref(trajectory_id: str, length: int, dimension: int) -> dict[str, Any]:
    return {
        "uri": blob_uri(trajectory_id),
        "sha256": trajectory_id,
        "media_type": MEDIA_TYPE,
        "dtype": "float32",
        "shape": [length, dimension],
        "byte_order": "little",
        "order": "C",
        "unit": "activation",
    }


def _token_index_ref(length: int) -> dict[str, Any]:
    digest = _sha(token_index_bytes(length))
    return {
        "uri": blob_uri(digest),
        "sha256": digest,
        "media_type": MEDIA_TYPE,
        "dtype": "int64",
        "shape": [length],
        "byte_order": "little",
        "order": "C",
        "quantization": None,
        "unit": "token_position",
    }


# ---- what the payload cannot hold -----------------------------------------------------------


def session_obstacles(document: Any) -> tuple[str, ...]:
    """Reasons no payload can be written for any position of ``document`` (empty: none).

    These are values the payload schema has no way to write, not conditions of this module:
    ``binding`` needs its four values, the space needs ``model_build_id`` and
    ``compute_environment``, ``capture`` and ``runtime`` are mappings of strings.
    """
    binding, origin, raw = (_section(document, name) for name in ("binding", "origin", "raw"))
    out: list[str] = []
    if binding.get("alignment") != ALIGNMENT_DECIDES_NEXT:
        out.append(_cannot("binding.alignment"))
    for key in ("n_in", "n_gen"):
        if _int_in(binding.get(key), 1, MAX_SAFE_INTEGER) is None:
            out.append(_cannot(f"binding.{key}"))
    if binding.get("finish_reason") not in FINISH_REASONS:
        out.append(_cannot("binding.finish_reason"))
    for key in _HASH_FIELDS:
        if not isinstance(binding.get(key), str):
            out.append(_cannot(f"binding.{key}"))
    for key in ("model_build_id", "compute_environment"):
        value = origin.get(key)
        if not isinstance(value, str) or not value or value == UNKNOWN:
            out.append(_cannot(f"origin.{key}"))
    if not isinstance(origin.get("weights_sha256"), str):
        out.append(_cannot("origin.weights_sha256"))
    capture = origin.get("capture")
    if not isinstance(capture, Mapping):
        out.append(_cannot("origin.capture"))
    else:
        for key in _CAPTURE_KEYS:
            if not isinstance(capture.get(key), str) or not capture[key]:
                out.append(_cannot(f"origin.capture.{key}"))
    runtime = origin.get("runtime")
    if not isinstance(runtime, Mapping) or "backend" not in runtime:
        out.append(_cannot("origin.runtime"))
    else:
        for key, value in runtime.items():
            if not isinstance(key, str) or not isinstance(value, str):
                out.append(_cannot(f"origin.runtime.{key}"))
    if not isinstance(raw.get("state_dtype"), str):
        out.append(_cannot("raw.state_dtype"))
    return tuple(out)


def _position_of(document: Any, point: str) -> Mapping[str, Any] | None:
    positions = document.get("positions") if isinstance(document, Mapping) else None
    if not isinstance(positions, list):
        return None
    for position in positions:
        if isinstance(position, Mapping) and position.get("point") == point:
            return position
    return None


def _rows(
    position: Mapping[str, Any], rows_bytes: bytes | None, out: list[str]
) -> tuple[str | None, int | None]:
    """``(trajectory_id, length)`` of a position, from the contract and, if given, from its rows.

    Where the contract says ``"unknown"`` the rows decide; where both know, they must agree.
    """
    dimension = _int_in(position.get("dimension"), 1, MAX_DIMENSION)
    recorded_id = position.get("trajectory_id")
    stored = position.get("stored_range")
    recorded_steps = stored.get("steps") if isinstance(stored, Mapping) else _MISSING
    known_id = isinstance(recorded_id, str) and _HEX64.fullmatch(recorded_id) is not None
    length = _int_in(recorded_steps, 1, MAX_SAFE_INTEGER)
    if not known_id and recorded_id != UNKNOWN:
        out.append(_cannot("trajectory_id"))
    if length is None and recorded_steps != UNKNOWN:
        out.append(_cannot("stored_range.steps"))
    trajectory_id: str | None = recorded_id if known_id else None
    if rows_bytes is None:
        if recorded_id == UNKNOWN:
            out.append(REASON_TRAJECTORY_ID_UNKNOWN)
        if recorded_steps == UNKNOWN:
            out.append(REASON_LENGTH_UNKNOWN)
        return trajectory_id, length
    if dimension is None:
        return trajectory_id, length  # reported as the dimension's own obstacle
    width = 4 * dimension
    if not rows_bytes or len(rows_bytes) % width:
        out.append(REASON_ROWS_LENGTH)
        return trajectory_id, length
    actual_id, actual_steps = _sha(rows_bytes), len(rows_bytes) // width
    if known_id and recorded_id != actual_id:
        out.append(REASON_ROWS_TRAJECTORY_ID)
    if length is not None and length != actual_steps:
        out.append(REASON_ROWS_STEPS)
    return actual_id, actual_steps


def _position_obstacles(
    position: Mapping[str, Any], point: str, rows_bytes: bytes | None
) -> tuple[list[str], str | None, int | None]:
    out: list[str] = []
    if _POINT.fullmatch(point) is None:
        out.append(_cannot("point"))
    if _int_in(position.get("dimension"), 1, MAX_DIMENSION) is None:
        out.append(_cannot("dimension"))
    if not isinstance(position.get("row_rule"), str):
        out.append(_cannot("row_rule"))
    if not isinstance(position.get("final_norm_applied"), bool):
        out.append(_cannot("final_norm_applied"))
    space = position.get("space")
    layer = space.get("layer", _MISSING) if isinstance(space, Mapping) else _MISSING
    if _int_in(layer, 0, MAX_SAFE_INTEGER) is None and layer != "final_norm":
        out.append(_cannot("space.layer"))
    trajectory_id, length = _rows(position, rows_bytes, out)
    return out, trajectory_id, length


def position_obstacles(
    document: Any, point: str, *, rows_bytes: bytes | None = None
) -> tuple[str, ...]:
    """Reasons the payload of ``point`` cannot be written (empty: none). Session-wide reasons
    (``session_obstacles``) are not repeated here."""
    position = _position_of(document, point) if isinstance(point, str) else None
    if position is None:
        return (REASON_POINT_NOT_IN_CONTRACT,)
    return tuple(_position_obstacles(position, point, rows_bytes)[0])


# ---- the payload ----------------------------------------------------------------------------


def _unknown_reasons(document: Mapping[str, Any]) -> dict[str, str]:
    """The contract's own ``unknowns``: dotted field -> reason."""
    listed = document.get("unknowns")
    out: dict[str, str] = {}
    if isinstance(listed, list):
        for entry in listed:
            if isinstance(entry, Mapping) and isinstance(entry.get("field"), str):
                reason = entry.get("reason")
                out[entry["field"]] = reason if reason in _UNKNOWN_REASONS else UNKNOWN
    return out


def _unknowns(payload: Mapping[str, Any], reasons: Mapping[str, str]) -> list[dict[str, str]]:
    """``[{field, reason}]`` for every ``"unknown"`` the payload may hold, in order of ``field``.

    ``places`` maps each JSON Pointer of the payload to the contract field it was copied from.
    """
    provenance = payload["capture_provenance"]
    places = {_pointer("state_dtype"): ("raw.state_dtype", payload["state_dtype"])}
    places[_pointer("capture_provenance", "weights_sha256")] = (
        "origin.weights_sha256",
        provenance["weights_sha256"],
    )
    for key in _HASH_FIELDS:
        places[_pointer("capture_provenance", key)] = (f"binding.{key}", provenance[key])
    for group in ("capture", "runtime"):
        for key, value in provenance[group].items():
            places[_pointer("capture_provenance", group, key)] = (f"origin.{group}.{key}", value)
    found = [
        {"field": pointer, "reason": reasons.get(source, UNKNOWN)}
        for pointer, (source, value) in places.items()
        if value == UNKNOWN
    ]
    return sorted(found, key=lambda entry: entry["field"].encode("utf-8"))


def build_payload(
    document: Mapping[str, Any],
    point: str,
    *,
    contract_ref: Mapping[str, Any] | None,
    rows_bytes: bytes | None = None,
    copied: bool = False,
    policy_name: str = POLICY_ALL_ROWS,
) -> dict[str, Any]:
    """The payload of one position. ``PayloadNotBuilt`` if a value cannot be written.

    ``contract_ref`` is ``{"name", "version", "contract_sha256"}`` or ``None`` and is written as
    given. ``rows_bytes`` are the position's rows; they are needed only where the contract does
    not know ``trajectory_id`` or ``stored_range.steps``, and are compared where it does.
    ``copied`` says whether the rows were copied into the store the envelope goes to.
    """
    reasons = list(session_obstacles(document))
    position = _position_of(document, point) if isinstance(point, str) else None
    trajectory_id: str | None = None
    length: int | None = None
    if position is None:
        reasons.append(REASON_POINT_NOT_IN_CONTRACT)
    else:
        found, trajectory_id, length = _position_obstacles(position, point, rows_bytes)
        reasons.extend(found)
    if reasons or position is None or trajectory_id is None or length is None:
        raise PayloadNotBuilt(tuple(reasons))
    binding, origin, raw = document["binding"], document["origin"], document["raw"]
    dimension = position["dimension"]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "position": legacy_position(point),
        "point": point,
        "kind": kind_of(point),
        "layer": position["space"]["layer"],
        "length": length,
        "dimension": dimension,
        "points": _points_ref(trajectory_id, length, dimension),
        "token_index": _token_index_ref(length),
        "raw_ref": {"trajectory_id": trajectory_id, "copied": copied},
        "row_rule": position["row_rule"],
        "binding": {
            "alignment": binding["alignment"],
            "n_in": binding["n_in"],
            "n_gen": binding["n_gen"],
            "finish_reason": binding["finish_reason"],
        },
        "final_norm_applied": position["final_norm_applied"],
        "state_dtype": raw["state_dtype"],
        "model_build_id": origin["model_build_id"],
        "compute_environment": origin["compute_environment"],
        "policy": {"name": policy_name, "sha256": _sha(policy_name.encode("utf-8"))},
        "capture_provenance": {
            "weights_sha256": origin["weights_sha256"],
            **{key: binding[key] for key in _HASH_FIELDS},
            "capture": {key: origin["capture"][key] for key in _CAPTURE_KEYS},
            "runtime": dict(origin["runtime"]),
            "model_source": None,
        },
        "contract_ref": None if contract_ref is None else dict(contract_ref),
    }
    payload["unknowns"] = _unknowns(payload, _unknown_reasons(document))
    return payload
