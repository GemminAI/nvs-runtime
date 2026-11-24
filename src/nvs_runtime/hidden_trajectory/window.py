"""Assemble one ``msr.trajectory-window`` 0.10.0 envelope from a stored observation.

The rows are the blob named by ``raw_ref.trajectory_id``. The caller passes the
blob reader, the space id, and the sealer. This module does not call the Gate
or the Formal Store, and it does not import MSR.

``envelope.build_envelope`` seals an ``nvs.hidden-trajectory`` for
``sensos-inference``. A window is a different schema and producer
(``nvs-runtime``, parent role ``source_trajectory``), so the envelope is
assembled here and sealed with the same ``seal(envelope, lookup)`` callback
that builder uses. IC-00 1.1.0 is ``envelope.IC00_VERSION``.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nvs_runtime.__version__ import __version__
from nvs_runtime.hidden_trajectory.envelope import IC00_VERSION
from nvs_runtime.hidden_trajectory.payload import MEDIA_TYPE, blob_uri, token_index_bytes

SCHEMA_ID = "nvs.hidden-trajectory"
SCHEMA_VERSION = "0.1.0"
WINDOW_SCHEMA = {"id": "msr.trajectory-window", "version": "0.10.0"}
WINDOW_RULE = "rows"
SPACE_RULE = "nvs.hidden-trajectory.space/1"
PRODUCER_NAME = "nvs-runtime"
PARENT_ROLE = "source_trajectory"
SOURCE = "T-hid"

_PLACEHOLDER_CONTENT = "cnt:" + "0" * 64
_PLACEHOLDER_ARTIFACT = "art:" + "0" * 64
_HEX64 = re.compile(r"[0-9a-f]{64}")


class WindowInputError(Exception):
    """A required value is missing or does not match. Nothing is filled in."""

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(f"{field}: {detail}")
        self.field = field
        self.detail = detail


@dataclass(frozen=True, slots=True)
class WindowArtifact:
    """A sealed window, the stored parent the Gate compares, and the bytes it reads.

    ``stored_parent`` is ``artifact_id -> {"envelope": observation}``.
    ``lookup`` is the IC-00 parent entry. ``blobs`` maps ``uri`` to bytes.
    ``origin`` is ``trajectory_id``, ``position``, ``length``, ``dimension``.
    """

    envelope: dict[str, Any]
    stored_parent: dict[str, Any]
    blobs: dict[str, bytes]
    lookup: dict[str, Any]
    origin: dict[str, Any]


def observation_origin(observation_envelope: Mapping[str, Any]) -> dict[str, Any]:
    """The parent fields the Gate reads back: trajectory id, length, position, dimension.

    ``dimension`` is the parent's value, the same type the window declares (W-1).
    """
    payload = _payload(observation_envelope)
    raw_ref = payload.get("raw_ref")
    if not isinstance(raw_ref, Mapping) or "trajectory_id" not in raw_ref:
        raise WindowInputError("raw_ref.trajectory_id", "not in the observation payload")
    for key in ("length", "position", "dimension"):
        if key not in payload:
            raise WindowInputError(key, "not in the observation payload")
    return {
        "trajectory_id": raw_ref["trajectory_id"],
        "length": payload["length"],
        "position": payload["position"],
        "dimension": payload["dimension"],
    }


def stored_parent_of(observation_envelope: Mapping[str, Any]) -> dict[str, Any]:
    """The Gate's ``stored`` map for this observation: one ``{"envelope": ...}``."""
    artifact_id = observation_envelope.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise WindowInputError("artifact_id", "the observation has no artifact_id")
    return {artifact_id: {"envelope": observation_envelope}}


def build_window_artifact(
    observation_envelope: Mapping[str, Any],
    blob_reader: Callable[[str], bytes],
    *,
    start: int = 0,
    length: int | None = None,
    producer_build: str,
    space_id_of: Callable[[Mapping[str, Any]], str],
    seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
) -> WindowArtifact:
    """The sealed window of ``observation_envelope``.

    ``blob_reader(sha256)`` returns the blob bytes. ``length is None`` takes the
    rows from ``start`` through the end of the origin. ``start + length`` must
    not pass ``origin.length``. ``producer_build`` is the IC-00 build; it is not
    a field of the observation. ``space_id_of(materials)`` and
    ``seal(envelope, lookup)`` are the existing sensos operations. The same
    arguments seal the same envelope: no clock and no random value is written.
    """
    if not isinstance(producer_build, str) or not producer_build:
        raise WindowInputError(
            "producer.build", "IC-00 requires a non-empty build; none was given"
        )
    _require_observation(observation_envelope)
    payload = _payload(observation_envelope)
    origin = observation_origin(observation_envelope)
    origin_length = _integer("length", origin["length"], minimum=1)
    dimension = origin["dimension"]
    if type(dimension) is not int or dimension < 1:
        raise WindowInputError("dimension", f"{dimension!r} is not an integer >= 1")
    start = _integer("start", start, minimum=0)
    if length is None:
        window_length = origin_length - start
    else:
        window_length = _integer("length", length, minimum=1)
    if window_length < 1 or start + window_length > origin_length:
        raise WindowInputError(
            "window",
            f"start {start} + length {window_length} exceeds origin.length {origin_length}",
        )
    _need(
        payload,
        "position",
        "layer",
        "source",
        "model_build_id",
        "compute_environment",
        "row_rule",
        "points",
        "token_index",
    )
    if payload["source"] != SOURCE:
        raise WindowInputError("source", f"observation source is {payload['source']!r}")
    points_ref = payload["points"]
    token_ref = payload["token_index"]
    if not isinstance(points_ref, Mapping) or "unit" not in points_ref:
        raise WindowInputError("points.unit", "not in the observation payload")
    if not isinstance(token_ref, Mapping) or "sha256" not in token_ref or "unit" not in token_ref:
        raise WindowInputError("token_index", "the observation blob ref has no sha256 or unit")
    if points_ref.get("shape") != [origin_length, dimension]:
        raise WindowInputError(
            "dimension",
            f"points.shape {points_ref.get('shape')!r} != [{origin_length}, {dimension}]",
        )

    trajectory_id = origin["trajectory_id"]
    if not isinstance(trajectory_id, str) or _HEX64.fullmatch(trajectory_id) is None:
        raise WindowInputError("raw_ref.trajectory_id", "not a sha256 hex string")
    points = _read(blob_reader, trajectory_id, "points")
    expected_bytes = origin_length * dimension * 4
    if len(points) != expected_bytes:
        raise WindowInputError(
            "points",
            f"{len(points)} bytes, expected {origin_length} x {dimension} x 4",
        )
    if hashlib.sha256(points).hexdigest() != trajectory_id:
        raise WindowInputError("sha256", "sha256 of the rows is not raw_ref.trajectory_id")
    values = struct.unpack(f"<{origin_length * dimension}f", points)
    if not all(math.isfinite(value) for value in values):
        raise WindowInputError("points", "a NaN or an infinity")

    token_bytes = _read(blob_reader, token_ref["sha256"], "token_index")
    if len(token_bytes) != origin_length * 8:
        raise WindowInputError(
            "token_index",
            f"{len(token_bytes)} bytes, expected {origin_length * 8}",
        )
    if hashlib.sha256(token_bytes).hexdigest() != token_ref["sha256"]:
        raise WindowInputError("sha256", "sha256 of token_index does not match the reference")
    if token_bytes != token_index_bytes(origin_length):
        raise WindowInputError("token_index", "the stored index is not 0 .. length-1")

    row = dimension * 4
    window_points = points[start * row : (start + window_length) * row]
    window_tokens = token_bytes[start * 8 : (start + window_length) * 8]
    point_sha = hashlib.sha256(window_points).hexdigest()
    token_sha = hashlib.sha256(window_tokens).hexdigest()
    policy_name = f"rows-{start}+{window_length}/1"
    derived_rule = f"{payload['row_rule']}+rows{start}+{window_length}/1"
    space_id = space_id_of(
        {
            "model_build_id": payload["model_build_id"],
            "layer": payload["layer"],
            "row_rule": derived_rule,
            "compute_environment": payload["compute_environment"],
            "dimension": dimension,
            "space_rule": SPACE_RULE,
        }
    )
    window_payload = {
        "window": {
            "rule": WINDOW_RULE,
            "start": start,
            "length": window_length,
            "content_sha256": point_sha,
        },
        "origin": {
            "trajectory_id": trajectory_id,
            "position": payload["position"],
            "length": origin_length,
        },
        "source": payload["source"],
        "layer": payload["layer"],
        "dimension": dimension,
        "model_build_id": payload["model_build_id"],
        "policy": {
            "name": policy_name,
            "sha256": hashlib.sha256(policy_name.encode("utf-8")).hexdigest(),
        },
        "points": {
            "uri": blob_uri(point_sha),
            "sha256": point_sha,
            "media_type": MEDIA_TYPE,
            "dtype": "float32",
            "shape": [window_length, dimension],
            "byte_order": "little",
            "order": "C",
            "unit": points_ref["unit"],
        },
        "token_index": {
            "uri": blob_uri(token_sha),
            "sha256": token_sha,
            "media_type": MEDIA_TYPE,
            "dtype": "int64",
            "shape": [window_length],
            "byte_order": "little",
            "order": "C",
            "quantization": None,
            "unit": token_ref["unit"],
        },
    }
    lookup = {
        observation_envelope["artifact_id"]: {
            "content_id": observation_envelope["content_id"],
            "seq": observation_envelope["seq"],
            "observation_id": observation_envelope["observation_id"],
        }
    }
    envelope = {
        "contract": {"id": "IC-00", "version": IC00_VERSION},
        "content_id": _PLACEHOLDER_CONTENT,
        "artifact_id": _PLACEHOLDER_ARTIFACT,
        "kind": "observation",
        "payload_schema": dict(WINDOW_SCHEMA),
        "payload": window_payload,
        "producer": {
            "name": PRODUCER_NAME,
            "version": __version__,
            "build": producer_build,
        },
        "session_id": observation_envelope["session_id"],
        "observation_id": observation_envelope["observation_id"],
        "event_span": None,
        "space_id": space_id,
        "parents": [
            {
                "role": PARENT_ROLE,
                "ordered": False,
                "artifact_ids": [observation_envelope["artifact_id"]],
            }
        ],
        "policy_refs": [],
        "calibration_ref": None,
        "rule_refs": [],
        "observed_at": None,
        "seq": observation_envelope["seq"] + 1,
        "produced_at": observation_envelope["produced_at"],
        "extensions": {},
    }
    sealed = seal(dict(envelope), lookup)
    return WindowArtifact(
        envelope=sealed,
        stored_parent=stored_parent_of(observation_envelope),
        blobs={
            window_payload["points"]["uri"]: window_points,
            window_payload["token_index"]["uri"]: window_tokens,
        },
        lookup=lookup,
        origin=origin,
    )


def _payload(observation_envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = observation_envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise WindowInputError("payload", "the observation has no payload")
    return payload


def _require_observation(observation_envelope: Mapping[str, Any]) -> None:
    schema = observation_envelope.get("payload_schema")
    if not isinstance(schema, Mapping) or (
        schema.get("id"),
        schema.get("version"),
    ) != (SCHEMA_ID, SCHEMA_VERSION):
        raise WindowInputError(
            "payload_schema", "the observation is not nvs.hidden-trajectory 0.1.0"
        )
    for key in ("artifact_id", "content_id", "session_id", "observation_id", "produced_at"):
        if not observation_envelope.get(key):
            raise WindowInputError(key, "not on the observation envelope")
    seq = observation_envelope.get("seq")
    if type(seq) is not int or seq < 0:
        raise WindowInputError("seq", f"{seq!r} is not an integer >= 0")


def _need(payload: Mapping[str, Any], *keys: str) -> None:
    for key in keys:
        if key not in payload:
            raise WindowInputError(key, "not in the observation payload")


def _integer(field: str, value: Any, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise WindowInputError(field, f"{value!r} is not an integer >= {minimum}")
    return value


def _read(blob_reader: Callable[[str], bytes], digest: str, field: str) -> bytes:
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        raise WindowInputError(field, "sha256 is not 64 hex characters")
    try:
        data = blob_reader(digest)
    except WindowInputError:
        raise
    except Exception as error:
        raise WindowInputError(field, f"the blob could not be read: {error}") from error
    if not isinstance(data, (bytes, bytearray)):
        raise WindowInputError(field, "the reader did not return bytes")
    return bytes(data)
