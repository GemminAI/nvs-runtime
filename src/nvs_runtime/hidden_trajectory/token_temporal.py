"""Assemble one ``nvs.token-temporal`` 0.1.0 envelope from a stored observation.

The id lists are the bytes ``ids_reader`` returns for the observation's
``prompt_ids_sha256`` and ``gen_token_ids_sha256`` (the same contract as
``capture_archive.CaptureArchive.read_ids``: bytes, or ``None`` when the
archive has no file). This module does not call the Gate or the Formal Store,
and it does not import MSR or sensos-inference.

``space_id`` is null. IC-00 1.1.0 allows null, and no space id is given to this
assembler. ``produced_at`` is the parent's: there is no clock and no random
value. A segment whose ``n`` is 0 is kept. ``chat_template_sha256`` is copied
when it is null or a sha256; any other recorded value, including
``not_applicable``, is refused rather than rewritten.

nvs-runtime has no RFC 8785 implementation (``capture_extra.canonical_json``
is a different encoding). The ids below are sha256 of the same JCS rules as
``contracts/jcs.py``. The checks in the tests recompute them separately.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from nvs_runtime.hidden_trajectory.envelope import IC00_VERSION
from nvs_runtime.hidden_trajectory.payload import MEDIA_TYPE, blob_uri

SCHEMA_ID = "nvs.hidden-trajectory"
SCHEMA_VERSION = "0.1.0"
TOKEN_SCHEMA = {"id": "nvs.token-temporal", "version": "0.1.0"}
PRODUCER_NAME = "sensos-inference"
# The parent's role name is not decided. It lives only here.
PARENT_ROLE = "source_observation"
KIND = "observation"
UNIT = "token_id"
SEGMENT_DOMAIN = "nvs.token-segment/1"
AXIS_DOMAIN = "nvs.token-axis/1"
UNKNOWN = "unknown"

_PLACEHOLDER_CONTENT = "cnt:" + "0" * 64
_PLACEHOLDER_ARTIFACT = "art:" + "0" * 64
_HEX64 = re.compile(r"[0-9a-f]{64}")
_MAX_SAFE = 2**53 - 1
_SHORT = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


class TokenTemporalInputError(Exception):
    """A required value is missing or does not match. Nothing is returned."""

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(f"{field}: {detail}")
        self.field = field
        self.detail = detail


@dataclass(frozen=True, slots=True)
class TokenTemporalArtifact:
    """A sealed token-temporal envelope, the parent the Gate compares, and the id bytes.

    ``stored_parent`` is ``artifact_id -> {"envelope": observation}``.
    ``lookup`` is the IC-00 parent entry. ``blobs`` maps ``uri`` to the bytes
    ``ids_reader`` returned, without re-encoding them.
    """

    envelope: dict[str, Any]
    stored_parent: dict[str, Any]
    blobs: dict[str, bytes]
    lookup: dict[str, Any]


def build_token_temporal_artifact(
    observation_envelope: Mapping[str, Any],
    ids_reader: Callable[[str], bytes | None],
    *,
    seal: Callable[[dict[str, Any], Mapping[str, Any]], dict[str, Any]],
) -> TokenTemporalArtifact:
    """The sealed token-temporal artifact of ``observation_envelope``.

    ``ids_reader(sha256)`` returns the archive bytes, or ``None``. ``seal``
    writes ``content_id`` and ``artifact_id``. The same arguments seal the
    same envelope.
    """
    _require_observation(observation_envelope)
    payload = _payload(observation_envelope)
    binding = _section(payload, "binding")
    provenance = _section(payload, "capture_provenance")
    n_in = _count(binding, "n_in")
    n_gen = _count(binding, "n_gen")
    tokenizer = _sha_field(provenance, "tokenizer_sha256")
    chat_template = _chat_template(provenance)
    prompt_sha = _sha_field(provenance, "prompt_ids_sha256")
    gen_sha = _sha_field(provenance, "gen_token_ids_sha256")
    model_build_id = _model_build_id(payload)
    version, build = _producer_version_build(observation_envelope)

    prompt_bytes = _read_ids(ids_reader, prompt_sha, n_in, "prompt_ids")
    gen_bytes = _read_ids(ids_reader, gen_sha, n_gen, "gen_token_ids")
    segments = [
        _segment("input", n_in, tokenizer, prompt_sha),
        _segment("generated", n_gen, tokenizer, gen_sha),
    ]
    token_payload = {
        "axis_id": _axis_id(segments),
        "model_build_id": model_build_id,
        "tokenizer_sha256": tokenizer,
        "chat_template_sha256": chat_template,
        "segments": segments,
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
        "kind": KIND,
        "payload_schema": dict(TOKEN_SCHEMA),
        "payload": token_payload,
        "producer": {"name": PRODUCER_NAME, "version": version, "build": build},
        "session_id": observation_envelope["session_id"],
        "observation_id": observation_envelope["observation_id"],
        "event_span": None,
        "space_id": None,
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
    blobs: dict[str, bytes] = {}
    blobs[blob_uri(prompt_sha)] = prompt_bytes
    blobs[blob_uri(gen_sha)] = gen_bytes
    return TokenTemporalArtifact(
        envelope=sealed,
        stored_parent=_stored_parent(observation_envelope),
        blobs=blobs,
        lookup=lookup,
    )


def _segment(origin: str, n: int, tokenizer: str, ids_sha256: str) -> dict[str, Any]:
    segment_id = _sha256_hex(
        _jcs(
            {
                "domain": SEGMENT_DOMAIN,
                "tokenizer_sha256": tokenizer,
                "ids_sha256": ids_sha256,
                "n": n,
            }
        )
    )
    return {
        "origin": origin,
        "n": n,
        "segment_id": segment_id,
        "ids": {
            "uri": blob_uri(ids_sha256),
            "sha256": ids_sha256,
            "media_type": MEDIA_TYPE,
            "dtype": "int64",
            "shape": [n],
            "byte_order": "little",
            "order": "C",
            "quantization": None,
            "unit": UNIT,
        },
    }


def _axis_id(segments: list[dict[str, Any]]) -> str:
    return _sha256_hex(
        _jcs(
            {
                "domain": AXIS_DOMAIN,
                "segments": [
                    {"origin": segment["origin"], "segment_id": segment["segment_id"]}
                    for segment in segments
                ],
            }
        )
    )


def _payload(observation_envelope: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = observation_envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise TokenTemporalInputError("payload", "the observation has no payload")
    return payload


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = payload.get(name)
    if not isinstance(section, Mapping):
        raise TokenTemporalInputError(name, "not recorded on the observation payload")
    return section


def _require_observation(observation_envelope: Mapping[str, Any]) -> None:
    schema = observation_envelope.get("payload_schema")
    if not isinstance(schema, Mapping) or (
        schema.get("id"),
        schema.get("version"),
    ) != (SCHEMA_ID, SCHEMA_VERSION):
        raise TokenTemporalInputError(
            "payload_schema", "the observation is not nvs.hidden-trajectory 0.1.0"
        )
    for key in ("artifact_id", "content_id", "session_id", "observation_id", "produced_at"):
        if not observation_envelope.get(key):
            raise TokenTemporalInputError(key, "not on the observation envelope")
    seq = observation_envelope.get("seq")
    if type(seq) is not int or seq < 0 or seq + 1 > _MAX_SAFE:
        raise TokenTemporalInputError("seq", f"{seq!r} is not an integer >= 0 whose successor fits")


def _producer_version_build(observation_envelope: Mapping[str, Any]) -> tuple[str, str]:
    producer = observation_envelope.get("producer")
    if not isinstance(producer, Mapping):
        raise TokenTemporalInputError("producer", "not recorded on the observation envelope")
    version = producer.get("version")
    build = producer.get("build")
    if not isinstance(version, str) or not version:
        raise TokenTemporalInputError("producer.version", "not recorded on the observation")
    if not isinstance(build, str) or not build:
        raise TokenTemporalInputError("producer.build", "not recorded on the observation")
    return version, build


def _count(binding: Mapping[str, Any], key: str) -> int:
    field = f"binding.{key}"
    if key not in binding:
        raise TokenTemporalInputError(field, "not recorded")
    value = binding[key]
    if value == UNKNOWN or value is None:
        raise TokenTemporalInputError(field, f"{value!r} is unknown or not recorded")
    if type(value) is not int or value < 0 or value > _MAX_SAFE:
        raise TokenTemporalInputError(field, f"{value!r} is not an integer >= 0")
    return value


def _sha_field(provenance: Mapping[str, Any], key: str) -> str:
    field = f"capture_provenance.{key}"
    if key not in provenance:
        raise TokenTemporalInputError(field, "not recorded")
    value = provenance[key]
    if value == UNKNOWN or value is None or not isinstance(value, str):
        raise TokenTemporalInputError(field, f"{value!r} is unknown or not recorded")
    if _HEX64.fullmatch(value) is None:
        raise TokenTemporalInputError(field, f"{value!r} is not a sha256 hex string")
    return value


def _chat_template(provenance: Mapping[str, Any]) -> str | None:
    field = "capture_provenance.chat_template_sha256"
    if "chat_template_sha256" not in provenance:
        raise TokenTemporalInputError(field, "not recorded")
    value = provenance["chat_template_sha256"]
    if value == UNKNOWN:
        raise TokenTemporalInputError(field, f"{value!r} is unknown or not recorded")
    if value is None:
        return None
    if isinstance(value, str) and _HEX64.fullmatch(value) is not None:
        return value
    raise TokenTemporalInputError(field, f"{value!r} is not a sha256 or null")


def _model_build_id(payload: Mapping[str, Any]) -> str:
    if "model_build_id" not in payload:
        raise TokenTemporalInputError("model_build_id", "not recorded")
    value = payload["model_build_id"]
    if value == UNKNOWN or not isinstance(value, str) or not value:
        raise TokenTemporalInputError("model_build_id", f"{value!r} is unknown or not recorded")
    return value


def _read_ids(
    ids_reader: Callable[[str], bytes | None], digest: str, n: int, field: str
) -> bytes:
    try:
        data = ids_reader(digest)
    except TokenTemporalInputError:
        raise
    except Exception as error:
        raise TokenTemporalInputError(field, f"the ids could not be read: {error}") from error
    if data is None:
        raise TokenTemporalInputError(field, "the archive has no bytes for this sha256")
    if not isinstance(data, (bytes, bytearray)):
        raise TokenTemporalInputError(field, "the reader did not return bytes")
    raw = data if isinstance(data, bytes) else bytes(data)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise TokenTemporalInputError(field, "sha256 of the bytes is not the key")
    if len(raw) != 8 * n:
        raise TokenTemporalInputError(field, f"{len(raw)} bytes, expected {8 * n}")
    if n and any(value < 0 for value in struct.unpack(f"<{n}q", raw)):
        raise TokenTemporalInputError(field, "an id is negative")
    return raw


def _stored_parent(observation_envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {observation_envelope["artifact_id"]: {"envelope": observation_envelope}}


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jcs(value: Any) -> bytes:
    """RFC 8785 bytes, the same rules as ``contracts/jcs.py`` (integers only)."""
    return _canon(value).encode("utf-8")


def _canon(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        raise TokenTemporalInputError("jcs", f"float {value!r} is not allowed")
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE:
            raise TokenTemporalInputError("jcs", f"{value} outside +-(2^53-1)")
        return str(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(_canon(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise TokenTemporalInputError("jcs", f"object key {key!r} is not a string")
        keys = sorted(value, key=lambda key: key.encode("utf-16-be", "surrogatepass"))
        return "{" + ",".join(_string(key) + ":" + _canon(value[key]) for key in keys) + "}"
    raise TokenTemporalInputError("jcs", f"{type(value).__name__} cannot be canonicalized")


def _string(text: str) -> str:
    out = ['"']
    for char in text:
        code = ord(char)
        if 0xD800 <= code <= 0xDFFF:
            raise TokenTemporalInputError("jcs", "lone surrogate in a string")
        if char in _SHORT:
            out.append(_SHORT[char])
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)
