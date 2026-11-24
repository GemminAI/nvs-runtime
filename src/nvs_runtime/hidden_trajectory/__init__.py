"""The ``nvs.hidden-trajectory`` path of nvs-runtime: from a RawObservationContract v2 to the Gate.

This package calls the owners of the rules and reimplements none of them: the contract's
structure and hash are checked by ``nvs_observation.contract_check``; the space id, the artifact
ids and the checks are sensos's (the contracts Gate).
"""

from __future__ import annotations

from nvs_runtime.hidden_trajectory.envelope import (
    BuildResult,
    EnvelopeIds,
    ParentRef,
    build_envelope,
    build_envelopes,
    contract_ref_of,
    session_refusals,
)
from nvs_runtime.hidden_trajectory.payload import (
    PAYLOAD_SCHEMA_ID,
    PAYLOAD_SCHEMA_VERSION,
    POLICY_ALL_ROWS,
    PayloadNotBuilt,
    blob_uri,
    build_payload,
    kind_of,
    legacy_position,
    position_obstacles,
    session_obstacles,
    token_index_bytes,
)
from nvs_runtime.hidden_trajectory.record import (
    RECORD_VERSION,
    capture_block,
    empty_generation_record,
    with_gate_result,
)

__all__ = [
    "PAYLOAD_SCHEMA_ID",
    "PAYLOAD_SCHEMA_VERSION",
    "POLICY_ALL_ROWS",
    "RECORD_VERSION",
    "BuildResult",
    "EnvelopeIds",
    "ParentRef",
    "PayloadNotBuilt",
    "blob_uri",
    "build_envelope",
    "build_envelopes",
    "build_payload",
    "capture_block",
    "contract_ref_of",
    "empty_generation_record",
    "kind_of",
    "legacy_position",
    "position_obstacles",
    "session_obstacles",
    "session_refusals",
    "token_index_bytes",
    "with_gate_result",
]
