"""``run_capture_session`` through the sensos door: the caller's side of U6-5d.

    door = SensosDoor.from_environment()                    # GATE_PY, SENSOS_TOOLS
    pair = AdapterSession(adapter).capture_with_origin(messages, request, ...)
    result = run_through_door(door, pair, contract_session_id="...", synthetic=False,
                              spec=..., item_id=..., produced_at="2026-11-16T00:00:00Z")

The ids come from the sensos rules (``door.session_id``, ``door.observation_id``); the space id,
the sealing and the Gate are the door's. A thin layer like ``sensos_door.py``. Imports no sensos
code; only what nvs-runtime's ``src`` offers.
"""

from __future__ import annotations

from typing import Any

from nvs_observation.adapter_session import CapturedWithOrigin
from nvs_runtime.hidden_trajectory.envelope import EnvelopeIds
from nvs_runtime.hidden_trajectory.run import RunResult, run_capture_session


def run_through_door(
    door: Any,
    captured: CapturedWithOrigin,
    *,
    contract_session_id: str,
    synthetic: bool,
    spec: str,
    item_id: str,
    produced_at: str,
    condition: str = "run",
    replicate: int = 0,
    rng_seed: int = 0,
    call_gate: bool = True,
    **kwargs: Any,
) -> RunResult:
    """Make the session and observation ids with the door, then ``run_capture_session`` with the
    door's ``space_id_of``, ``seal`` and (unless ``call_gate`` is false) ``check`` as the Gate."""
    session_id = door.session_id(
        spec=spec, item_id=item_id, condition=condition, replicate=replicate, rng_seed=rng_seed
    )
    observation_id = door.observation_id(session_id, 0)
    sequence = iter(range(1, 10**6))
    ids = EnvelopeIds(session_id, observation_id, produced_at, lambda: next(sequence))
    return run_capture_session(
        captured,
        contract_session_id=contract_session_id,
        synthetic=synthetic,
        ids=ids,
        space_id_of=door.space_id_of,
        seal=door.seal,
        gate=door.verdict if call_gate else None,
        **kwargs,
    )
