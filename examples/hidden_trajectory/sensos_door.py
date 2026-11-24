"""The caller's side of the Gate child: the two functions ``build_envelopes`` needs, and the Gate.

A THIN COMPATIBILITY LAYER since U6T-1c (DC-T1-1): the Gate is nvs-runtime's own now
(``nvs_runtime.hidden_trajectory.gate_client``: ``GateProcess`` starts the child script in
``GATE_PY``, ``GateClient`` builds the ``GateVerdict``), and this class keeps the operations that
the U6-4 tests use, unchanged. nvs-runtime's interpreter never imports the sensos contracts (parts
do not import each other, by design); what crosses is JSON.

    door = SensosDoor.from_environment()          # GATE_PY, SENSOS_TOOLS
    result = build_envelopes(document, check, ids=..., space_id_of=door.space_id_of, seal=door.seal)
    answer = door.check(result.envelopes[point], blobs=result.blobs, contract_facts=check.facts)
    verdict = door.verdict(result.envelopes[point], lookup=..., stored=..., blobs=..., ...)

``check`` answers in the form of the U6-4 door (``l1_codes``, ``payload``, ...); ``verdict`` is
``GateClient.verdict`` (a ``GateVerdict``). Needs ``nvs_runtime`` on the path.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from nvs_runtime.hidden_trajectory.gate_client import (
    GATE_INTERNAL_ERROR,
    GateClient,
    GateInternalError,
    GateProcess,
    GateUnavailable,
    GateVerdict,
)

CLI = Path(__file__).resolve().parent / "gate_nvs_cli.py"
__all__ = ["CLI", "GATE_INTERNAL_ERROR", "GateInternalError", "SensosDoor"]


@dataclass(frozen=True)
class SensosDoor:
    gate_py: Path
    sensos_tools: Path

    @classmethod
    def from_environment(cls) -> SensosDoor:
        missing = [name for name in ("GATE_PY", "SENSOS_TOOLS") if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"environment variables not set: {', '.join(missing)}")
        return cls(Path(os.environ["GATE_PY"]), Path(os.environ["SENSOS_TOOLS"]))

    @cached_property
    def client(self) -> GateClient:
        """The formal Gate (one per door: it builds its checker bundle once)."""
        return GateClient(self.gate_py, self.sensos_tools)

    def call(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return GateProcess(self.gate_py, self.sensos_tools).call(request)
        except GateInternalError:
            raise
        except GateUnavailable as error:  # the U6-4 door called every other failure this
            raise GateInternalError(GATE_INTERNAL_ERROR, str(error)) from error

    def space_id_of(self, materials: Mapping[str, Any]) -> str:
        return str(self.call({"op": "hid_space_id", "materials": dict(materials)})["space_id"])

    def seal(self, envelope: dict[str, Any], lookup: Mapping[str, Any]) -> dict[str, Any]:
        sealed = self.call({"op": "seal", "envelope": envelope, "lookup": dict(lookup)})["envelope"]
        assert isinstance(sealed, dict)
        return sealed

    def session_id(
        self, *, spec: str, item_id: str, condition: str, replicate: int, rng_seed: int
    ) -> str:
        request = {"op": "session_id", "spec": spec, "item_id": item_id, "condition": condition}
        return str(self.call({**request, "replicate": replicate, "rng_seed": rng_seed})["id"])

    def observation_id(self, session_id: str, ordinal: int = 0) -> str:
        request = {"op": "observation_id", "session_id": session_id, "ordinal": ordinal}
        return str(self.call(request)["id"])

    def check(
        self,
        envelope: Mapping[str, Any],
        *,
        lookup: Mapping[str, Any] | None = None,
        stored: Mapping[str, Any] | None = None,
        blobs: Mapping[str, bytes] | None = None,
        contract_facts: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``{l1_codes, l1_messages, payload, window}`` (see ``_gate_child.py``)."""
        return self.call(
            {
                "op": "check",
                "envelope": dict(envelope),
                "lookup": dict(lookup or {}),
                "stored": dict(stored or {}),
                "blobs": {u: base64.b64encode(d).decode() for u, d in (blobs or {}).items()},
                "contract_facts": None if contract_facts is None else dict(contract_facts),
            }
        )

    def verdict(
        self,
        envelope: Mapping[str, Any],
        *,
        lookup: Mapping[str, Any] | None = None,
        stored: Mapping[str, Any] | None = None,
        blobs: Mapping[str, bytes] | None = None,
        contract_facts: Mapping[str, Any] | None = None,
    ) -> GateVerdict:
        """``GateClient.verdict``: the ``GateVerdict`` of ``envelope``."""
        return self.client.verdict(
            envelope, lookup=lookup, stored=stored, blobs=blobs, contract_facts=contract_facts
        )
