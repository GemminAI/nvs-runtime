"""Hidden-state replay provider — reads recorded layer-32 hidden states token by token.

Same capability shape a live-LLM provider would offer (``input_type="replay_request"``
=> ``output_type="hidden_state"``): this provider is a recorded-data double for that
contract, not a special case the planner favors. Backed by
the experiment's input directory (manifest.json + hidden/<session_id>/
hidden.npy, non-Locked sessions only -- the manifest never lists a Locked session_id,
so this provider has no Locked content to accidentally read).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nvs_runtime.core.errors import ProviderError
from nvs_runtime.core.models import Capability
from nvs_runtime.providers.base import BaseProvider

#: axis-1 index into hidden.npy's (n_steps, 8, 4096) array for layer 32 -- the 8 saved
#: layers are [4, 8, 12, 16, 20, 24, 28, 32] (MSD0001 v1.1 meta.json), so index 7 is
#: layer 32, matching the NVS-Kernel Edge/TRJ003 convention (layer_index=7,
#: layer_number=32).
LAYER_32_INDEX = 7


class ReplayError(ProviderError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="REPLAY_ERROR")


class HiddenStateReplayProvider(BaseProvider):
    """Replays one recorded session's hidden states, one token at a time.

    ``invoke("replay.hidden_state", {"session_id", "token_index"})`` returns that
    token's 4096-D layer-32 hidden state as a plain list of floats (not a numpy array:
    ``InvokeResult``/events may be serialized, and a live-LLM provider offering this same
    capability would not produce a numpy array either).
    """

    kind = "replay"

    def __init__(
        self,
        provider_id: str = "replay",
        *,
        manifest_path: Path,
        hidden_root: Path | None = None,
        layer_index: int = LAYER_32_INDEX,
    ) -> None:
        super().__init__(provider_id)
        self._manifest_path = Path(manifest_path)
        self._hidden_root = (
            Path(hidden_root) if hidden_root else self._manifest_path.parent / "hidden"
        )
        self._layer_index = layer_index
        self._manifest = json.loads(self._manifest_path.read_text())["sessions"]
        self._mmaps: dict[str, np.memmap] = {}
        self.register_capability(
            Capability(
                id="replay.hidden_state",
                name="Replay One Token's Hidden State",
                provider_id=self.provider_id,
                input_type="replay_request",
                output_type="hidden_state",
                tags=["replay", "msd0001"],
                cost_units=1,
                latency_ms=1,
            )
        )
        self.register_capability(
            Capability(
                id="replay.session_info",
                name="Session Length And Metadata",
                provider_id=self.provider_id,
                input_type="replay_request",
                output_type="session_info",
                tags=["replay", "msd0001"],
                cost_units=1,
                latency_ms=1,
            )
        )

    def session_ids(self) -> list[str]:
        return list(self._manifest.keys())

    def _hidden(self, session_id: str) -> np.memmap:
        if session_id not in self._manifest:
            raise ReplayError(f"unknown (or Locked, never listed here) session_id: {session_id!r}")
        if session_id not in self._mmaps:
            path = self._hidden_root / session_id / "hidden.npy"
            if not path.is_file():
                raise ReplayError(f"session {session_id}: hidden.npy not found at {path}")
            self._mmaps[session_id] = np.load(path, mmap_mode="r")
        return self._mmaps[session_id]

    async def invoke(self, capability_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "")
        if capability_id == "replay.session_info":
            hidden = self._hidden(session_id)
            n_steps = int(hidden.shape[0])
            return {
                "type": "session_info",
                "session_id": session_id,
                "n_steps": n_steps,
                "split": self._manifest[session_id]["split"],
                "layer_index": self._layer_index,
            }
        if capability_id == "replay.hidden_state":
            token_index = int(payload.get("token_index", -1))
            hidden = self._hidden(session_id)
            n_steps = int(hidden.shape[0])
            if not 0 <= token_index < n_steps:
                raise ReplayError(
                    f"session {session_id}: token_index {token_index} out of range [0, {n_steps})"
                )
            vector = np.asarray(hidden[token_index, self._layer_index, :], dtype=np.float64)
            return {
                "type": "hidden_state",
                "session_id": session_id,
                "token_index": token_index,
                "n_steps": n_steps,
                "is_last": token_index == n_steps - 1,
                "hidden_state": vector.tolist(),
            }
        from nvs_runtime.core.errors import CapabilityNotFoundError

        raise CapabilityNotFoundError(capability_id)

    async def shutdown(self) -> None:
        self._mmaps.clear()
        await super().shutdown()


__all__ = ["LAYER_32_INDEX", "HiddenStateReplayProvider", "ReplayError"]
