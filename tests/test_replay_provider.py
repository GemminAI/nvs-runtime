"""HiddenStateReplayProvider: mmap-backed, token-ordered hidden-state replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nvs_runtime.providers.replay import HiddenStateReplayProvider, ReplayError


@pytest.fixture
def manifest_root(tmp_path: Path) -> Path:
    session_id = "p11-fake-0001"
    n_steps = 3
    hidden = np.arange(n_steps * 8 * 4096, dtype=np.float32).reshape(n_steps, 8, 4096)
    (tmp_path / "hidden" / session_id).mkdir(parents=True)
    np.save(tmp_path / "hidden" / session_id / "hidden.npy", hidden)
    manifest = {
        "sessions": {
            session_id: {
                "split": "benign_fit", "bytes": hidden.nbytes, "sha256": "x",
                "shape": list(hidden.shape), "dtype": "float32",
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


async def test_replays_position_0_and_layer_32_in_order(manifest_root: Path) -> None:
    provider = HiddenStateReplayProvider(manifest_path=manifest_root / "manifest.json")
    session_id = "p11-fake-0001"

    info = await provider.invoke("replay.session_info", {"session_id": session_id})
    assert info["n_steps"] == 3 and info["layer_index"] == 7

    first = await provider.invoke(
        "replay.hidden_state", {"session_id": session_id, "token_index": 0}
    )
    assert first["token_index"] == 0 and first["is_last"] is False
    full = np.arange(3 * 8 * 4096, dtype=np.float32).reshape(3, 8, 4096)
    assert np.allclose(first["hidden_state"], full[0, 7, :])

    last = await provider.invoke(
        "replay.hidden_state", {"session_id": session_id, "token_index": 2}
    )
    assert last["is_last"] is True


async def test_out_of_range_token_index_raises(manifest_root: Path) -> None:
    provider = HiddenStateReplayProvider(manifest_path=manifest_root / "manifest.json")
    with pytest.raises(ReplayError, match="out of range"):
        await provider.invoke(
            "replay.hidden_state", {"session_id": "p11-fake-0001", "token_index": 99}
        )


async def test_unknown_session_id_raises(manifest_root: Path) -> None:
    provider = HiddenStateReplayProvider(manifest_path=manifest_root / "manifest.json")
    with pytest.raises(ReplayError, match="unknown"):
        await provider.invoke(
            "replay.hidden_state", {"session_id": "not-in-manifest", "token_index": 0}
        )


async def test_hidden_state_is_a_plain_list_not_a_numpy_array(manifest_root: Path) -> None:
    provider = HiddenStateReplayProvider(manifest_path=manifest_root / "manifest.json")
    result = await provider.invoke(
        "replay.hidden_state", {"session_id": "p11-fake-0001", "token_index": 1}
    )
    assert isinstance(result["hidden_state"], list)
    assert len(result["hidden_state"]) == 4096
