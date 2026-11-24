"""Phase 4, method B, the part that stays in nvs-runtime: reading the stored cycles of a capture.

The replay of Risk and Control itself (it calls the kernel's own classes) is tested in the Developer
Tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nvs_runtime.hidden_trajectory.replay import risk_control as rcx


def test_a_capture_without_a_measured_k_cannot_be_read(tmp_path: Path) -> None:
    with pytest.raises(rcx.ReplayError):
        rcx.read_cycles(tmp_path)  # an empty archive: no capture to name
