"""A save records the K a measurer returns, and does not import a measurement itself.

The save tests use a measurer that returns a fixed measured K, so no kernel is involved (the K a
real kernel measures is tested in the Developer Tools). They need ``nvs_edge``'s contract v2,
because a save builds its contract with it.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from nvs_observation import capture_edge
from nvs_observation.adapter_session import CapturedWithOrigin
from test_hidden_trajectory_envelope import Ports
from test_hidden_trajectory_run import clock, pair
from test_hidden_trajectory_save import BUNDLE, FakeStore, Gate
from u65_support import make_capture

from nvs_runtime.hidden_trajectory import k_measure, record
from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.formal_save import STAGE_BUILT, _CaptureSaver
from nvs_runtime.hidden_trajectory.record_schema import validate_record

EDGE, _ = capture_edge.current()
needs_edge = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")
KEY = bytes.fromhex("5a" * 32)  # a test's key: no measurer here uses it


def _top_level_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names.extend(alias.name for alias in node.names)
    return names


def _developer_tools_modules() -> dict[str, Any]:
    return {n: m for n, m in sys.modules.items() if n.split(".")[0] == "nvs_developer_tools"}


def test_a_save_without_k_does_not_import_the_developer_tools(tmp_path: Path) -> None:
    saved = _developer_tools_modules()
    for name in saved:
        del sys.modules[name]
    try:
        ports = Ports()
        saver = _CaptureSaver(
            store=FakeStore(),
            archive=CaptureArchive(tmp_path),
            gate=Gate(),
            space_id_of=ports.space_id_of,
            seal=ports.seal,
            checker_bundle=BUNDLE,
            clock=clock,
            k_path=False,
        )
        saver.save(
            CapturedWithOrigin(
                capture=object(),
                origin_facts={},
                adapter_class="test",
                sequence=1,
                last_capture_check="same",
            ),
            contract_session_id="no-k",
            synthetic=False,
            ids=ports.ids,
        )
        assert _developer_tools_modules() == {}
    finally:
        sys.modules.update(saved)


def test_formal_save_and_k_measure_do_not_import_the_developer_tools_at_the_top() -> None:
    import nvs_runtime.hidden_trajectory.formal_save as formal_save

    for module in (formal_save, k_measure):
        names = _top_level_names(Path(module.__file__))
        assert not any(name.split(".")[0] == "nvs_developer_tools" for name in names), module


def _saver(tmp_path: Path, measurer: Any = None) -> tuple[_CaptureSaver, Ports]:
    ports = Ports()
    saver = _CaptureSaver(
        store=FakeStore(),
        archive=CaptureArchive(tmp_path),
        gate=Gate(),
        space_id_of=ports.space_id_of,
        seal=ports.seal,
        checker_bundle=BUNDLE,
        clock=clock,
        k_path=True,
        key_source=lambda: KEY,
        k_measurer=measurer,
    )
    return saver, ports


def _save(saver: _CaptureSaver, ports: Ports) -> Any:
    return saver.save(
        pair(make_capture(kind="llama", dimension=4096)),
        contract_session_id="k-measure",
        synthetic=False,
        ids=ports.ids,
    )


#: A measured K entry of the shape a real measurement gives (a Llama-shaped capture through
#: ``tf-1a``). The values are fixed here: these tests are about what a save does with an outcome.
_MEASURED_K: dict[str, Any] = {
    "chain_head": "a40a96dba89859c46d5e7dda5349d9ecf16e881641307b851a948f04e357f8d0",
    "edge_input_cast_exact": True,
    "edge_session_id": "k-f39ff630ab9a96cf6f9aca5be913232b",
    "input_dimension": 4096,
    "integrity": {
        "chain_and_crc": "recomputable",
        "hmac": "verified_at_run_time",
        "hmac_key": "ephemeral_not_kept",
        "verifier": "nvs_edge.verify.verify_record",
        "verifier_build": "nvs-kernel 7.0.0 0123456789abcdef0123456789abcdef01234567",
    },
    "n_records": 4,
    "policies": {
        "anchor": "ee231a892886b82ba5a1a28234f6c95009aa5b5dca8597687eb11f54b11e77e4",
        "board": "board-42-v2",
        "time": {
            "name": "event-time-v1",
            "sha256": "1dc75e6833c3f7a7da17477ca9a0a2d8ff90694b4b93e48db2c86a76ec68b30a",
        },
        "window": None,
        "wiring": {
            "name": "traj-wired-v2",
            "sha256": "706fcc8d055e3d1e79203f467852b57c25c1bc194cad69c1210af99d698cd4dc",
        },
    },
    "projection": {
        "id": "tf-1a",
        "p_sha256": "aa4c3113b09f9b91b2a20c7949d1f0f6652c8854f1ec4fd37d82ec1735d4f33b",
        "w_sha256": "d3105d04005aac942512e9cef040737b360109a6114575531b5baee5d368f9d9",
    },
    "reason": None,
    "reason_detail": None,
    "space": {
        "anchor_policy_sha256": "ee231a892886b82ba5a1a28234f6c95009aa5b5dca8597687eb11f54b11e77e4",
        "board_format": "board-42-v2",
        "compute_environment": "mlx-4bit",
        "model_build_id": "test-model-4bit",
        "point": "final_norm",
        "projection_id": "tf-1a",
        "projection_sha256": "aa4c3113b09f9b91b2a20c7949d1f0f6652c8854f1ec4fd37d82ec1735d4f33b",
        "quant_step_log2": -30,
        "time_policy_sha256": "1dc75e6833c3f7a7da17477ca9a0a2d8ff90694b4b93e48db2c86a76ec68b30a",
        "w_sha256": "d3105d04005aac942512e9cef040737b360109a6114575531b5baee5d368f9d9",
        "wiring_policy_sha256": "706fcc8d055e3d1e79203f467852b57c25c1bc194cad69c1210af99d698cd4dc",
    },
    "space_id": "ksp:8a3503a8800a5f89275fa3f8ea5216ab7a9df40e5760620780e22bd72c72bcf3",
    "status": "measured",
    "token_index_range": [0, 3],
    "verify_failure": None,
}
_BOARD = b'[{"seq":0,"token_index":0}]'
_CAUSAL = b'[{"token_index":0,"cycle_raw":{}}]'


class _ReturnsAFixedPosition:
    """A ``KMeasurer`` that returns the same measured K (and the same board and causal log bytes)
    for every position; the outcomes it returned are kept."""

    def __init__(self) -> None:
        self.outcomes: list[Any] = []

    def run_position(self, **kwargs: Any) -> Any:
        entry = {
            **_MEASURED_K,
            "trajectory_id": kwargs["trajectory_id"],
            "compute_environment": kwargs["compute_environment"],
            "board_records_sha256": hashlib.sha256(_BOARD).hexdigest(),
            "causal_log_sha256": hashlib.sha256(_CAUSAL).hexdigest(),
        }
        outcome = SimpleNamespace(k_path=entry, board=_BOARD, causal=_CAUSAL)
        self.outcomes.append(outcome)
        return outcome


@needs_edge
def test_a_measurer_return_is_archived_and_recorded(tmp_path: Path) -> None:
    measurer = _ReturnsAFixedPosition()
    saver, ports = _saver(tmp_path, measurer)
    result = _save(saver, ports)
    assert len(measurer.outcomes) == 1
    outcome = measurer.outcomes[0]
    assert outcome.board is not None and outcome.causal is not None
    entry = result.record["positions"]["final_norm"]["k_path"]
    assert entry == outcome.k_path
    archive = CaptureArchive(tmp_path)
    assert archive.read_k(entry["board_records_sha256"]) == outcome.board
    assert archive.read_k(entry["causal_log_sha256"]) == outcome.causal


@needs_edge
def test_the_default_measurer_without_an_entry_point_does_not_measure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(k_measure.metadata, "entry_points", lambda *, group: [])
    saver, ports = _saver(tmp_path)
    result = _save(saver, ports)
    entry = result.record["positions"]["final_norm"]["k_path"]
    assert entry["status"] == record.K_NOT_MEASURED
    assert entry["reason"] == record.REASON_K_EDGE_UNAVAILABLE
    assert validate_record(result.record) == []
    assert not (tmp_path / "capture" / "k").exists()


@needs_edge
def test_an_archive_write_failure_is_not_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(self: CaptureArchive, data: bytes) -> dict[str, Any]:
        raise OSError("k disk full")

    monkeypatch.setattr(CaptureArchive, "write_k", boom)
    saver, ports = _saver(tmp_path, _ReturnsAFixedPosition())
    result = _save(saver, ports)
    assert result.stage == STAGE_BUILT
    entry = result.record["positions"]["final_norm"]["k_path"]
    assert entry["status"] == record.K_NOT_MEASURED
    assert entry["reason"] == record.REASON_K_ARCHIVE_WRITE_FAILED
    assert entry["board_records_sha256"] is None
    assert entry["causal_log_sha256"] is None
    assert entry["chain_head"] is None
    assert entry["integrity"] is None
