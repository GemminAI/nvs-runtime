"""The K part of ``rerun_from_store``: it belongs to ``k_checks``, to the checks of the installed
entry point, or, with neither, it is reported as not checked.

No kernel is needed: the checks are recording stand-ins that satisfy the ``KChecks`` protocol.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nvs_runtime.hidden_trajectory import k_measure, rerun
from nvs_runtime.hidden_trajectory.record import REASON_K_EDGE_UNAVAILABLE


def _record() -> dict[str, Any]:
    measured = {"status": "measured"}
    return {
        "observation_id": "obs-1",
        "contract": {"contract_sha256": "c1"},
        "positions": {
            "final_norm": {"k_path": measured},
            "mid": {"k_path": {"status": "not_measured"}},
            "early": {"k_path": None},
        },
    }


class RecordingChecks:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def check_k_route(self, archive: Any, record: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"archive": archive, "record": record, **kwargs})
        return {"final_norm": {"chain_and_crc": {"ok": True}}}


def _k_part(checks: Any, **changes: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "raw": {"final_norm": b"raw"},
        "trajectory": {"final_norm": "trj-1"},
        "contract": {"origin": {}},
        "rerun_k": True,
    }
    arguments.update(changes)
    return rerun._k_part(checks, "ARCHIVE", _record(), **arguments)  # type: ignore[arg-type]


def test_the_k_part_is_handed_to_k_checks_whole() -> None:
    checks = RecordingChecks()
    report = _k_part(checks)
    assert report == {"final_norm": {"chain_and_crc": {"ok": True}}}
    (call,) = checks.calls
    assert call["archive"] == "ARCHIVE" and call["record"]["observation_id"] == "obs-1"
    assert call["raw"] == {"final_norm": b"raw"} and call["trajectory"] == {"final_norm": "trj-1"}
    assert call["contract"] == {"origin": {}} and call["rerun_k"] is True


def test_without_any_checks_every_measured_position_is_reported_as_not_checked() -> None:
    assert _k_part(None) == {"final_norm": {"checked": False, "reason": REASON_K_EDGE_UNAVAILABLE}}
    assert REASON_K_EDGE_UNAVAILABLE == "edge_unavailable"


def test_a_not_checked_k_does_not_fail_the_capture_and_is_not_a_pass() -> None:
    k_report = _k_part(None)
    # what _check_capture does with a k part: an unchecked position neither passes nor fails it
    assert all(
        i.get("checked") is False or i["chain_and_crc"]["ok"] for i in k_report.values()
    )
    assert all(i.get("checked") is False for i in k_report.values())


def _offer(monkeypatch: pytest.MonkeyPatch, *providers: Any) -> None:
    class Point:
        def __init__(self, name: str, provider: Any) -> None:
            self.name, self._provider = name, provider

        def load(self) -> Any:
            return self._provider

    points = [Point(f"tools-{i}", p) for i, p in enumerate(providers)]
    monkeypatch.setattr(k_measure.metadata, "entry_points", lambda *, group: list(points))


def _archive(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"observation_id": "o1", "contract": {"contract_sha256": None}}]
    monkeypatch.setattr(
        rerun, "CaptureArchive", lambda data_dir: SimpleNamespace(records=lambda: records)
    )


def test_rerun_from_store_asks_the_entry_point_for_the_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checks = RecordingChecks()
    _offer(monkeypatch, SimpleNamespace(default_k_checks=lambda: checks))
    _archive(monkeypatch)
    seen: list[Any] = []
    monkeypatch.setattr(
        rerun, "_check_capture", lambda *a, k_checks, **k: seen.append(k_checks) or {"ok": True}
    )
    # a capture without a contract is skipped before the K part; give it one
    monkeypatch.setattr(
        rerun,
        "CaptureArchive",
        lambda data_dir: SimpleNamespace(
            records=lambda: [{"observation_id": "o1", "contract": {"contract_sha256": "c"}}]
        ),
    )
    rerun.rerun_from_store(tmp_path)
    assert seen == [checks]


def test_two_entry_points_are_an_error_for_the_checks_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = SimpleNamespace(default_k_checks=lambda: RecordingChecks())
    _offer(monkeypatch, provider, provider)
    _archive(monkeypatch)
    with pytest.raises(k_measure.KMeasurerEntryPointError, match="more than one"):
        rerun.rerun_from_store(tmp_path)
