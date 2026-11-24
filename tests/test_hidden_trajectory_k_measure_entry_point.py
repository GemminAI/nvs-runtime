"""The entry-point lookup of ``k_measure.default_k_measurer`` / ``default_key_source``.

The lookup is tested by giving it the ``entry_points`` result (a stand-in for what an installed
distribution would declare); no kernel is involved, so these tests run in S1 and S3 alike.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nvs_runtime.hidden_trajectory import k_measure, record
from nvs_runtime.hidden_trajectory.record_schema import K_KEYS

KWARGS: dict[str, Any] = {
    "observation_id": "o",
    "point": "final_norm",
    "raw_bytes": b"",
    "trajectory_id": None,
    "model_build_id": None,
    "compute_environment": None,
    "key_source": lambda: b"k",
}


class FakeEntryPoint:
    def __init__(self, name: str, provider: Any) -> None:
        self.name = name
        self._provider = provider

    def load(self) -> Any:
        return self._provider


def _offer(monkeypatch: pytest.MonkeyPatch, *entry_points: FakeEntryPoint) -> list[str]:
    asked: list[str] = []

    def fake(*, group: str) -> list[FakeEntryPoint]:
        asked.append(group)
        return list(entry_points)

    monkeypatch.setattr(k_measure.metadata, "entry_points", fake)
    return asked


def test_without_an_entry_point_nothing_is_measured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _offer(monkeypatch)

    outcome = k_measure.default_k_measurer().run_position(**KWARGS)
    assert outcome.k_path["status"] == record.K_NOT_MEASURED
    assert outcome.k_path["reason"] == record.REASON_K_EDGE_UNAVAILABLE
    assert set(outcome.k_path) == set(K_KEYS)
    assert outcome.k_path["trajectory_id"] is None  # KWARGS has none; no other field is invented
    assert outcome.board is None and outcome.causal is None
    key = k_measure.default_key_source()()
    assert isinstance(key, bytes) and len(key) == 32
    assert k_measure.default_k_checks() is None
    assert asked == [k_measure.K_MEASURER_GROUP] * 3
    assert k_measure.K_MEASURER_GROUP == "nvs_runtime.k_measurer"


def test_an_entry_point_supplies_the_measurer_and_the_key_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measurer = object()

    def key_source() -> bytes:
        return b"from-entry-point"

    checks = object()
    provider = SimpleNamespace(
        default_k_measurer=lambda: measurer,
        default_key_source=lambda: key_source,
        default_k_checks=lambda: checks,
    )
    _offer(monkeypatch, FakeEntryPoint("developer-tools", provider))
    assert k_measure.default_k_measurer() is measurer
    assert k_measure.default_key_source() is key_source
    assert k_measure.default_k_checks() is checks


def test_more_than_one_entry_point_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(
        default_k_measurer=lambda: None,
        default_key_source=lambda: None,
        default_k_checks=lambda: None,
    )
    _offer(monkeypatch, FakeEntryPoint("b-tools", provider), FakeEntryPoint("a-tools", provider))
    for call in (
        k_measure.default_k_measurer,
        k_measure.default_key_source,
        k_measure.default_k_checks,
    ):
        with pytest.raises(k_measure.KMeasurerEntryPointError, match="a-tools, b-tools"):
            call()


def test_an_entry_point_without_the_callables_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _offer(monkeypatch, FakeEntryPoint("incomplete", SimpleNamespace(default_k_measurer=lambda: 1)))
    for call, missing in (
        (k_measure.default_key_source, "default_key_source"),
        (k_measure.default_k_checks, "default_k_checks"),
    ):
        with pytest.raises(k_measure.KMeasurerEntryPointError, match=missing):
            call()
