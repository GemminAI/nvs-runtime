"""The environment decision of the tests, and that the paired tests exclude each other."""

from __future__ import annotations

import importlib.util
from typing import Any

import edge_env
import test_hidden_trajectory_record_capture_extra as paired
from nvs_observation.contract_check import edge_contract_available


def _skip_condition(function: Any) -> bool:
    (mark,) = [m for m in function.pytestmark if m.name == "skipif"]
    return bool(mark.args[0])


def test_the_decision_is_the_one_of_the_package_and_follows_the_edge_package() -> None:
    assert edge_env.EDGE_AVAILABLE is edge_contract_available()
    assert edge_env.EDGE_AVAILABLE == (importlib.util.find_spec("nvs_edge") is not None)


def test_the_two_marks_are_complementary() -> None:
    edge_skipped = bool(edge_env.requires_edge.args[0])
    no_edge_skipped = bool(edge_env.requires_no_edge.args[0])
    assert edge_skipped is (not edge_env.EDGE_AVAILABLE)
    assert no_edge_skipped is edge_env.EDGE_AVAILABLE
    assert edge_skipped != no_edge_skipped


def _marked(reason: str) -> list[Any]:
    """The tests of the paired module that carry the skipif mark with ``reason``, in file order."""
    found = []
    for name, function in vars(paired).items():
        marks = getattr(function, "pytestmark", [])
        if name.startswith("test_") and any(m.kwargs.get("reason") == reason for m in marks):
            found.append(function)
    return found


def test_exactly_one_of_each_pair_runs() -> None:
    with_edge = _marked(edge_env.requires_edge.kwargs["reason"])
    without_edge = _marked(edge_env.requires_no_edge.kwargs["reason"])
    assert len(with_edge) == len(without_edge) == 2
    for first, second in zip(with_edge, without_edge, strict=True):
        skipped_with, skipped_without = _skip_condition(first), _skip_condition(second)
        assert skipped_with != skipped_without
        assert skipped_with is (not edge_env.EDGE_AVAILABLE)
        assert skipped_without is edge_env.EDGE_AVAILABLE
