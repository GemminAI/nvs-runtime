"""Phase 4B, the part that stays in nvs-runtime: the metrics of a series and the scan of an evidence
document. The variants, the replay and the real small record are tested in the Developer Tools.
"""

from __future__ import annotations

from nvs_runtime.hidden_trajectory.phase4 import summary as sm
from nvs_runtime.hidden_trajectory.phase4b import metrics as mt
from nvs_runtime.hidden_trajectory.phase4b import scan as sc


def make_series(
    tiers: list[int], pid: str = "p", group: str = "g", finish: str = "eos"
) -> sm.Series:
    cycles = [
        {
            "risk": {"defined": True, "state": "CRITICAL", "smoothed": 0.9},
            "control": {"tier": t},
            "geometry": {"position": [0.0]},
        }
        for t in tiers
    ]
    return sm.Series(pid, group, cycles, [{"dynamic": 1}] * len(tiers), finish, None, len(tiers))


def test_metrics_are_descriptions_with_none_for_an_empty_denominator() -> None:
    base = make_series([1, 7, 7, 6, 7])
    other = make_series([1, 6, 6, 6, 6])
    assert mt.transitions(base) == {"up": 2, "down": 1}
    assert mt.share_different_from(other, base) == 3 / 5
    row = mt.per_series(other, base)
    assert row["q3_class_differs_from_v0"] is True and row["first_L7_cycle"] is None
    assert row["first_L7_cycle_v0"] == 1 and row["n_cycles_with_a_tier_different_from_v0"] == 3
    empty = make_series([])
    assert mt.share_different_from(empty, empty) is None  # not 0 %


def test_moved_a_lot_needs_three_series_of_the_l7_kept_classes() -> None:
    def rows(kept: int) -> list[dict]:
        out = []
        for i in range(12):
            klass = "L7_kept_ended_by_limit" if i < kept else "not_reached"
            out.append(
                {
                    "group": "g",
                    "prompt_id": f"p{i}",
                    "risk_never_defined": False,
                    "first_defined_tier": 7,
                    "q3_class": klass,
                    "q3_class_differs_from_v0": False,
                    "smoothed_median": 0.9,
                    "smoothed_max": 0.9,
                    "l7_share_all_cycles": 1.0,
                    "l7_share_defined_cycles": 1.0,
                }
            )
        return out

    v0 = rows(9)
    assert mt.variant_summary(rows(9), v0)["moved_a_lot"] is False
    assert mt.variant_summary(rows(7), v0)["moved_a_lot"] is False  # a change of 2
    assert mt.variant_summary(rows(6), v0)["moved_a_lot"] is True  # a change of 3
    assert mt.variant_summary(rows(12), v0)["moved_a_lot"] is True
    summary = mt.variant_summary(rows(6), v0)
    assert "rule_0_75" not in summary["Q1"] and summary["change_of_L7_kept_series"] == -3


def test_the_scan_allows_a_numeric_scalar_under_a_forbidden_key_name() -> None:
    allowed = {"per_series": {"p": {"max_abs_difference_by_group": {"guidance_direction": 0.0}}}}
    assert sc.violations(allowed) == []
    assert sc.violations({"velocity": 3, "text": 0.0}) == []  # counts and differences


def test_the_scan_finds_a_list_under_position() -> None:
    found = sc.violations({"cycle": [{"geometry": {"position": [0.1, 0.2, 0.3]}}]})
    assert found == ["$.cycle[0].geometry.position"]


def test_the_scan_finds_a_string_under_text_and_other_non_scalars() -> None:
    assert sc.violations({"a": {"text": "a prompt"}}) == ["$.a.text"]
    assert sc.violations({"guidance_direction": {"x": 1.0}}) == ["$.guidance_direction"]
    assert sc.violations({"position": [1.0], "velocity": True}) == ["$.position", "$.velocity"]
    assert sc.violations({"position": None, "guidance_direction": None}) == []  # nothing held


def test_the_scan_finds_a_nested_numeric_list_under_velocity() -> None:
    nested = {"cycle": {"geometry": {"velocity": {"values": [0.1, 0.2, 0.3]}}}}
    assert sc.violations(nested) == ["$.cycle.geometry.velocity"]
    deeper = {"a": [{"b": {"velocity": {"inner": {"x": [1, 2, 3]}}}}]}
    assert sc.violations(deeper) == ["$.a[0].b.velocity"]
