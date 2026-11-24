"""Phase 4 acquisition support: the normalised comparison and the Q1-Q4 aggregates (synthetic)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nvs_runtime.hidden_trajectory.phase4 import normalize as nz
from nvs_runtime.hidden_trajectory.phase4 import summary as sm

# ---- normalisation and the repeat comparison -------------------------------------------------


def test_hmac_is_removed_at_every_depth_and_the_order_of_lists_is_kept() -> None:
    doc = {
        "b": 1,
        "hmac": "x",
        "a": [{"hmac": "y", "v": [3, 2, 1]}, {"k": {"hmac": "z", "n": {"hmac": "w"}}}],
        "hmac_note": "kept",
    }
    cleaned, removed = nz.strip_hmac(doc)
    assert removed == 4
    assert cleaned["a"][0] == {"v": [3, 2, 1]} and "hmac_note" in cleaned and "hmac" not in cleaned


def test_normalisation_is_canonical_and_ignores_the_hmac() -> None:
    one = json.dumps({"x": [1, 2], "hmac": "aaa", "y": {"hmac": "b", "z": 1.5}}).encode()
    two = json.dumps({"y": {"z": 1.5}, "x": [1, 2], "hmac": "different"}).encode()
    a, b = nz.normalize_causal_log(one), nz.normalize_causal_log(two)
    assert a.sha256 == b.sha256 and a.removed_hmac_keys == 2 and b.removed_hmac_keys == 1
    assert a.data == b'{"x":[1,2],"y":{"z":1.5}}'
    swapped = json.dumps({"x": [2, 1], "y": {"z": 1.5}}).encode()
    assert nz.normalize_causal_log(swapped).sha256 != a.sha256  # the order of a list matters


def test_nan_and_infinity_are_errors_not_replacements() -> None:
    with pytest.raises(nz.NormalizationError):
        nz.normalize_causal_log(b'{"a": NaN}')
    with pytest.raises(nz.NormalizationError):
        nz.normalize_causal_log(b'{"a": [Infinity]}')
    with pytest.raises(nz.NormalizationError):
        nz.normalize_causal_log(b"not json")


def test_the_normaliser_is_identified() -> None:
    record = nz.normalizer_record()
    assert (
        record["normalizer_id"] == "causal-log-normalizer/1" and len(record["source_sha256"]) == 64
    )
    assert record == nz.normalizer_record()


def wire(seq: int, hmac: str, record_hash: str = "h") -> dict:
    return {
        "seq": seq,
        "prev_hash": f"p{seq}",
        "crc": 7 + seq,
        "record_hash": record_hash + str(seq),
        "hmac": hmac,
        "x": seq,
    }


def test_the_repeat_comparison_reports_each_item() -> None:
    log = json.dumps([{"cycle_raw": {"v": 1.0}, "hmac": "k1"}]).encode()
    log2 = json.dumps([{"cycle_raw": {"v": 1.0}, "hmac": "k2"}]).encode()
    first = nz.RunArtifacts([5, 6, 7], log, [wire(0, "a"), wire(1, "b")])
    same = nz.RunArtifacts([5, 6, 7], log2, [wire(0, "c"), wire(1, "d")])
    result = nz.compare_repeat(first, same)
    assert result["all_same"] is True
    assert result["item_1_generated_ids"]["same"] and result["item_2_causal_log"]["same"]
    assert (
        result["item_3_board"]["all_chain_crc_record_hash_same"]
        and result["item_3_board"]["key_independent_same"]
    )
    assert not any(r["hmac_same_not_compared"] for r in result["item_3_board"]["per_record"])
    other_ids = nz.compare_repeat(
        first, nz.RunArtifacts([5, 6, 8], log2, [wire(0, "c"), wire(1, "d")])
    )
    assert other_ids["item_1_generated_ids"]["same"] is False and other_ids["all_same"] is False
    assert other_ids["item_2_causal_log"]["same"] is True  # each item is reported on its own
    changed = nz.compare_repeat(
        first, nz.RunArtifacts([5, 6, 7], log2, [wire(0, "c"), wire(1, "d", "other")])
    )
    assert changed["item_3_board"]["per_record"][1]["record_hash_same"] is False
    assert changed["item_3_board"]["key_independent_same"] is False
    nan = nz.compare_repeat(
        first, nz.RunArtifacts([5, 6, 7], b'[{"a": NaN}]', [wire(0, "c"), wire(1, "d")])
    )
    assert (
        nan["item_2_causal_log"]["same"] is None
        and "normalization_error" in nan["item_2_causal_log"]
    )
    assert nan["all_same"] is False


def test_the_id_hash_is_that_of_little_endian_int64() -> None:
    import hashlib
    import struct

    assert nz.token_ids_sha256([1, 2]) == hashlib.sha256(struct.pack("<2q", 1, 2)).hexdigest()


# ---- the aggregates --------------------------------------------------------------------------


def cycle(
    tier: int, state: str = "CRITICAL", smoothed: float = 0.9, pos: list[float] | None = None
) -> dict:
    return {
        "risk": {"defined": state != "UNDEFINED", "state": state, "smoothed": smoothed},
        "control": {"tier": tier, "reason": "r"},
        "geometry": {"position": pos or [0.0, 0.0]},
    }


def series(
    tiers: list[int],
    *,
    group: str = "g",
    finish: str = "eos",
    note: str | None = None,
    undefined: int = 0,
    anchors: list[dict] | None = None,
    positions=None,
    pid: str = "p",
) -> sm.Series:
    cycles = []
    for i, t in enumerate(tiers):
        state = "UNDEFINED" if i < undefined else "CRITICAL"
        cycles.append(
            cycle(
                t, state, 0.0 if i < undefined else 0.9, None if positions is None else positions[i]
            )
        )
    counts = anchors or [{"config_bootstrap": 0, "dynamic": 0, "manual": 0} for _ in tiers]
    return sm.Series(pid, group, cycles, counts, finish, note, len(tiers))


def test_q3_has_five_classes_decided_in_advance() -> None:
    assert sm.classify_l7(series([1, 1, 3]))["klass"] == "not_reached"
    recovered = sm.classify_l7(series([1, 7, 7, 6, 7]))
    assert recovered["klass"] == "recovered" and recovered["first_L7_cycle"] == 1
    assert recovered["first_transition_cycle"] == 3 and recovered["returned_to_L7"] is True
    assert recovered["returned_to_L7_cycle"] == 4
    assert sm.classify_l7(series([1, 7, 7], finish="eos"))["klass"] == "L7_kept_ended_by_eos"
    assert sm.classify_l7(series([1, 7, 7], finish="length"))["klass"] == "L7_kept_ended_by_limit"
    other = sm.classify_l7(series([1, 7, 7], finish="eos", note="RuntimeError"))
    assert other["klass"] == "other_end" and other["reason"] == "RuntimeError"
    assert sm.classify_l7(series([], finish=None, note="no generation"))["klass"] == "other_end"
    kept = sm.classify_l7(series([7, 7, 7, 7]))
    assert kept["cycles_from_first_L7_to_end"] == 4
    only = sm.classify_l7(series([1, 7, 5]))
    assert only["klass"] == "recovered" and only["returned_to_L7"] is False


def test_per_prompt_values_and_the_none_cases() -> None:
    row = sm.per_prompt(series([1, 1, 7, 7, 6], undefined=2))
    assert row["n_cycles"] == 5 and row["n_defined_cycles"] == 3 and row["first_defined_cycle"] == 2
    assert row["first_defined_tier"] == 7 and row["risk_never_defined"] is False
    assert row["l7_share_all_cycles"] == 2 / 5 and row["l7_share_defined_cycles"] == 2 / 3
    assert row["smoothed_median"] == pytest.approx(0.9) and row["smoothed_max"] == pytest.approx(
        0.9
    )
    never = sm.per_prompt(series([1, 1, 1], undefined=3))
    assert never["risk_never_defined"] is True and never["first_defined_tier"] is None
    assert never["smoothed_median"] is None and never["l7_share_defined_cycles"] is None
    assert never["l7_share_all_cycles"] == 0.0  # not none: cycles exist
    empty = sm.per_prompt(series([]))
    assert empty["l7_share_all_cycles"] is None and empty["l7_share_defined_cycles"] is None


def rows(spec: dict[str, list[int | None]]) -> list[dict]:
    out = []
    for group, firsts in spec.items():
        for i, tier in enumerate(firsts):
            out.append(
                {
                    "group": group,
                    "prompt_id": f"{group}{i}",
                    "risk_never_defined": tier is None,
                    "first_defined_tier": tier,
                }
            )
    return out


def test_q1_counts_are_fractions_and_the_rule_needs_three_defined_per_group() -> None:
    q1 = sm.q1_groups(rows({"a": [7, 7, 5, None], "b": [7, 5, 3, 3], "c": [6, 7, 7, 7]}))
    a = q1["groups"]["a"]
    assert a["risk_defined"]["fraction"] == "3/4" and a["risk_never_defined"]["fraction"] == "1/4"
    assert (
        a["L5_or_above_of_defined"]["fraction"] == "3/3"
        and a["L5_or_above_of_all"]["fraction"] == "3/4"
    )
    assert q1["rule_0_75"]["applicable"] is True
    assert q1["rule_0_75"]["all_groups_at_least_0_75"] is False and q1["rule_0_75"][
        "groups_below_0_75"
    ] == ["b"]
    few = sm.q1_groups(rows({"a": [7, None, None, 7], "b": [7, 7, 7, 7], "c": [7, 7, 7, 7]}))
    assert (
        few["rule_0_75"]["applicable"] is False
        and "all_groups_at_least_0_75" not in few["rule_0_75"]
    )
    assert (
        few["groups"]["a"]["L5_or_above_of_defined"]["fraction"] == "2/2"
    )  # a reference value only
    boundary = sm.q1_groups(rows({"a": [7, 7, 7, 3], "b": [7, 7, 7, 7], "c": [5, 5, 5, 5]}))
    assert boundary["rule_0_75"]["all_groups_at_least_0_75"] is True  # 0.75 exactly counts


def test_q4_groups_use_the_prompt_as_the_unit_and_count_the_left_out() -> None:
    items = [
        {
            "group": "g",
            "smoothed_median": 0.2,
            "smoothed_max": 0.4,
            "l7_share_all_cycles": 0.0,
            "l7_share_defined_cycles": None,
        },
        {
            "group": "g",
            "smoothed_median": 0.4,
            "smoothed_max": 0.6,
            "l7_share_all_cycles": 0.5,
            "l7_share_defined_cycles": 1.0,
        },
        {
            "group": "g",
            "smoothed_median": None,
            "smoothed_max": None,
            "l7_share_all_cycles": 1.0,
            "l7_share_defined_cycles": None,
        },
        {
            "group": "g",
            "smoothed_median": 0.6,
            "smoothed_max": 0.8,
            "l7_share_all_cycles": None,
            "l7_share_defined_cycles": 1.0,
        },
    ]
    out = sm.q4_groups(items)["g"]
    assert out["smoothed_median"]["n"] == 3 and out["smoothed_median"]["n_left_out_none"] == 1
    assert out["smoothed_median"]["median"] == pytest.approx(0.4)
    assert out["smoothed_median"]["iqr"] == pytest.approx(
        np.percentile([0.2, 0.4, 0.6], 75) - np.percentile([0.2, 0.4, 0.6], 25)
    )
    assert (
        out["l7_share_defined_cycles"]["n"] == 2
        and out["l7_share_defined_cycles"]["n_left_out_none"] == 2
    )
    assert (
        sm.q4_groups(
            [
                {
                    "group": "g",
                    "smoothed_median": None,
                    "smoothed_max": None,
                    "l7_share_all_cycles": None,
                    "l7_share_defined_cycles": None,
                }
            ]
        )["g"]["smoothed_max"]["median"]
        is None
    )


def test_the_anchor_formation_record_and_the_not_formed_classes() -> None:
    tight = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.1, 0.1]]
    counts = [{"dynamic": 0}, {"dynamic": 1}, {"dynamic": 1}, {"dynamic": 1}]
    formed = sm.anchor_formation(series([1, 1, 7, 7], anchors=counts, positions=tight))
    assert formed["anchor_formed_cycle"] == 1 and formed["recorded_reason"] is None
    assert formed["spread_at_formation"] == pytest.approx(sm.spread(tight[:2]))
    assert formed["spread_within_stability"] is True
    far = [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]
    none = [{"dynamic": 0}] * 4
    wide = sm.anchor_formation(series([1, 1, 1, 1], anchors=none, positions=far))
    assert wide["anchor_formed_cycle"] is None
    assert wide["not_formed_class"] == "spread_above_stability_in_the_bootstrap_cycles"
    assert (
        sm.anchor_formation(series([1], anchors=none[:1], positions=far[:1]))["not_formed_class"]
        == "fewer_than_two_positions"
    )
    assert sm.spread([[1.0, 0.0]]) is None
    # two unit vectors at distance d have spread d^2 / 4
    d = 1.2
    assert sm.spread([[0.0, 0.0], [d, 0.0]]) == pytest.approx(d * d / 4)
