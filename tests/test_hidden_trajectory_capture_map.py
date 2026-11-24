"""U6-5b: the checks of a capture (V-0..V-7), the mapping to contract v2 fields, the comparison of
the capture with ``origin_facts()``, the spaces, ``AdapterSession`` (design U6-5 v0.4).

Tests: T-U65-02..08, 12, 14, 15, 18, 19, 20, 23. A fake capture shaped like U1's
``ObservationCapture`` is used (``u65_support``); the cases that compare with U1 itself are skipped
where ``sensos_inference`` is not importable (``PYTHONPATH`` with the sensos checkout's
``packages/sensos-inference/src``). Cases that call ``nvs_edge`` are skipped where it is absent.
"""

from __future__ import annotations

import dataclasses
import threading
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from nvs_observation import capture_edge, capture_map, origin_map
from nvs_observation.adapter_session import (
    LAST_CAPTURE_NOT_CHECKABLE,
    LAST_CAPTURE_NOT_LAST,
    LAST_CAPTURE_SAME,
    AdapterSession,
    CapturedWithOrigin,
)
from nvs_observation.capture_map import Mapped, map_capture
from nvs_observation.contract_check import check_contract_v2
from u65_support import (
    COMMIT,
    DELETE,
    HASH_A,
    LLAMA_RULE,
    FakeCapture,
    FakeProvenance,
    make_capture,
    make_facts,
    unknown,
    with_provenance,
)

EDGE, _EDGE_FAILURE = capture_edge.current()
needs_edge = pytest.mark.skipif(
    EDGE is None, reason="nvs_edge contract v2 is not usable"
)
pytestmark = needs_edge

TRAJECTORY_IDS = {
    point: f"{index:064x}"
    for index, point in enumerate(
        ("final_norm", "block_out:3", "mlp_out:3", "block_out:7", "mlp_out:7")
    )
}


def mapped(capture: FakeCapture, facts: dict[str, Any] | None = None, **kwargs: Any) -> Mapped:
    kwargs.setdefault("session_id", "session-1")
    kwargs.setdefault("synthetic", False)
    return map_capture(capture, make_facts(capture) if facts is None else facts, **kwargs)


def contract_of(result: Mapped) -> dict[str, Any]:
    assert result.parts is not None, result.details
    assert EDGE is not None
    return EDGE.build_contract_v2(**result.contract_parts(TRAJECTORY_IDS))


def built(
    capture: FakeCapture, facts: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    return contract_of(mapped(capture, facts, **kwargs))


def unknown_fields(contract: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {u["field"]: (u["source"], u["reason"]) for u in contract["unknowns"]}


def lookup(document: Any, dotted: str) -> Any:
    node = document
    for part in dotted.split("."):
        if part.endswith("[]"):
            node = node[part[:-2]][0]
        else:
            node = node[part]
    return node


KINDS = ["llama", "qwen"]


# ---- the mapping: T-U65-03, T-U65-15 ---------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_t_u65_03_each_value_of_the_capture_is_written_as_it_is(kind: str) -> None:
    capture = make_capture(kind=kind)
    facts = make_facts(capture)
    contract = built(capture, facts)
    prov = capture.provenance
    binding, raw, origin = contract["binding"], contract["raw"], contract["origin"]
    assert binding["alignment"] == "decides-next/1" and binding["token_source"] == "model-input-ids"
    assert (binding["n_in"], binding["n_gen"]) == (6, 4)
    assert binding["special_tokens_in"] == [[0, 1]]
    assert binding["tokenizer_sha256"] == prov.tokenizer_sha256
    assert binding["chat_template_sha256"] == prov.chat_template_sha256
    assert binding["prompt_ids_sha256"] == prov.prompt_ids_sha256
    assert binding["gen_token_ids_sha256"] == prov.gen_token_ids_sha256
    assert binding["finish_reason"] == "length" and binding["eos_ids"] == [99]
    assert raw["state_dtype"] == capture.state_dtype and raw["stored_dtype"] == "float32"
    assert raw["cast_exact"] is True
    assert origin["model"] == prov.model_ref and origin["revision"] == prov.revision
    assert origin["weights_sha256"] == prov.weights_sha256
    assert origin["model_build_id"] == capture.model_build_id
    assert origin["compute_environment"] == capture.compute_environment
    assert origin["quantization"] == "4bit/group64" and origin["hardware"].startswith("Mac15,6")
    assert origin["synthetic"] is False
    assert origin["capture"] == {
        "package": "sensos-inference",
        "version": "0.1.0",
        "adapter": capture.adapter,
        "alignment": "decides-next/1",
        "source_commit": COMMIT,
    }
    assert origin["runtime"] == {
        "backend": "mlx",
        "device": "Device(gpu, 0)",
        "mlx": "0.32.4",
        "mlx_lm": "0.32.2",
        "python": "3.12.13",
        "transformers": "5.22.2",
    }
    assert list(origin["runtime"]) == sorted(origin["runtime"])
    assert origin["instruments"] == [
        {
            "name": "raw-nd-v1",
            "version": 1,
            "settings_sha256": origin["instruments"][0]["settings_sha256"],
        }
    ]
    assert [p["point"] for p in contract["positions"]] == list(capture.points)
    for position in contract["positions"]:
        point = position["point"]
        assert position["row_rule"] == capture.row_rule[point]
        assert position["alignment"] == "decides-next/1" and position["k_path"] is None
        assert position["dimension"] == capture.dimension
        assert position["final_norm_applied"] is (point == "final_norm")
        assert position["stored_range"] == {"steps": 4, "first_row_pos": 5}
        assert position["trajectory_id"] == TRAJECTORY_IDS[point]
    assert unknown_fields(contract) == {"origin.blas": ("capture", "not_recorded")}
    assert check_contract_v2(contract).status == "VERIFIED"


def test_the_inputs_are_not_changed() -> None:
    capture = make_capture(kind="qwen")
    facts = make_facts(capture)
    import copy

    before = copy.deepcopy(facts)
    states = {p: a.copy() for p, a in capture.states.items()}
    mapped(capture, facts)
    assert facts == before
    for point, array in capture.states.items():
        assert np.array_equal(array, states[point]) and not array.flags.writeable


@pytest.mark.parametrize(
    "points",
    [
        ("final_norm",),
        ("block_out:3", "mlp_out:3", "block_out:7", "mlp_out:7", "final_norm"),
        ("block_out:0", "block_out:5"),
        ("mlp_out:0", "mlp_out:5"),
    ],
)
def test_t_u65_15_excluded_is_the_same_four_words_whatever_was_captured(
    points: tuple[str, ...],
) -> None:
    capture = make_capture(kind="qwen")
    rules = {p: capture_rule(p) for p in points}
    capture = dataclasses.replace(
        capture,
        points=points,
        states=MappingProxyType({p: np.zeros((4, 8), np.float32) for p in points}),
        row_rule=MappingProxyType(rules),
        row_pos=MappingProxyType({p: np.arange(5, 9) for p in points}),
        row_decides=MappingProxyType({p: np.arange(4) for p in points}),
        final_norm_applied=MappingProxyType({p: p == "final_norm" for p in points}),
    )
    result = mapped(capture)
    assert result.parts is not None, result.details
    assert result.parts["raw"]["excluded"] == ["other-layers", "attention", "logits", "kv"]


def capture_rule(point: str) -> str:
    return {
        "block_out": "qwen35-mlx-block-out/1",
        "mlp_out": "qwen35-mlx-mlp-out/1",
        "final_norm": "qwen35-mlx-final-norm/1",
    }[point.split(":")[0]]


# ---- the unknown rules: T-U65-04, T-U65-19 ---------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "path", "eligible"),
    [
        ({"tokenizer_sha256": None}, "binding.tokenizer_sha256", False),
        ({"chat_template_sha256": None}, "binding.chat_template_sha256", False),
        ({"special_tokens_in": None}, "binding.special_tokens_in", False),
        ({"eos_ids": None}, "binding.eos_ids", True),
    ],
)
def test_t_u65_04_a_value_the_capture_lacks_is_an_unknown_with_its_source(
    changes: dict[str, Any], path: str, eligible: bool
) -> None:
    capture = with_provenance(make_capture(), **changes)
    contract = built(capture)
    (name,) = changes
    assert unknown_fields(contract)[path] == (f"capture.provenance.{name}", "not_recorded")
    report = check_contract_v2(contract).report
    assert report is not None
    assert report["eligibility"]["result"] == ("ELIGIBLE" if eligible else "INELIGIBLE")


@pytest.mark.parametrize(
    ("name", "path"),
    [("revision", "origin.revision"), ("weights_sha256", "origin.weights_sha256")],
)
def test_t_u65_04_a_value_neither_side_has_is_an_unknown_of_the_captures_path_and_ineligible(
    name: str, path: str
) -> None:
    capture = with_provenance(make_capture(), **{name: None})
    facts = make_facts(capture, **{f"origin.{name}": unknown(f"origin.{name}", "unavailable")})
    contract = built(capture, facts)
    assert unknown_fields(contract)[path] == (f"capture.provenance.{name}", "not_recorded")
    report = check_contract_v2(contract).report
    assert report is not None and report["eligibility"]["result"] == "INELIGIBLE"


def test_t_u65_04_a_value_only_origin_facts_has_is_an_unknown_of_its_own_source() -> None:
    capture = make_capture()
    facts = make_facts(
        capture,
        **{
            "origin.hardware": unknown("origin.hardware", "unavailable"),
            "origin.quantization": unknown("origin.quantization", "mystery"),
        },
    )
    contract = built(capture, facts)
    found = unknown_fields(contract)
    assert found["origin.hardware"] == ("origin.hardware", "unavailable")
    assert found["origin.quantization"] == ("origin.quantization", "unknown")  # not a U4 reason
    assert found["origin.blas"] == ("capture", "not_recorded")


def test_t_u65_19_the_four_cases_of_4_0_and_the_side_not_used_is_kept() -> None:
    capture = with_provenance(make_capture(), revision=None, tokenizer_sha256=None)
    facts = make_facts(
        capture,
        **{
            "origin.revision": unknown("origin.revision", "unavailable"),
            "origin.hardware": unknown("origin.hardware", "failed"),
        },
    )
    result = mapped(capture, facts)
    found = unknown_fields(contract_of(result))
    assert found["origin.revision"] == ("capture.provenance.revision", "not_recorded")  # U-3
    assert found["binding.tokenizer_sha256"] == (
        "capture.provenance.tokenizer_sha256",
        "not_recorded",
    )
    assert found["origin.hardware"] == ("origin.hardware", "failed")  # U-2
    assert found["origin.blas"] == ("capture", "not_recorded")  # U-4
    kept = {entry["field"]: entry for entry in result.extra_values["origin_facts_unknowns"]}
    assert kept["origin_facts.origin.revision"] == {
        "field": "origin_facts.origin.revision",
        "source": "origin.revision",
        "reason": "unavailable",
    }
    assert kept["origin_facts.origin.hardware"]["reason"] == "failed"
    assert list(kept) == sorted(kept)


def test_t_u65_19_the_same_input_gives_the_same_unknowns() -> None:
    capture = with_provenance(make_capture(), tokenizer_sha256=None, eos_ids=None)
    first = built(capture)["unknowns"]
    second = built(capture)["unknowns"]
    assert first == second and first == sorted(first, key=lambda u: u["field"])


# ---- eos_ids and source_commit: T-U65-05, T-U65-06 -------------------------------------------


@pytest.mark.parametrize(
    ("eos", "expected"),
    [(None, "unknown"), ((), []), ((2, 3), [2, 3])],
)
def test_t_u65_05_eos_ids_none_empty_and_values_are_kept_apart(eos: Any, expected: Any) -> None:
    capture = with_provenance(make_capture(), eos_ids=eos)
    contract = built(capture)
    if eos is None:
        assert contract["binding"]["eos_ids"] == "unknown"
        assert unknown_fields(contract)["binding.eos_ids"] == (
            "capture.provenance.eos_ids",
            "not_recorded",
        )
    else:
        assert contract["binding"]["eos_ids"] == expected
        assert "binding.eos_ids" not in unknown_fields(contract)
    report = check_contract_v2(contract).report
    assert report is not None and report["eligibility"]["result"] == "ELIGIBLE"


def test_t_u65_06_source_commit_is_written_when_there_is_one_and_changes_the_hash() -> None:
    capture = make_capture()
    with_commit = built(capture)
    without = built(capture, make_facts(capture, **{"origin.capture.source_commit": DELETE}))
    assert with_commit["origin"]["capture"]["source_commit"] == COMMIT
    assert "source_commit" not in without["origin"]["capture"]
    assert (
        check_contract_v2(with_commit).contract_sha256 != check_contract_v2(without).contract_sha256
    )
    # version carries no commit
    assert with_commit["origin"]["capture"]["version"] == "0.1.0"


def test_t_u65_06_a_source_commit_that_is_not_a_string_is_not_written_and_is_noted() -> None:
    capture = make_capture()
    facts = make_facts(capture, **{"origin.capture.source_commit": unknown("x", "failed")})
    result = mapped(capture, facts)
    assert result.parts is not None
    assert "source_commit" not in result.parts["origin"]["capture"]
    assert result.extra_values["adapter_notes"]["source_commit_not_used"]


# ---- the comparison: T-U65-07, T-U65-08 ------------------------------------------------------


SINGLE_FIELDS = [
    ("origin.model", "model_ref", "origin.model"),
    ("origin.revision", "revision", "origin.revision"),
    ("origin.weights_sha256", "weights_sha256", "origin.weights_sha256"),
]


@pytest.mark.parametrize(("field_name", "attribute", "facts_path"), SINGLE_FIELDS)
def test_t_u65_07_the_five_rows_for_a_provenance_value(
    field_name: str, attribute: str, facts_path: str
) -> None:
    base = make_capture()
    value = getattr(base.provenance, attribute)
    # row 1: A and A
    assert mapped(base).refusals == ()
    # row 2: A and B
    result = mapped(base, make_facts(base, **{facts_path: "other"}))
    assert result.refusals == ("origin_mismatch",) and result.details == (
        f"origin_mismatch: {field_name}",
    )
    # row 3: A and unknown
    result = mapped(base, make_facts(base, **{facts_path: unknown(facts_path)}))
    assert result.details == (f"origin_mismatch: {field_name}",)
    if attribute == "model_ref":
        return  # model_ref is never None in U1; rows 4 and 5 are for the nullable ones
    nulled = with_provenance(base, **{attribute: None})
    # row 4: None and a value
    result = mapped(nulled, make_facts(base, **{facts_path: value}))
    assert result.details == (f"origin_mismatch: {field_name}",)
    # row 5: None and unknown -> an Unknown2, no mismatch
    result = mapped(nulled, make_facts(base, **{facts_path: unknown(facts_path)}))
    assert result.refusals == () and result.parts is not None


def test_t_u65_07_the_adapter_and_the_alignment() -> None:
    capture = make_capture()
    assert mapped(capture).refusals == ()
    other = make_facts(capture, **{"origin.capture.adapter": "qwen35-mlx-capture"})
    assert mapped(capture, other).details == ("origin_mismatch: origin.capture.adapter",)
    unknown_adapter = make_facts(capture, **{"origin.capture.adapter": unknown("a")})
    assert mapped(capture, unknown_adapter).details == ("origin_mismatch: origin.capture.adapter",)
    for bad in ("decides-prev/1", unknown("origin.capture.alignment"), None):
        facts = make_facts(capture, **{"origin.capture.alignment": bad})
        assert mapped(capture, facts).details == ("origin_mismatch: origin.capture.alignment",)
    # the capture's own adapter is None and the facts' is unknown: an Unknown2 (rows 4 and 5)
    nameless = dataclasses.replace(capture, adapter=None)  # type: ignore[arg-type]
    assert mapped(nameless, make_facts(capture)).details == (
        "origin_mismatch: origin.capture.adapter",
    )
    facts = make_facts(capture, **{"origin.capture.adapter": unknown("origin.capture.adapter")})
    result = mapped(nameless, facts)
    assert result.refusals == ()
    assert unknown_fields(contract_of(result))["origin.capture.adapter"] == (
        "capture.adapter",
        "not_recorded",
    )


def test_t_u65_07_origin_facts_of_another_adapter_is_refused() -> None:
    llama, qwen = make_capture(kind="llama"), make_capture(kind="qwen")
    result = mapped(llama, make_facts(qwen))
    assert "origin_mismatch" in result.refusals
    assert "origin_mismatch: origin.capture.adapter" in result.details


def test_t_u65_08_runtime_the_five_cases() -> None:
    capture = make_capture()
    # the same keys, a different value
    facts = make_facts(capture, **{"origin.runtime.mlx": "0.99.0"})
    assert mapped(capture, facts).details == ("origin_mismatch: origin.runtime.mlx",)
    # one side unknown
    facts = make_facts(capture, **{"origin.runtime.device": unknown("origin.runtime.device")})
    assert mapped(capture, facts).details == ("origin_mismatch: origin.runtime.device",)
    # both unknown: an Unknown2, the contract is built, a payload cannot be written (6.3)
    nulled = with_provenance(
        capture, runtime={"backend": "mlx", "mlx": None, "mlx_lm": "0.32.2", "device": "d"}
    )
    facts = make_facts(nulled, **{"origin.runtime.mlx": unknown("origin.runtime.mlx")})
    contract = built(nulled, facts)
    assert unknown_fields(contract)["origin.runtime.mlx"] == (
        "capture.provenance.runtime.mlx",
        "not_recorded",
    )
    # (The contract writes the marker "unknown" for it; whether a payload may then be written is
    # the question of 6.3 and U6-5d -- see the report: session_obstacles looks only at str.)
    assert contract["origin"]["runtime"]["mlx"] == "unknown"
    # a key only the facts have (python, transformers) and a key only the capture has
    extra_key = with_provenance(
        capture, runtime={**capture.provenance.runtime, "extra_key": "only-in-capture"}
    )
    contract = built(extra_key)
    assert contract["origin"]["runtime"]["extra_key"] == "only-in-capture"
    assert contract["origin"]["runtime"]["python"] == "3.12.13"
    # a key only the facts have, unknown: an Unknown2 of the facts' own source
    facts = make_facts(
        capture,
        **{"origin.runtime.transformers": unknown("origin.runtime.transformers", "unavailable")},
    )
    contract = built(capture, facts)
    assert unknown_fields(contract)["origin.runtime.transformers"] == (
        "origin.runtime.transformers",
        "unavailable",
    )
    # backend: different -> a mismatch; unknown on either side -> capture_invalid
    facts = make_facts(capture, **{"origin.runtime.backend": "vllm"})
    assert mapped(capture, facts).details == ("origin_mismatch: origin.runtime.backend",)
    nobackend = with_provenance(
        capture, runtime={"mlx": "0.32.4", "mlx_lm": "0.32.2", "device": "d"}
    )
    facts = make_facts(nobackend, **{"origin.runtime.backend": unknown("origin.runtime.backend")})
    result = mapped(nobackend, facts)
    assert result.refusals == ("capture_invalid",)
    assert "origin.runtime.backend" in result.details[0]


def test_origin_map_rules_on_plain_values() -> None:
    both_unknown = origin_map.merge_single(None, "capture.x", unknown("o.x"), "o.x", "origin.x")
    assert both_unknown.value == origin_map.Missing("capture.x", "not_recorded")
    assert origin_map.merge_single("A", "c", "A", "f", "origin.x").value == "A"
    assert origin_map.merge_single("A", "c", "B", "f", "origin.x").mismatch == "origin.x"
    # a missing key of origin_facts is taken as unknown, not as a value
    assert origin_map.merge_single("A", "c", None, "f", "origin.x").mismatch == "origin.x"
    assert origin_map.is_unknown_dict({"source": "s", "reason": "r"})
    assert not origin_map.is_unknown_dict({"source": "s", "reason": "r", "extra": 1})
    assert origin_map.facts_unknowns({"a": unknown("s", "r"), "b": {"c": unknown("t", "u")}}) == [
        {"field": "origin_facts.origin.a", "source": "s", "reason": "r"},
        {"field": "origin_facts.origin.b.c", "source": "t", "reason": "u"},
    ]


def test_origin_facts_that_is_not_shaped_like_one_is_invalid_not_a_mismatch() -> None:
    capture = make_capture()
    for facts in ({}, {"origin": None}, {"origin": "x"}):
        result = mapped(capture, facts)
        assert result.refusals == ("origin_facts_invalid",) and result.parts is None
    result = map_capture(capture, None, session_id="s", synthetic=False)  # type: ignore[arg-type]
    assert result.refusals == ("origin_facts_invalid",)


# ---- the checks: T-U65-12, T-U65-23 ----------------------------------------------------------


def break_capture(capture: FakeCapture, how: str) -> FakeCapture:
    point = capture.points[0]
    rows = dict(capture.row_pos)
    decides = dict(capture.row_decides)
    states = dict(capture.states)
    if how == "V-1 duplicate point":
        return dataclasses.replace(capture, points=(point, point))
    if how == "V-1 empty points":
        return dataclasses.replace(capture, points=())
    if how == "V-1 keys":
        return dataclasses.replace(capture, row_rule=MappingProxyType({}))
    if how == "V-2 shape":
        states[point] = np.zeros((3, capture.dimension), np.float32)
        return dataclasses.replace(capture, states=MappingProxyType(states))
    if how == "V-2 dtype":
        states[point] = np.zeros((4, capture.dimension), np.float64)
        return dataclasses.replace(capture, states=MappingProxyType(states))
    if how == "V-2 finite":
        bad = np.zeros((4, capture.dimension), np.float32)
        bad[1, 1] = np.nan
        states[point] = bad
        return dataclasses.replace(capture, states=MappingProxyType(states))
    if how == "V-3 row_pos":
        rows[point] = np.array([6, 7, 8, 9])  # one too many
        return dataclasses.replace(capture, row_pos=MappingProxyType(rows))
    if how == "V-3 row_decides":
        decides[point] = np.arange(5)
        return dataclasses.replace(capture, row_decides=MappingProxyType(decides))
    if how == "V-5 prompt":
        return with_provenance(capture, prompt_ids_sha256=HASH_A)
    if how == "V-5 gen":
        return with_provenance(capture, gen_token_ids_sha256=HASH_A)
    if how == "V-6":
        return with_provenance(capture, eos_ids=(52,))
    raise AssertionError(how)


@pytest.mark.parametrize(
    "how",
    [
        "V-1 duplicate point",
        "V-1 empty points",
        "V-1 keys",
        "V-2 shape",
        "V-2 dtype",
        "V-2 finite",
        "V-3 row_pos",
        "V-3 row_decides",
        "V-5 prompt",
        "V-5 gen",
        "V-6",
    ],
)
def test_t_u65_12_each_check_refuses_with_its_number(how: str) -> None:
    result = mapped(break_capture(make_capture(), how))
    assert result.refusals == ("capture_invalid",)
    assert how.split()[0] in result.details[0] and result.parts is None
    assert result.extra_values["origin"]["capture"]["adapter"] == "llama31-mlx-capture"


def test_t_u65_12_an_n_in_attribute_that_disagrees_is_invalid() -> None:
    capture = make_capture()
    liar = SimpleNamespace(
        **{f.name: getattr(capture, f.name) for f in dataclasses.fields(capture)}
    )
    liar.n_in = 99
    result = map_capture(liar, make_facts(capture), session_id="s", synthetic=False)
    assert result.refusals == ("capture_invalid",) and "V-0" in result.details[0]


def test_t_u65_12_the_end_of_the_range_and_one_over() -> None:
    assert mapped(make_capture(n_gen=1)).parts is not None
    over_dimension = make_capture(dimension=65536, n_gen=1)
    assert mapped(over_dimension).refusals == ("dimension_out_of_range",)
    assert mapped(make_capture(dimension=65535, n_gen=1)).refusals == ()
    no_stop = {"eos_ids": (1,)}  # the ids of the generation run from 50 up
    over_steps = make_capture(n_gen=65536, dimension=1, provenance=no_stop)
    assert mapped(over_steps).refusals == ("steps_out_of_range",)
    assert mapped(make_capture(n_gen=65535, dimension=1, provenance=no_stop)).refusals == ()
    zero = dataclasses.replace(make_capture(), dimension=0)
    assert "dimension_out_of_range" in mapped(zero).refusals


def test_a_capture_with_no_generated_token_is_an_empty_generation() -> None:
    result = mapped(make_capture(n_gen=0))
    assert result.refusals == ("empty_generation",) and result.parts is None
    assert result.extra_values["origin"]["capture"]["adapter"] == "llama31-mlx-capture"
    assert {"sampler", "max_tokens", "chat_template_kwargs"} <= set(result.extra_values)


@pytest.mark.parametrize(
    "ids",
    [
        np.zeros((2, 2), np.int64),  # 2-D
        np.array([True, False]),  # bool array
        np.array([1.0, 2.0]),  # float array
        np.array([-1, 2]),  # negative id
        [1, True],  # bool element
        [1, 2.5],  # float element
        [1, -3],  # negative
        "abc",
        None,
    ],
)
def test_t_u65_23_v0_ids_that_are_not_ids(ids: Any) -> None:
    capture = dataclasses.replace(make_capture(), prompt_ids=ids)
    result = mapped(capture)
    assert result.refusals == ("capture_invalid",)
    assert result.details[0].startswith("capture_invalid: V-0")
    assert len(result.details) == 1  # the later checks were not made


def test_t_u65_23_v0_token_ids_and_n_in_zero() -> None:
    bad_tokens = dataclasses.replace(make_capture(), token_ids=np.array([-5, 1]))
    assert mapped(bad_tokens).details[0].startswith("capture_invalid: V-0 token_ids")
    no_prompt = dataclasses.replace(make_capture(), prompt_ids=np.array([], dtype=np.int64))
    assert "n_in is 0" in mapped(no_prompt).details[0]
    # a list of ints, and numpy integers, are ids; an unsigned array too
    ok = dataclasses.replace(
        make_capture(),
        prompt_ids=[10, 11, 12, 13, 14, 15],
        token_ids=np.arange(50, 54, dtype=np.uint32),
    )
    assert mapped(ok).refusals == ()
    unsigned_too_big = dataclasses.replace(make_capture(), prompt_ids=np.array([2**63], np.uint64))
    assert mapped(unsigned_too_big).details[0].startswith("capture_invalid: V-0")


def test_a_missing_attribute_is_v0() -> None:
    capture = make_capture()
    partial = SimpleNamespace(
        **{
            f.name: getattr(capture, f.name)
            for f in dataclasses.fields(capture)
            if f.name != "states"
        }
    )
    assert "V-0 attribute states is missing" in mapped(partial).details[0]  # type: ignore[arg-type]
    assert map_capture(object(), make_facts(capture), session_id="s", synthetic=False).parts is None


def test_a_point_name_the_contract_does_not_know_is_invalid() -> None:
    capture = make_capture()
    odd = dataclasses.replace(
        capture,
        points=("attn_out:3",),
        states=MappingProxyType({"attn_out:3": capture.states["final_norm"]}),
        row_rule=MappingProxyType({"attn_out:3": LLAMA_RULE}),
        row_pos=MappingProxyType({"attn_out:3": capture.row_pos["final_norm"]}),
        row_decides=MappingProxyType({"attn_out:3": capture.row_decides["final_norm"]}),
        final_norm_applied=MappingProxyType({"attn_out:3": False}),
    )
    assert mapped(odd).details[0].startswith("capture_invalid: V-1 point")


# ---- spaces: T-U65-16 (collision), T-U65-18 --------------------------------------------------


def test_two_points_with_the_same_layer_and_row_rule_collide() -> None:
    capture = make_capture(kind="qwen")
    points = ("block_out:3", "block_out:3")
    # a duplicate name is V-1; two names with one layer and one rule: block_out:3 and block_out:03
    same_layer = dataclasses.replace(
        capture,
        row_rule=MappingProxyType({**capture.row_rule, "mlp_out:3": "qwen35-mlx-block-out/1"}),
    )
    # (the layer of mlp_out:3 is 3, the rule is the block-out rule: same materials as block_out:3)
    result = mapped(same_layer)
    assert result.refusals == ("space_collision",)
    assert result.details == ("space_collision: block_out:3 and mlp_out:3 have the same space",)
    assert points  # the case above is the collision; this keeps the intent visible


def test_t_u65_18_the_space_materials_are_the_ones_u1_makes() -> None:
    obs = pytest.importorskip("sensos_inference.observation")
    for kind in KINDS:
        capture = make_capture(kind=kind)
        u1 = obs.ObservationCapture(
            **{
                **{f.name: getattr(capture, f.name) for f in dataclasses.fields(capture)},
                "provenance": obs.CaptureProvenance(
                    **{
                        f.name: getattr(capture.provenance, f.name)
                        for f in dataclasses.fields(capture.provenance)
                    }
                ),
            }
        )
        assert EDGE is not None
        for point in capture.points:
            assert capture_map.space_materials(capture, point, EDGE) == obs.space_materials(
                u1, point
            )
        assert capture_map.space_collisions(capture, EDGE) == obs.space_collisions(u1)
        assert mapped(capture).refusals == ()


# ---- T-U65-02: where every value goes --------------------------------------------------------


#: field -> where it goes: "contract:<path>", "capture_block:<key>", "capture_extra:<key>" or
#: "not_carried:<why>". The contract paths are checked against a contract built from a capture.
DESTINATION = {
    "CaptureProvenance.model_ref": "contract:origin.model",
    "CaptureProvenance.revision": "contract:origin.revision",
    "CaptureProvenance.weights_sha256": "contract:origin.weights_sha256",
    "CaptureProvenance.tokenizer_sha256": "contract:binding.tokenizer_sha256",
    "CaptureProvenance.chat_template_sha256": "contract:binding.chat_template_sha256",
    "CaptureProvenance.special_tokens_in": "contract:binding.special_tokens_in",
    "CaptureProvenance.prompt_ids_sha256": "contract:binding.prompt_ids_sha256",
    "CaptureProvenance.gen_token_ids_sha256": "contract:binding.gen_token_ids_sha256",
    "CaptureProvenance.runtime": "contract:origin.runtime",
    "CaptureProvenance.sampler": "capture_extra:sampler",
    "CaptureProvenance.max_tokens": "capture_extra:max_tokens",
    "CaptureProvenance.chat_template_kwargs": "capture_extra:chat_template_kwargs",
    "CaptureProvenance.eos_ids": "contract:binding.eos_ids",
    "ObservationCapture.adapter": "contract:origin.capture.adapter",
    "ObservationCapture.model_build_id": "contract:origin.model_build_id",
    "ObservationCapture.compute_environment": "contract:origin.compute_environment",
    "ObservationCapture.dimension": "contract:positions[].dimension",
    "ObservationCapture.prompt_ids": "contract:binding.prompt_ids_sha256",
    "ObservationCapture.token_ids": "contract:binding.gen_token_ids_sha256",
    "ObservationCapture.finish_reason": "contract:binding.finish_reason",
    "ObservationCapture.points": "contract:positions[].point",
    "ObservationCapture.states": "not_carried:the rows go to the RAW file",
    "ObservationCapture.row_rule": "contract:positions[].row_rule",
    "ObservationCapture.row_pos": "not_carried:checked by V-3, kept as stored_range",
    "ObservationCapture.row_decides": "not_carried:checked by V-3, kept as stored_range",
    "ObservationCapture.final_norm_applied": "contract:positions[].final_norm_applied",
    "ObservationCapture.state_dtype": "contract:raw.state_dtype",
    "ObservationCapture.provenance": "not_carried:each of its fields is in this table",
    "ObservationCapture.unknowns": "capture_extra:capture_unknowns",
    "ObservationCapture.n_in": "contract:binding.n_in",
    "ObservationCapture.n_gen": "contract:binding.n_gen",
    "ObservationCapture.n_rows": "not_carried:the same value as n_gen",
    "origin_facts.origin.capture.package": "contract:origin.capture.package",
    "origin_facts.origin.capture.version": "contract:origin.capture.version",
    "origin_facts.origin.capture.adapter": "contract:origin.capture.adapter",
    "origin_facts.origin.capture.alignment": "contract:origin.capture.alignment",
    "origin_facts.origin.capture.source_commit": "contract:origin.capture.source_commit",
    "origin_facts.origin.runtime.backend": "contract:origin.runtime",
    "origin_facts.origin.runtime.mlx": "contract:origin.runtime",
    "origin_facts.origin.runtime.mlx_lm": "contract:origin.runtime",
    "origin_facts.origin.runtime.device": "contract:origin.runtime",
    "origin_facts.origin.runtime.python": "contract:origin.runtime",
    "origin_facts.origin.runtime.transformers": "contract:origin.runtime",
    "origin_facts.origin.hardware": "contract:origin.hardware",
    "origin_facts.origin.quantization": "contract:origin.quantization",
    "origin_facts.origin.model": "contract:origin.model",
    "origin_facts.origin.revision": "contract:origin.revision",
    "origin_facts.origin.weights_sha256": "contract:origin.weights_sha256",
    "origin_facts.capabilities_phase": "capture_extra:adapter_notes",
    "origin_facts.observed_state_dtype": "capture_extra:adapter_notes",
    "origin_facts.version_policy": "capture_extra:adapter_notes",
}
NOT_CARRIED_ALLOWED = {
    "ObservationCapture.states",
    "ObservationCapture.row_pos",
    "ObservationCapture.row_decides",
    "ObservationCapture.n_rows",
    "ObservationCapture.provenance",
}


def leaves(node: Any, prefix: str) -> set[str]:
    if isinstance(node, dict) and not origin_map.is_unknown_dict(node):
        out: set[str] = set()
        for key, value in node.items():
            out |= leaves(value, f"{prefix}.{key}")
        return out
    return {prefix}


def test_t_u65_02_the_fake_has_the_fields_of_the_table() -> None:
    names = {f"CaptureProvenance.{f.name}" for f in dataclasses.fields(FakeProvenance)}
    names |= {f"ObservationCapture.{f.name}" for f in dataclasses.fields(FakeCapture)}
    names |= {"ObservationCapture.n_in", "ObservationCapture.n_gen", "ObservationCapture.n_rows"}
    facts = make_facts(make_capture())
    names |= leaves(facts["origin"], "origin_facts.origin")
    names |= {f"origin_facts.{k}" for k in facts if k != "origin"}
    assert names == set(DESTINATION)


def test_t_u65_02_every_field_of_u1_and_every_leaf_of_origin_facts_has_a_destination() -> None:
    obs = pytest.importorskip("sensos_inference.observation")
    adapters = pytest.importorskip("sensos_inference.adapters")
    found = {f"CaptureProvenance.{f.name}" for f in dataclasses.fields(obs.CaptureProvenance)}
    found |= {f"ObservationCapture.{f.name}" for f in dataclasses.fields(obs.ObservationCapture)}
    found |= {"ObservationCapture.n_in", "ObservationCapture.n_gen", "ObservationCapture.n_rows"}
    for adapter in (
        adapters.Llama31MLXAdapter(SimpleNamespace(model_id="m", revision="r")),
        adapters.Qwen35MLXAdapter(SimpleNamespace(model_id="m"), n_layers=2, dimension=4),
    ):
        facts = adapter.origin_facts()
        found |= leaves(facts["origin"], "origin_facts.origin")
        found |= {f"origin_facts.{k}" for k in facts if k != "origin"}
    missing = sorted(found - set(DESTINATION))
    assert missing == [], f"a new field or leaf has no destination in the table: {missing}"
    # the source commit is present only inside a git checkout; every other row must exist
    stale = sorted(set(DESTINATION) - found - {"origin_facts.origin.capture.source_commit"})
    assert stale == [], f"the table has rows for fields that no longer exist: {stale}"


def test_t_u65_02_the_destinations_are_real() -> None:
    capture = make_capture()
    result = mapped(capture)
    contract = contract_of(result)
    block = capture_map.describe_capture(capture, result.check)
    for name, destination in DESTINATION.items():
        kind, _, target = destination.partition(":")
        if kind == "contract":
            lookup(contract, target)
        elif kind == "capture_block":
            assert target in block
        elif kind == "capture_extra":
            assert target in result.extra_values, name
        else:
            assert kind == "not_carried" and name in NOT_CARRIED_ALLOWED, name
    assert {k for k, v in DESTINATION.items() if v.startswith("not_carried")} == NOT_CARRIED_ALLOWED
    assert {"n_in", "n_gen", "finish_reason", "points", "model_build_id", "adapter"} <= set(block)
    assert block["n_in"] == 6 and block["n_gen"] == 4 and block["points"] == ["final_norm"]


# ---- the edge: T-U65-14 ----------------------------------------------------------------------


class _Module(SimpleNamespace):
    pass


def real_modules() -> dict[str, Any]:
    import importlib

    return {
        name: importlib.import_module(name)
        for name in (
            "nvs_edge",
            "nvs_edge.contract",
            "nvs_edge.contract_v2",
            "nvs_edge.raw_store",
            "nvs_edge.raw_nd",
            "nvs_edge.client",
        )
    }


def loader(modules: dict[str, Any]) -> Any:
    def import_module(name: str) -> Any:
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    return import_module


def stand_in(real: Any, **changes: Any) -> Any:
    names = {name: getattr(real, name) for name in dir(real) if not name.startswith("__")}
    names.update(changes)
    return _Module(**names)


def test_t_u65_14_the_real_edge_loads() -> None:
    edge, failure = capture_edge.load_capture_edge()
    assert failure is None and edge is not None
    assert edge.instrument_name == "raw-nd-v1" and edge.instrument_version == 1


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("nvs_edge.contract", "ContractError"),
        ("nvs_edge.contract_v2", "build_contract_v2"),
        ("nvs_edge.contract_v2", "Unknown2"),
        ("nvs_edge.contract_v2", "layer_label"),
        ("nvs_edge.contract_v2", "POINT_PATTERN"),
        ("nvs_edge.contract_v2", "SPACE_RULE"),
        ("nvs_edge.contract_v2", "instrument_record"),
        ("nvs_edge.raw_store", "encode_raw"),
        ("nvs_edge.raw_store", "decode_raw"),
        ("nvs_edge.raw_nd", "settings_sha256"),
        ("nvs_edge.client", "source_build_id"),
    ],
)
def test_t_u65_14_a_missing_name_is_edge_unavailable_with_its_name(module: str, name: str) -> None:
    modules = real_modules()
    real = modules[module]
    names = {n: getattr(real, n) for n in dir(real) if not n.startswith("__") and n != name}
    modules[module] = _Module(**names)
    edge, failure = capture_edge.load_capture_edge(loader(modules))
    assert edge is None and failure == capture_edge.EdgeFailure(
        "edge_unavailable", f"{module}.{name}"
    )


def test_t_u65_14_a_missing_module_and_a_missing_package() -> None:
    modules = real_modules()
    del modules["nvs_edge.raw_nd"]
    assert capture_edge.load_capture_edge(loader(modules))[1] == capture_edge.EdgeFailure(
        "edge_unavailable", "nvs_edge.raw_nd"
    )
    assert capture_edge.load_capture_edge(loader({}))[1] == capture_edge.EdgeFailure(
        "edge_unavailable", "nvs_edge"
    )


@pytest.mark.parametrize(
    ("module", "name", "value"),
    [
        ("nvs_edge.contract_v2", "CONTRACT_VERSION_V2", 3),
        ("nvs_edge.contract_v2", "MAX_DIMENSION", 4096),
        ("nvs_edge.raw_nd", "NAME", "raw-nd-v2"),
        ("nvs_edge.raw_nd", "VERSION", 2),
    ],
)
def test_t_u65_14_a_version_that_does_not_fit_is_edge_incompatible(
    module: str, name: str, value: Any
) -> None:
    modules = real_modules()
    modules[module] = stand_in(modules[module], **{name: value})
    edge, failure = capture_edge.load_capture_edge(loader(modules))
    assert edge is None and failure is not None
    assert failure.code == "edge_incompatible" and name in failure.detail


def test_t_u65_14_a_refusal_when_the_edge_is_not_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    for code, detail in (
        ("edge_unavailable", "nvs_edge.raw_nd.NAME"),
        ("edge_incompatible", "nvs_edge.contract_v2.CONTRACT_VERSION_V2 is 3, need 2"),
    ):
        monkeypatch.setattr(capture_edge, "_EDGE", None)
        monkeypatch.setattr(capture_edge, "_FAILURE", capture_edge.EdgeFailure(code, detail))
        result = mapped(make_capture())
        assert result.refusals == (code,) and result.details == (f"{code}: {detail}",)
        assert result.parts is None


def test_t_u65_14_source_build_id_failing_makes_edge_version_an_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert EDGE is not None

    def broken() -> str:
        raise PermissionError("cannot read the source")

    monkeypatch.setattr(capture_edge, "_EDGE", dataclasses.replace(EDGE, source_build_id=broken))
    result = mapped(make_capture())
    assert result.refusals == ()
    contract = contract_of(result)
    assert unknown_fields(contract)["origin.edge_version"] == (
        "nvs_edge.client.source_build_id",
        "failed",
    )
    assert result.extra_unknowns == (
        {"field": "origin.edge_version", "reason": "failed", "detail": "PermissionError"},
    )
    report = check_contract_v2(contract).report
    assert (
        report is not None and report["eligibility"]["result"] == "ELIGIBLE"
    )  # not an eligibility field


# ---- T-U65-20: AdapterSession ----------------------------------------------------------------


class _Adapter:
    """An Adapter whose ``origin_facts()`` follows its last capture, like U2's."""

    def __init__(self, *, interleave: threading.Event | None = None) -> None:
        self.calls: list[str] = []
        self._count = 0
        self._last_capture: Any = None
        self._count_lock = threading.Lock()
        self._interleave = interleave

    def capture(
        self, messages: Any, request: Any, *, model_build_id: str, compute_environment: str
    ) -> Any:
        with self._count_lock:
            self._count += 1
            number = self._count
        capture = SimpleNamespace(revision=f"rev-{number}", messages=messages)
        self._last_capture = capture
        self.calls.append(f"capture:{number}")
        if self._interleave is not None:
            self._interleave.wait(timeout=0.05)  # lets another thread in, if the caller allows it
        return capture

    def origin_facts(self) -> dict[str, Any]:
        self.calls.append("facts")
        return {"origin": {"revision": self._last_capture.revision}}


def test_t_u65_20_two_threads_get_each_their_own_facts() -> None:
    adapter = _Adapter(interleave=threading.Event())
    session = AdapterSession(adapter)
    pairs: list[CapturedWithOrigin] = []

    def worker(tag: str) -> None:
        for _ in range(5):
            pairs.append(
                session.capture_with_origin(tag, None, model_build_id="m", compute_environment="c")
            )

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(pairs) == 10
    for pair in pairs:
        assert pair.origin_facts["origin"]["revision"] == pair.capture.revision
        assert pair.last_capture_check == LAST_CAPTURE_SAME
        assert pair.adapter_class == "_Adapter"
    assert sorted(p.sequence for p in pairs) == list(range(1, 11))
    # the calls come in pairs: a capture and then the facts of it, never two captures in a row
    assert adapter.calls[0::2] == [c for c in adapter.calls if c.startswith("capture")]
    assert all(c == "facts" for c in adapter.calls[1::2])


def test_t_u65_20_a_capture_made_around_the_session_is_found_by_the_last_capture_check() -> None:
    adapter = _Adapter()
    session = AdapterSession(adapter)
    original = adapter.capture

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        adapter.capture = original  # type: ignore[method-assign]
        original("someone else", None, model_build_id="m", compute_environment="c")
        return result

    adapter.capture = interrupted  # type: ignore[method-assign]
    pair = session.capture_with_origin("mine", None, model_build_id="m", compute_environment="c")
    assert pair.last_capture_check == LAST_CAPTURE_NOT_LAST
    assert pair.origin_facts["origin"]["revision"] != pair.capture.revision
    capture = make_capture()
    result = map_capture(
        capture,
        make_facts(capture),
        session_id="s",
        synthetic=False,
        last_capture_check=pair.last_capture_check,
    )
    assert result.refusals == ("origin_mismatch",) and result.details == (
        "origin_mismatch: capture_not_last",
    )


def test_t_u65_20_an_adapter_without_last_capture_is_not_checked_and_the_note_says_so() -> None:
    class Plain:
        def capture(self, *_: Any, **__: Any) -> Any:
            return SimpleNamespace()

        def origin_facts(self) -> dict[str, Any]:
            return {}

    pair = AdapterSession(Plain()).capture_with_origin(
        "m", None, model_build_id="m", compute_environment="c"
    )
    assert pair.last_capture_check == LAST_CAPTURE_NOT_CHECKABLE
    capture = make_capture()
    result = map_capture(
        capture,
        make_facts(capture),
        session_id="s",
        synthetic=False,
        last_capture_check=pair.last_capture_check,
    )
    assert result.refusals == ()
    assert result.extra_values["adapter_notes"]["last_capture_check"] == "not_checkable"


def test_t_u65_20_an_error_of_the_adapter_is_raised_and_the_lock_is_released() -> None:
    class Failing(_Adapter):
        def capture(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    session = AdapterSession(Failing())
    for _ in range(2):  # the second call would block forever if the lock were still held
        with pytest.raises(RuntimeError):
            session.capture_with_origin("m", None, model_build_id="m", compute_environment="c")


def test_the_captured_pair_is_frozen() -> None:
    pair = CapturedWithOrigin(
        capture=1, origin_facts={}, adapter_class="A", sequence=1, last_capture_check="same"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pair.sequence = 2  # type: ignore[misc]


def test_a_provenance_fake_default_is_consistent() -> None:
    assert FakeProvenance().eos_ids == (99,)  # the base of the tests above
