"""capture_extra (design U6-5 v0.4, 5.2): T-U65-09 (values and unknowns), T-U65-21 (limits and
their order), T-U65-22 (the canonical JSON). The module imports the standard library only; the
numpy cases are skipped where numpy is absent."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from nvs_observation import capture_extra as ce
from nvs_observation.capture_extra import (
    KEEP_KEYS,
    VERSION,
    build_capture_extra,
    canonical_json,
)

FIELD_SEP = "."


def made(values: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    result = build_capture_extra(values, **kwargs)
    assert result.extra is not None, result.overflow
    return result.extra


def reasons_of(extra: dict[str, Any]) -> dict[str, str]:
    return {u["field"]: u["reason"] for u in extra["unknowns"]}


# ---- T-U65-09 ------------------------------------------------------------------------------


def test_good_values_go_in_as_they_are_and_only_tuple_and_mapping_change_type() -> None:
    from types import MappingProxyType

    values = {
        "sampler": {"temperature": 0.7, "top_k": 1, "seed": None, "greedy": True, "name": "x"},
        "max_tokens": 64,
        "chat_template_kwargs": MappingProxyType({"enable_thinking": False}),
        "capture_unknowns": ("provenance.revision", "provenance.eos_ids"),
    }
    extra = made(values)
    assert extra["version"] == VERSION == "nvs.capture-extra/1"
    assert extra["unknowns"] == []
    assert extra["values"] == {
        "sampler": {"temperature": 0.7, "top_k": 1, "seed": None, "greedy": True, "name": "x"},
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "capture_unknowns": ["provenance.revision", "provenance.eos_ids"],
    }
    assert type(extra["values"]["chat_template_kwargs"]) is dict
    assert type(extra["values"]["capture_unknowns"]) is list
    # int stays int and float stays float
    assert type(extra["values"]["sampler"]["top_k"]) is int
    assert type(extra["values"]["sampler"]["temperature"]) is float


@pytest.mark.parametrize(
    ("bad", "reason", "detail"),
    [
        (float("nan"), "not_json_value", "float not finite"),
        (float("inf"), "not_json_value", "float not finite"),
        (-float("inf"), "not_json_value", "float not finite"),
        (2**53, "not_json_value", "int out of range"),
        (-(2**53), "not_json_value", "int out of range"),
        ("a\ud800b", "not_json_value", "str not UTF-8"),
        (b"abc", "not_json_value", "bytes"),
        ({1, 2}, "not_json_value", "set"),
        (object(), "not_json_value", "object"),
    ],
)
def test_a_value_that_json_cannot_hold_is_left_out_with_a_reason(
    bad: Any, reason: str, detail: str
) -> None:
    extra = made({"sampler": {"top_p": bad, "top_k": 1}, "max_tokens": 8})
    assert extra["values"]["sampler"] == {"top_k": 1}  # the rest of the value is kept
    assert extra["unknowns"] == [
        {"field": "sampler.top_p", "reason": reason, "detail": detail}
    ]
    assert "nan" not in json.dumps(extra["values"]).lower()  # nothing was stringified


def test_the_limits_of_int_are_in() -> None:
    extra = made({"max_tokens": 2**53 - 1, "sampler": {"a": -(2**53 - 1)}})
    assert extra["unknowns"] == []


def test_numpy_scalars_and_arrays_are_not_converted() -> None:
    np = pytest.importorskip("numpy")
    extra = made(
        {
            "sampler": {
                "f": np.float32(0.5),
                "f64": np.float64(0.5),  # a subclass of float: still not a plain float
                "i": np.int64(3),
                "b": np.bool_(True),
                "a": np.zeros(2),
            },
            "max_tokens": 1,
        }
    )
    assert extra["values"]["sampler"] == {}
    assert reasons_of(extra) == {
        "sampler.a": "not_json_value",
        "sampler.b": "not_json_value",
        "sampler.f": "not_json_value",
        "sampler.f64": "not_json_value",
        "sampler.i": "not_json_value",
    }


def test_a_key_that_is_not_str_is_left_out_with_not_json_key() -> None:
    extra = made({"sampler": {1: "x", "ok": 2, "a\ud800": 3}, "max_tokens": 1})
    assert extra["values"]["sampler"] == {"ok": 2}
    assert [(u["field"], u["reason"], u["detail"]) for u in extra["unknowns"]] == [
        ("sampler", "not_json_key", "int"),
        ("sampler", "not_json_key", "str"),
    ]


def test_a_list_keeps_the_original_index_in_the_path_of_a_dropped_element() -> None:
    extra = made({"capture_unknowns": ["a", float("nan"), "c"], "max_tokens": 1})
    assert extra["values"]["capture_unknowns"] == ["a", "c"]
    assert extra["unknowns"][0]["field"] == "capture_unknowns[1]"


def nested(levels: int) -> Any:
    value: Any = "leaf"
    for _ in range(levels):
        value = [value]
    return value


def test_a_nest_of_sixteen_is_kept_and_seventeen_is_too_deep() -> None:
    assert made({"sampler": nested(16), "max_tokens": 1})["unknowns"] == []
    extra = made({"sampler": nested(17), "max_tokens": 1})
    assert [u["reason"] for u in extra["unknowns"]] == ["too_deep"]
    assert extra["unknowns"][0]["field"] == "sampler" + "[0]" * 16
    assert extra["unknowns"][0]["detail"] == "list"


def test_a_value_over_16_kib_is_left_out_and_its_hash_is_recorded() -> None:
    big = {"x": "a" * 20000}
    extra = made({"sampler": big, "max_tokens": 1})
    assert "sampler" not in extra["values"]
    (line,) = extra["unknowns"]
    canonical = canonical_json(big)
    assert line == {
        "field": "sampler",
        "reason": "too_large",
        "detail": f"{len(canonical)} bytes",
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def test_a_value_of_exactly_16_kib_is_kept() -> None:
    text = "a" * (16 * 1024 - 2)  # the canonical JSON adds two quotes
    assert len(canonical_json(text)) == 16 * 1024
    assert made({"sampler": text, "max_tokens": 1})["values"]["sampler"] == text


def test_no_value_is_ever_made_into_a_string() -> None:
    extra = made({"sampler": {"x": float("nan"), "y": b"1", "z": object()}, "max_tokens": 1})
    assert extra["values"]["sampler"] == {}


# ---- T-U65-21 ------------------------------------------------------------------------------


def blob(size: int) -> str:
    return "a" * (size - 2)  # canonical JSON of the str is `size` bytes


def sizes_of(extra: dict[str, Any]) -> dict[str, int]:
    return {k: len(canonical_json(v)) for k, v in extra["values"].items()}


def test_over_64_kib_the_largest_droppable_key_goes_first_then_by_key_name() -> None:
    values = {
        "sampler": blob(16384),
        "chat_template_kwargs": blob(16384),
        "capture_unknowns": blob(16384),
        "origin": blob(16384),
        "max_tokens": 8,
    }
    extra = made(values)
    # 4 x 16384 + small > 65536: one key is dropped; equal sizes go in key order
    assert sorted(extra["values"]) == ["chat_template_kwargs", "max_tokens", "origin", "sampler"]
    assert reasons_of(extra) == {"capture_unknowns": "too_large"}  # "capture_..." < "chat_..."
    assert len(canonical_json(extra)) <= 65536


def test_the_order_is_size_descending_and_stops_as_soon_as_it_fits() -> None:
    values = {
        "sampler": blob(16000),
        "chat_template_kwargs": blob(16384),
        "capture_unknowns": blob(16384),
        "origin": blob(15000),
        "adapter_notes": blob(16384),
        "max_tokens": 8,
    }
    result = build_capture_extra(values)
    assert result.extra is not None
    # the three 16384 keys tie: kept ones are "adapter_notes" (never dropped) and the rest in order
    dropped = [u["field"] for u in result.extra["unknowns"] if u["reason"] == "too_large"]
    assert dropped == ["capture_unknowns"]  # "capture_unknowns" < "chat_template_kwargs"
    assert len(canonical_json(result.extra)) <= 65536


def test_keys_that_are_not_dropped_even_when_the_total_is_over() -> None:
    values = {k: blob(16384) for k in KEEP_KEYS}
    result = build_capture_extra(values)  # 5 x 16384 > 64 KiB, nothing droppable
    assert result.overflowed
    assert result.overflow is not None
    assert "total too large after dropping every droppable key" in result.overflow["reasons"]
    assert result.overflow["total_bytes"] > 65536


def test_a_kept_key_over_16_kib_overflows() -> None:
    result = build_capture_extra({"adapter_notes": blob(16385), "sampler": {"a": 1}})
    assert result.overflowed
    assert result.overflow is not None
    assert result.overflow["reasons"] == ["kept key too large: adapter_notes"]


def test_unknowns_over_64_lines_overflow_and_64_lines_do_not() -> None:
    def with_unknowns(n: int) -> Any:
        values = {"sampler": {f"k{i:03d}": float("nan") for i in range(n)}, "max_tokens": 1}
        return build_capture_extra(values)

    ok = with_unknowns(64)
    assert ok.extra is not None and len(ok.extra["unknowns"]) == 64
    over = with_unknowns(65)
    assert over.overflowed
    assert over.overflow is not None
    assert over.overflow["unknowns_count"] == 65
    assert over.overflow["reasons"] == ["too many unknowns"]


def test_unknowns_over_8_kib_overflow() -> None:
    long_path = "k" * 200
    lines = [
        {"field": f"{i}{long_path}", "reason": "failed", "detail": "d" * 100} for i in range(40)
    ]
    result = build_capture_extra({"max_tokens": 1}, extra_unknowns=lines)
    assert result.overflowed and result.overflow is not None
    assert result.overflow["unknowns_count"] == 40
    assert result.overflow["unknowns_bytes"] > 8192
    assert result.overflow["reasons"] == ["unknowns too large"]


def test_a_path_over_256_characters_becomes_sha256_of_the_path() -> None:
    path = "sampler" + "".join(f".{'k' * 30}{i}" for i in range(10))
    value: Any = float("nan")
    for part in reversed(path.split(".")[1:]):
        value = {part: value}
    extra = made({"sampler": value, "max_tokens": 1})
    (line,) = extra["unknowns"]
    assert len(path) > 256
    assert line["field"] == "sha256:" + hashlib.sha256(path.encode("utf-8")).hexdigest()


def test_a_path_of_exactly_256_characters_is_kept() -> None:
    path = "sampler." + "k" * (256 - len("sampler."))
    extra = made({"sampler": {path[len("sampler.") :]: float("nan")}, "max_tokens": 1})
    assert extra["unknowns"][0]["field"] == path and len(path) == 256


def test_the_same_input_gives_the_same_bytes_and_the_order_of_input_does_not_matter() -> None:
    a = {"sampler": {"b": float("nan"), "a": float("inf")}, "max_tokens": 3, "origin": {"z": 1}}
    b = {"origin": {"z": 1}, "max_tokens": 3, "sampler": {"a": float("inf"), "b": float("nan")}}
    assert canonical_json(made(a)) == canonical_json(made(a)) == canonical_json(made(b))
    lines = made(a)["unknowns"]
    assert lines == sorted(lines, key=lambda u: (u["field"], u["reason"]))


def test_extra_unknowns_from_the_caller_are_kept_and_ordered() -> None:
    extra = made(
        {"max_tokens": 1},
        extra_unknowns=[{"field": "origin.edge_version", "reason": "failed", "detail": "OSError"}],
    )
    assert extra["unknowns"] == [
        {"field": "origin.edge_version", "reason": "failed", "detail": "OSError"}
    ]


# ---- T-U65-22 ------------------------------------------------------------------------------


def test_canonical_json_bytes() -> None:
    assert canonical_json({"b": 1, "a": [True, None, "x"]}) == b'{"a":[true,null,"x"],"b":1}'
    assert canonical_json(-0.0) == b"-0.0"
    assert canonical_json(0.1) == b"0.1"
    assert canonical_json(1e-05) == b"1e-05"
    assert canonical_json(1) == b"1" and canonical_json(1.0) == b"1.0"  # kept apart
    assert canonical_json([1, 1.0]) == b"[1,1.0]"
    assert canonical_json(True) == b"true" and canonical_json(2**53 - 1) == b"9007199254740991"


def test_no_space_no_trailing_newline_no_bom() -> None:
    data = canonical_json({"a": {"b": [1, 2]}})
    assert b" " not in data and not data.endswith(b"\n") and not data.startswith(b"\xef\xbb\xbf")


def test_keys_are_ordered_by_code_point_not_by_utf16_unit() -> None:
    low_bmp = "￿"  # BMP, above the surrogates
    astral = "\U00010000"  # outside the BMP: UTF-16 code units D800 DC00
    assert sorted([low_bmp, astral]) == [low_bmp, astral]  # code point order
    # (RFC 8785 would put the astral key first.)
    out = canonical_json({astral: 1, low_bmp: 2})
    assert out == ('{"' + low_bmp + '":2,"' + astral + '":1}').encode("utf-8")


def test_text_is_written_as_it_is_and_only_the_required_characters_are_escaped() -> None:
    assert canonical_json("é") == '"é"'.encode()
    assert canonical_json("\U0001f600") == '"\U0001f600"'.encode()
    assert canonical_json('"\\') == b'"\\"\\\\"'
    assert canonical_json("\b\f\n\r\t") == b'"\\b\\f\\n\\r\\t"'
    assert canonical_json("\x00\x1f\x0b") == b'"\\u0000\\u001f\\u000b"'
    assert canonical_json("\x7f") == b'"\x7f"'  # DEL is not escaped
    assert canonical_json("/") == b'"/"'


def test_no_unicode_normalization() -> None:
    composed, decomposed = "é", "é"
    assert canonical_json(composed) != canonical_json(decomposed)
    extra = made({"sampler": {composed: 1, decomposed: 2}, "max_tokens": 1})
    assert set(extra["values"]["sampler"]) == {composed, decomposed}


def test_a_lone_surrogate_is_not_a_json_value_and_never_reaches_the_bytes() -> None:
    extra = made({"sampler": {"x": "a\udc80"}, "max_tokens": 1})
    assert extra["unknowns"][0]["reason"] == "not_json_value"
    canonical_json(extra)  # does not raise


def test_nan_is_refused_by_the_canonical_json_itself() -> None:
    with pytest.raises(ValueError):
        canonical_json(float("nan"))


# ---- the module imports the standard library only -----------------------------------------------


def test_the_module_imports_the_standard_library_only() -> None:
    source = Path(ce.__file__).read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    allowed = {"__future__", "hashlib", "json", "math", "collections", "dataclasses", "typing"}
    assert roots <= allowed
