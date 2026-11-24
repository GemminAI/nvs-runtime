"""The strict schema of ``nvs.capture-record/1`` and ``validate_record`` (U6-5a', design U6-5 v0.5
5.2 and DC-U65-7 = (a)).

The records of the existing U6a tests (built with ``build_envelopes``, ``empty_generation_record``
and ``with_gate_result``) must all pass. ``validate_record`` and the schema dictionary must agree:
where ``jsonschema`` can be imported (in this process, or in the interpreter ``GATE_PY``), the
same records are checked with it too; otherwise those cases are skipped with the reason.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import nvs_observation.capture_extra as extra_module
import pytest
from nvs_observation import contract_check as contract_module
from nvs_observation.capture_extra import build_capture_extra
from test_hidden_trajectory_envelope import (
    NOT_VERIFIED,
    OBSERVATION,
    REJECTED,
    SESSION,
    Ports,
    check_of,
    document,
    empty_capture,
    position,
    rows,
)

import nvs_runtime.hidden_trajectory.record as record_module
import nvs_runtime.hidden_trajectory.record_schema as schema_module
from nvs_runtime.hidden_trajectory import empty_generation_record, with_gate_result
from nvs_runtime.hidden_trajectory.record_schema import (
    RECORD_SCHEMA,
    validate_record,
)


def enveloped() -> dict[str, Any]:
    return Ports().build(document()).record


def partly() -> dict[str, Any]:
    doc = document()
    position(doc, "final_norm")["trajectory_id"] = "unknown"
    return Ports().build(doc).record


def refused() -> dict[str, Any]:
    doc = document()
    doc["origin"]["synthetic"] = True
    return Ports().build(doc).record


def not_a_document(value: Any) -> dict[str, Any]:
    return Ports().build(value, check_of(value, NOT_VERIFIED, sha256=None)).record


def rejected() -> dict[str, Any]:
    doc = document()
    check = check_of(doc, REJECTED, sha256=None, reasons=("contract_structure",))
    return Ports().build(doc, check).record


def empty() -> dict[str, Any]:
    return empty_generation_record(
        session_id=SESSION, observation_id=OBSERVATION, capture=empty_capture()
    )


def with_gates() -> dict[str, Any]:
    record = with_gate_result(enveloped(), "final_norm", status="FAIL", violations=[{"code": "E"}])
    return with_gate_result(
        record, "block_out:3", status="PASS_UNVERIFIED", unverifiable=[{"check": "points"}]
    )


def with_rows() -> dict[str, Any]:
    ports = Ports()
    result = ports.build(document(), rows={p: rows() for p in ("final_norm", "block_out:3")})
    return result.record


U6A_RECORDS = {
    "enveloped": enveloped,
    "partly_enveloped": partly,
    "refused_synthetic": refused,
    "not_a_document_none": lambda: not_a_document(None),
    "not_a_document_text": lambda: not_a_document("text"),
    "not_a_document_empty": lambda: not_a_document({}),
    "rejected_contract": rejected,
    "empty_generation": empty,
    "with_gate_results": with_gates,
    "with_rows": with_rows,
}


def valid_extra() -> dict[str, Any]:
    result = build_capture_extra(
        {
            "sampler": {"temperature": 0.0, "top_k": 1, "bad": float("nan")},
            "max_tokens": 8,
            "recorded_at": "2026-11-16T09:30:15+09:00",
            "adapter_notes": {"version_policy": "package_version_only"},
            "origin_facts_unknowns": [],
            "capture_unknowns": ["provenance.eos_ids"],
            "chat_template_kwargs": {},
            "raw_encoding": {"created_at_ms": 0},
            "origin": {"model": "m"},
        },
        extra_unknowns=[{"field": "origin.edge_version", "reason": "failed", "detail": "OSError"}],
    )
    assert result.extra is not None
    return result.extra


def too_large_extra() -> dict[str, Any]:
    result = build_capture_extra({"sampler": {"x": "a" * 20000}, "max_tokens": 1})
    assert result.extra is not None
    return result.extra


# ---- the existing records pass ---------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(U6A_RECORDS))
def test_every_record_of_the_u6a_tests_is_valid(name: str) -> None:
    record = U6A_RECORDS[name]()
    assert "capture_extra" not in record
    assert validate_record(record) == []
    assert validate_record(json.loads(json.dumps(record))) == []


def test_a_record_may_have_capture_extra_and_may_not() -> None:
    for extra in (valid_extra(), too_large_extra()):
        record = {**enveloped(), "capture_extra": extra}
        assert validate_record(record) == []
    assert validate_record(enveloped()) == []
    assert "capture_extra" not in RECORD_SCHEMA["required"]
    assert "capture_extra" in RECORD_SCHEMA["properties"]


def test_the_constants_are_the_ones_of_the_modules_they_copy() -> None:
    m = schema_module
    assert m.RECORD_VERSION == record_module.RECORD_VERSION == "nvs.capture-record/1"
    assert m.CAPTURE_KEYS == record_module.CAPTURE_KEYS
    assert set(m.OUTCOMES) == {
        record_module.OUTCOME_ENVELOPED,
        record_module.OUTCOME_PARTLY_ENVELOPED,
        record_module.OUTCOME_EMPTY_GENERATION,
        record_module.OUTCOME_REFUSED,
        record_module.OUTCOME_NOT_ENVELOPED,
    }
    assert set(m.POSITION_STATUSES) == {
        record_module.POSITION_ENVELOPED,
        record_module.POSITION_NOT_BUILT,
    }
    assert set(m.CONTRACT_STATUSES) == {
        contract_module.VERIFIED,
        contract_module.NOT_VERIFIED,
        contract_module.REJECTED,
        record_module.CONTRACT_NONE,
    }
    assert set(m.HASH_SOURCES) == {
        contract_module.HASH_RECOMPUTED_MATCHES_RECORDED,
        contract_module.HASH_RECOMPUTED_NO_RECORDED,
        contract_module.HASH_RECORDED_ONLY,
        contract_module.HASH_NONE,
    }
    assert m.CAPTURE_EXTRA_VERSION == extra_module.VERSION
    assert m.MAX_EXTRA_UNKNOWNS == extra_module.MAX_UNKNOWNS
    assert m.MAX_FIELD_CHARS == extra_module.MAX_FIELD_CHARS
    assert m.MAX_DETAIL_CHARS == extra_module.MAX_DETAIL_CHARS
    assert set(m.EXTRA_UNKNOWN_REASONS) == {
        extra_module.REASON_NOT_JSON_VALUE,
        extra_module.REASON_NOT_JSON_KEY,
        extra_module.REASON_TOO_DEEP,
        extra_module.REASON_TOO_LARGE,
        "failed",
    }
    # the nine keys of 5.2, and no other
    assert set(m.EXTRA_VALUE_KEYS) == {
        "sampler",
        "max_tokens",
        "chat_template_kwargs",
        "capture_unknowns",
        "adapter_notes",
        "origin_facts_unknowns",
        "raw_encoding",
        "recorded_at",
        "origin",
    }
    assert set(m.EXTRA_VALUE_KEYS) >= set(extra_module.KEEP_KEYS)


# ---- what is refused -------------------------------------------------------------------------


def with_extra() -> dict[str, Any]:
    return {**enveloped(), "capture_extra": valid_extra()}


def unknown_key_cases() -> list[tuple[str, Any]]:
    def top(r: dict[str, Any]) -> None:
        r["surprise"] = 1

    def cap(r: dict[str, Any]) -> None:
        r["capture"]["surprise"] = 1

    def contract(r: dict[str, Any]) -> None:
        r["contract"]["surprise"] = 1

    def pos(r: dict[str, Any]) -> None:
        r["positions"]["final_norm"]["surprise"] = 1

    def gate(r: dict[str, Any]) -> None:
        r["positions"]["final_norm"]["gate"]["surprise"] = 1

    def extra(r: dict[str, Any]) -> None:
        r["capture_extra"]["surprise"] = 1

    def value(r: dict[str, Any]) -> None:
        r["capture_extra"]["values"]["surprise"] = 1

    def unk(r: dict[str, Any]) -> None:
        r["capture_extra"]["unknowns"][0]["surprise"] = 1

    return [
        ("top level", top),
        ("capture", cap),
        ("contract", contract),
        ("position", pos),
        ("gate", gate),
        ("capture_extra", extra),
        ("capture_extra.values", value),
        ("capture_extra.unknowns[]", unk),
    ]


@pytest.mark.parametrize(
    ("where", "change"), unknown_key_cases(), ids=lambda x: x if isinstance(x, str) else ""
)
def test_one_unknown_key_anywhere_is_a_violation(where: str, change: Any) -> None:
    record = with_extra()
    assert record["capture_extra"]["unknowns"], "the case needs a line in unknowns"
    change(record)
    violations = validate_record(record)
    assert len(violations) == 1 and "surprise" in violations[0], (where, violations)


def test_each_required_key_of_the_record_and_the_capture_is_needed() -> None:
    for key in RECORD_SCHEMA["required"]:
        record = enveloped()
        del record[key]
        assert any(f"missing key '{key}'" in v for v in validate_record(record)), key
    for key in record_module.CAPTURE_KEYS:
        record = enveloped()
        del record["capture"][key]
        assert any(f"missing key '{key}'" in v for v in validate_record(record)), key


def mutate(path: tuple[Any, ...], value: Any, base: Any = enveloped) -> dict[str, Any]:
    record = base()
    node: Any = record
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    return record


BAD_VALUES: list[tuple[str, dict[str, Any]]] = [
    ("version", mutate(("record_version",), "nvs.capture-record/2")),
    ("outcome", mutate(("outcome",), "done")),
    ("session id type", mutate(("session_id",), 5)),
    ("contract_session_id type", mutate(("contract_session_id",), 5)),
    ("n_in type", mutate(("capture", "n_in"), 1.5)),
    ("n_in bool", mutate(("capture", "n_in"), True)),
    ("n_gen type", mutate(("capture", "n_gen"), [6])),
    ("finish_reason type", mutate(("capture", "finish_reason"), 3)),
    ("points type", mutate(("capture", "points"), "final_norm")),
    ("points item", mutate(("capture", "points"), [1])),
    ("prompt hash type", mutate(("capture", "prompt_ids_sha256"), 0)),
    ("contract status", mutate(("contract", "status"), "verified")),
    ("contract hash", mutate(("contract", "contract_sha256"), "abc")),
    ("contract hash upper", mutate(("contract", "contract_sha256"), "A" * 64)),
    ("hash_source", mutate(("contract", "hash_source"), "somewhere")),
    ("refusals type", mutate(("refusals",), "synthetic")),
    ("refusals item", mutate(("refusals",), [""])),
    ("positions type", mutate(("positions",), [])),
    ("position status", mutate(("positions", "final_norm", "status"), "built")),
    ("position reasons", mutate(("positions", "final_norm", "reasons"), "x")),
    ("enveloped without artifact", mutate(("positions", "final_norm", "artifact_id"), None)),
    ("not built with artifact", mutate(("positions", "final_norm", "artifact_id"), "a", partly)),
    ("gate status", mutate(("positions", "final_norm", "gate", "status"), "")),
    ("gate violations", mutate(("positions", "final_norm", "gate", "violations"), {})),
    ("extra version", mutate(("capture_extra", "version"), "nvs.capture-extra/2", with_extra)),
    ("extra values", mutate(("capture_extra", "values"), [], with_extra)),
    ("extra unknowns", mutate(("capture_extra", "unknowns"), {}, with_extra)),
    ("unknown reason", mutate(("capture_extra", "unknowns", 0, "reason"), "odd", with_extra)),
    ("unknown field", mutate(("capture_extra", "unknowns", 0, "field"), "", with_extra)),
    (
        "unknown field long",
        mutate(("capture_extra", "unknowns", 0, "field"), "f" * 257, with_extra),
    ),
    (
        "unknown detail long",
        mutate(("capture_extra", "unknowns", 0, "detail"), "d" * 129, with_extra),
    ),
    (
        "unknown sha256 on a non-too_large",
        mutate(("capture_extra", "unknowns", 0, "sha256"), "a" * 64, with_extra),
    ),
    (
        "too_large without sha256",
        {
            **enveloped(),
            "capture_extra": {
                **too_large_extra(),
                "unknowns": [{"field": "sampler", "reason": "too_large", "detail": "1 bytes"}],
            },
        },
    ),
    (
        "too_large bad sha256",
        {
            **enveloped(),
            "capture_extra": {
                **too_large_extra(),
                "unknowns": [
                    {
                        "field": "sampler",
                        "reason": "too_large",
                        "detail": "1 bytes",
                        "sha256": "xyz",
                    }
                ],
            },
        },
    ),
    (
        "65 unknowns",
        {
            **enveloped(),
            "capture_extra": {
                **valid_extra(),
                "unknowns": [
                    {"field": f"f{i}", "reason": "not_json_value", "detail": "set"}
                    for i in range(65)
                ],
            },
        },
    ),
    (
        "values not a dict item key",
        mutate(("capture_extra", "values", "sampler"), {1: 2}, with_extra),
    ),
    ("record not an object", []),  # type: ignore[list-item]
]


@pytest.mark.parametrize(("name", "record"), BAD_VALUES, ids=[name for name, _ in BAD_VALUES])
def test_a_wrong_type_or_value_is_a_violation(name: str, record: Any) -> None:
    assert validate_record(record) != [], name


def test_64_unknowns_are_allowed() -> None:
    extra = {
        **valid_extra(),
        "unknowns": [
            {"field": f"f{i}", "reason": "not_json_value", "detail": "set"} for i in range(64)
        ],
    }
    assert validate_record({**enveloped(), "capture_extra": extra}) == []


def test_the_checker_does_not_change_the_record_and_is_repeatable() -> None:
    record = with_extra()
    before = copy.deepcopy(record)
    record["surprise"] = 1
    first = validate_record(record)
    assert validate_record(record) == first
    del record["surprise"]
    assert record == before


# ---- the schema as data ----------------------------------------------------------------------


def test_the_schema_is_json_and_its_references_resolve() -> None:
    text = json.dumps(RECORD_SCHEMA, sort_keys=True)
    assert json.loads(text) == RECORD_SCHEMA
    assert RECORD_SCHEMA["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert RECORD_SCHEMA["additionalProperties"] is False
    refs: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "$ref" in node:
                refs.append(node["$ref"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(RECORD_SCHEMA)
    assert refs and all(r.startswith("#/$defs/") and r[8:] in RECORD_SCHEMA["$defs"] for r in refs)
    for name in ("capture", "contract", "gate", "position", "extra_unknown", "capture_extra"):
        assert RECORD_SCHEMA["$defs"][name].get("additionalProperties") is not True
    assert RECORD_SCHEMA["$defs"]["capture"]["required"] == list(record_module.CAPTURE_KEYS)


def test_the_module_imports_the_standard_library_only() -> None:
    tree = ast.parse(Path(schema_module.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "re", "collections", "typing"}


# ---- the same answer with jsonschema ---------------------------------------------------------


def corpus() -> list[Any]:
    records: list[Any] = [factory() for factory in U6A_RECORDS.values()]
    records += [with_extra(), {**enveloped(), "capture_extra": too_large_extra()}]
    records += [record for _, record in BAD_VALUES]
    records += [change_of(change) for _, change in unknown_key_cases()]
    for key in RECORD_SCHEMA["required"]:
        record = enveloped()
        del record[key]
        records.append(record)
    for key in record_module.CAPTURE_KEYS:
        record = enveloped()
        del record["capture"][key]
        records.append(record)
    return records


def change_of(change: Any) -> dict[str, Any]:
    record = with_extra()
    change(record)
    return record


_SCRIPT = """
import json, sys
from jsonschema import Draft202012Validator
data = json.load(sys.stdin)
validator = Draft202012Validator(data["schema"])
Draft202012Validator.check_schema(data["schema"])
print(json.dumps([validator.is_valid(case) for case in data["cases"]]))
"""


def jsonschema_answers(records: list[Any]) -> list[bool] | str:
    """``[valid, ...]`` by jsonschema, or the reason it cannot be asked."""
    cases = json.loads(json.dumps(records))
    try:
        import jsonschema
    except ImportError:
        interpreter = os.environ.get("GATE_PY")
        if not interpreter or not Path(interpreter).is_file():
            return "jsonschema is not importable and GATE_PY is not an interpreter that has it"
        payload = json.dumps({"schema": RECORD_SCHEMA, "cases": cases})
        done = subprocess.run(
            [interpreter, "-c", _SCRIPT], input=payload, capture_output=True, text=True, timeout=60
        )
        if done.returncode != 0:
            return f"GATE_PY cannot run jsonschema: {done.stderr.strip()[-200:]}"
        answers: list[bool] = json.loads(done.stdout)
        return answers
    jsonschema.Draft202012Validator.check_schema(RECORD_SCHEMA)
    validator = jsonschema.Draft202012Validator(RECORD_SCHEMA)
    return [validator.is_valid(case) for case in cases]


def test_validate_record_and_the_schema_agree_under_jsonschema() -> None:
    records = corpus()
    answers = jsonschema_answers(records)
    if isinstance(answers, str):
        pytest.skip(answers)
    ours = [validate_record(json.loads(json.dumps(r))) == [] for r in records]
    assert len(answers) == len(ours)
    disagree = [i for i, (a, b) in enumerate(zip(answers, ours, strict=True)) if a != b]
    assert disagree == [], [(i, validate_record(records[i])) for i in disagree]
    assert any(ours) and not all(ours)  # the corpus has both kinds
