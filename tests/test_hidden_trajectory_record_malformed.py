"""A malformed contract document still gives a capture record (U6-5 v0.5, 6.1; U6-5a').

``capture_block_of_contract`` types every value to the ``capture`` definition of the record
schema: what does not fit is ``None``, nothing is converted. A well-formed document gives the
same capture block as before.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from nvs_observation.contract_check import HASH_NONE, REJECTED
from test_hidden_trajectory_envelope import Ports, check_of, document

from nvs_runtime.hidden_trajectory.record import (
    capture_block,
    capture_block_of_contract,
    points_of_contract,
)
from nvs_runtime.hidden_trajectory.record_schema import validate_record


def build_record_of(doc: Any) -> dict[str, Any]:
    check = check_of(doc, REJECTED, sha256=None, hash_source=HASH_NONE, reasons=("structure",))
    return Ports().build(doc, check).record


def mutated(section: str, key: str, value: Any) -> dict[str, Any]:
    doc = document()
    doc[section][key] = value
    return doc


CASES = [
    ("n_in is a dict", mutated("binding", "n_in", {"a": 1}), "n_in"),
    ("n_gen is a bool", mutated("binding", "n_gen", True), "n_gen"),
    ("n_in is a float", mutated("binding", "n_in", 1.5), "n_in"),
    ("model_build_id is an int", mutated("origin", "model_build_id", 7), "model_build_id"),
    ("finish_reason is a list", mutated("binding", "finish_reason", ["stop"]), "finish_reason"),
    (
        "prompt_ids_sha256 is an int",
        mutated("binding", "prompt_ids_sha256", 12),
        "prompt_ids_sha256",
    ),
    (
        "compute_environment is a dict",
        mutated("origin", "compute_environment", {}),
        "compute_environment",
    ),
]


@pytest.mark.parametrize(("name", "doc", "field"), CASES, ids=[c[0] for c in CASES])
def test_a_value_of_the_wrong_type_is_none_and_the_record_is_returned(
    name: str, doc: dict[str, Any], field: str
) -> None:
    record = build_record_of(doc)
    assert validate_record(record) == []
    assert record["capture"][field] is None
    assert record["contract"]["status"] == "REJECTED"
    assert record["refusals"]


def test_a_dict_adapter_is_none() -> None:
    doc = document()
    doc["origin"]["capture"] = {"adapter": {"x": 1}}
    record = build_record_of(doc)
    assert validate_record(record) == []
    assert record["capture"]["adapter"] is None


def test_no_binding_gives_none_and_a_record() -> None:
    doc = document()
    del doc["binding"]
    record = build_record_of(doc)
    assert validate_record(record) == []
    for field in ("n_in", "n_gen", "finish_reason", "prompt_ids_sha256", "gen_token_ids_sha256"):
        assert record["capture"][field] is None
    assert record["refusals"]


def test_a_string_origin_gives_none_and_a_record() -> None:
    doc = document()
    doc["origin"] = "not a mapping"
    record = build_record_of(doc)
    assert validate_record(record) == []
    assert record["capture"]["model_build_id"] is None
    assert record["capture"]["compute_environment"] is None
    assert record["capture"]["adapter"] is None
    assert record["refusals"]


@pytest.mark.parametrize("doc", ["a string", None, 3, ["x"]], ids=["str", "none", "int", "list"])
def test_a_document_that_is_not_a_mapping_gives_a_record(doc: Any) -> None:
    record = build_record_of(doc)
    assert validate_record(record) == []
    assert record["contract"]["status"] == "REJECTED"
    assert record["refusals"]
    assert all(record["capture"][k] is None for k in record["capture"] if k != "points")
    assert record["capture"]["points"] == []


def test_nothing_is_converted_to_a_string() -> None:
    block = capture_block_of_contract(mutated("binding", "n_in", 12.0))
    assert block["n_in"] is None
    block = capture_block_of_contract(mutated("origin", "model_build_id", 7))
    assert block["model_build_id"] is None


def test_int_and_string_counts_are_kept_as_they_are() -> None:
    assert capture_block_of_contract(mutated("binding", "n_in", 5))["n_in"] == 5
    assert capture_block_of_contract(mutated("binding", "n_in", "5"))["n_in"] == "5"


def _old_capture_block_of_contract(doc: Any) -> dict[str, Any]:
    """The function before U6-5a' (a copy), the reference for a well-formed document."""

    def section(name: str) -> Any:
        found = doc.get(name) if isinstance(doc, dict) else None
        return found if isinstance(found, dict) else {}

    binding, origin = section("binding"), section("origin")
    capture = origin.get("capture")
    positions = doc.get("positions") if isinstance(doc, dict) else None
    return capture_block(
        n_in=binding.get("n_in"),
        n_gen=binding.get("n_gen"),
        finish_reason=binding.get("finish_reason"),
        points=points_of_contract(positions),
        model_build_id=origin.get("model_build_id"),
        compute_environment=origin.get("compute_environment"),
        adapter=capture.get("adapter") if isinstance(capture, dict) else None,
        prompt_ids_sha256=binding.get("prompt_ids_sha256"),
        gen_token_ids_sha256=binding.get("gen_token_ids_sha256"),
    )


def test_a_well_formed_document_gives_the_same_bytes_as_before() -> None:
    doc = document()
    new = capture_block_of_contract(doc)
    old = _old_capture_block_of_contract(copy.deepcopy(doc))
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
    assert new["n_in"] is not None
    assert new["model_build_id"] is not None
