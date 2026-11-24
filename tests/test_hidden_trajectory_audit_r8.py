"""R-8 (U6T-2b design 5): the four items the U6 integrated audit found without a test. Tests only;
no source file is changed.

(1) the ``nvs_edge`` loader of ``capture_edge`` (the name of what is missing, the two reasons told
apart);
(2) ``ALIGNMENT_DECIDES_NEXT``, which ``capture_map.py`` and ``payload.py`` each hold: the two are
the same, and the capture a Llama and a
    Qwen Adapter make carries it into the contract;
(3) the ``capture`` block of the record is the one that can be read back from the contract document;
(4) the six materials of a position's space, checked without ``sensos_inference`` (U1) and without
sensos: the keys, and that
    ``space_id_of`` gets exactly them.
"""

from __future__ import annotations

from typing import Any

import pytest
from nvs_observation import capture_edge, capture_map
from nvs_observation.capture_edge import (
    EDGE_INCOMPATIBLE,
    EDGE_UNAVAILABLE,
    load_capture_edge,
)
from test_hidden_trajectory_envelope import Ports
from test_hidden_trajectory_run import pair, run
from u65_support import make_capture

from nvs_runtime.hidden_trajectory import payload
from nvs_runtime.hidden_trajectory import record as rec

EDGE, _ = capture_edge.current()
needs_edge = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")


# ---- (1) the loader ------------------------------------------------------------------------------


class _Module:
    def __init__(self, **names: Any) -> None:
        self.__dict__.update(names)


def _fake_edge(**changes: Any) -> dict[str, Any]:
    """The modules ``load_capture_edge`` reads, with the values it expects; ``changes`` is
    ``{"module.name": value}`` (``None`` removes it)."""
    values: dict[str, dict[str, Any]] = {
        "nvs_edge": {},
        "nvs_edge.contract": {"ContractError": ValueError},
        "nvs_edge.contract_v2": {
            "build_contract_v2": print,
            "Unknown2": dict,
            "layer_label": str,
            "POINT_PATTERN": None,
            "SPACE_RULE": "r",
            "CONTRACT_VERSION_V2": 2,
            "MAX_DIMENSION": 65535,
            "instrument_record": print,
        },
        "nvs_edge.raw_store": {"encode_raw": print, "decode_raw": print},
        "nvs_edge.raw_nd": {"NAME": "raw-nd-v1", "VERSION": 1, "settings_sha256": print},
        "nvs_edge.client": {"source_build_id": print},
    }
    for key, value in changes.items():
        module, name = key.rsplit(".", 1)
        if value is None:
            values[module].pop(name, None)
        else:
            values[module][name] = value
    return {name: _Module(**attrs) for name, attrs in values.items()}


def _importer(modules: dict[str, Any], missing: tuple[str, ...] = ()) -> Any:
    def import_module(name: str) -> Any:
        if name in missing:
            raise ImportError(name)
        return modules[name]

    return import_module


def test_r8_1_a_complete_edge_loads() -> None:
    edge, failure = load_capture_edge(_importer(_fake_edge()))
    assert (
        failure is None
        and edge is not None
        and edge.instrument_name == "raw-nd-v1"
        and edge.space_rule == "r"
    )


@pytest.mark.parametrize(
    "missing",
    [
        "nvs_edge",
        "nvs_edge.contract",
        "nvs_edge.contract_v2",
        "nvs_edge.raw_store",
        "nvs_edge.raw_nd",
        "nvs_edge.client",
    ],
)
def test_r8_1_a_module_that_cannot_be_imported_is_unavailable_and_named(missing: str) -> None:
    edge, failure = load_capture_edge(_importer(_fake_edge(), (missing,)))
    assert edge is None and failure is not None
    assert failure.code == EDGE_UNAVAILABLE and failure.detail == missing


@pytest.mark.parametrize(
    "name",
    [
        "contract.ContractError",
        "contract_v2.build_contract_v2",
        "contract_v2.Unknown2",
        "contract_v2.layer_label",
        "contract_v2.POINT_PATTERN",
        "contract_v2.SPACE_RULE",
        "contract_v2.instrument_record",
        "raw_store.encode_raw",
        "raw_store.decode_raw",
        "raw_nd.settings_sha256",
        "client.source_build_id",
    ],
)
def test_r8_1_a_missing_attribute_is_unavailable_and_named(name: str) -> None:
    edge, failure = load_capture_edge(_importer(_fake_edge(**{f"nvs_edge.{name}": None})))
    assert edge is None and failure is not None
    assert failure.code == EDGE_UNAVAILABLE and failure.detail == f"nvs_edge.{name}"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("contract_v2.CONTRACT_VERSION_V2", 3),
        ("contract_v2.MAX_DIMENSION", 1024),
        ("raw_nd.NAME", "raw-nd-v2"),
        ("raw_nd.VERSION", 2),
        ("contract_v2.CONTRACT_VERSION_V2", True),
        ("raw_nd.VERSION", True),
    ],
)
def test_r8_1_a_version_that_does_not_fit_is_incompatible(name: str, value: Any) -> None:
    edge, failure = load_capture_edge(_importer(_fake_edge(**{f"nvs_edge.{name}": value})))
    assert edge is None and failure is not None and failure.code == EDGE_INCOMPATIBLE
    assert failure.detail.startswith(f"nvs_edge.{name} is ")


def test_r8_1_the_two_reasons_are_the_record_constants() -> None:
    assert (EDGE_UNAVAILABLE, EDGE_INCOMPATIBLE) == (
        rec.REASON_EDGE_UNAVAILABLE,
        rec.REASON_EDGE_INCOMPATIBLE,
    )
    assert EDGE_UNAVAILABLE != EDGE_INCOMPATIBLE


# ---- (2) ALIGNMENT_DECIDES_NEXT ------------------------------------------------------------------


def test_r8_2_the_two_copies_of_the_alignment_name_are_the_same() -> None:
    assert capture_map.ALIGNMENT_DECIDES_NEXT == payload.ALIGNMENT_DECIDES_NEXT == "decides-next/1"


@needs_edge
@pytest.mark.parametrize("kind", ["llama", "qwen"])
def test_r8_2_the_alignment_goes_from_the_capture_into_the_contract_and_the_payload(
    kind: str,
) -> None:
    result = run(pair(make_capture(kind=kind)), Ports())
    assert result.contract is not None
    assert result.contract["binding"]["alignment"] == capture_map.ALIGNMENT_DECIDES_NEXT
    assert {p["alignment"] for p in result.contract["positions"]} == {
        capture_map.ALIGNMENT_DECIDES_NEXT
    }
    for envelope in result.envelopes.values():
        assert envelope["payload"]["contract_ref"] is not None


@needs_edge
def test_r8_2_a_capture_with_another_alignment_is_refused() -> None:
    from u65_support import make_facts

    capture = make_capture()
    facts = make_facts(capture)
    facts["origin"] = {
        **facts["origin"],
        "capture": {**facts["origin"]["capture"], "alignment": "something-else/1"},
    }
    result = run(pair(capture, facts), Ports())
    assert result.stage == "refused" and result.envelopes == {}
    assert result.record["refusals"] == ["origin_mismatch: origin.capture.alignment"]


# ---- (3) the capture block -----------------------------------------------------------------------


@needs_edge
@pytest.mark.parametrize("kind", ["llama", "qwen"])
def test_r8_3_the_capture_block_is_the_one_read_back_from_the_contract(kind: str) -> None:
    result = run(pair(make_capture(kind=kind)), Ports())
    assert result.contract is not None
    from_capture = result.build.capture_block  # read from the capture object
    from_contract = rec.capture_block_of_contract(result.contract)  # read back from the document
    assert from_capture == from_contract == result.record["capture"]
    assert (
        set(from_capture) == set(rec.CAPTURE_KEYS)
        and from_capture["n_gen"]
        and from_capture["points"]
    )


# ---- (4) the six materials of a space, without U1 and without sensos -----------------------------


@needs_edge
@pytest.mark.parametrize(("kind", "count"), [("llama", 1), ("qwen", 5)])
def test_r8_4_space_id_of_gets_exactly_the_six_materials(kind: str, count: int) -> None:
    ports = Ports()
    result = run(pair(make_capture(kind=kind)), ports)
    assert len(ports.spaces) == count == len(result.envelopes)
    for materials in ports.spaces:
        assert set(materials) == {
            "model_build_id",
            "layer",
            "row_rule",
            "space_rule",
            "dimension",
            "compute_environment",
        }
        assert all(value is not None for value in materials.values())
    assert len({tuple(sorted(m.items())) for m in ports.spaces}) == count  # one space per position
