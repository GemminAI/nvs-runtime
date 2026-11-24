"""The shared capture contract inside nvs-runtime, before it moves to nvs-observation.

The names moved out of ``record.py`` are the same objects. ``capture_record`` and
``observation_constants`` import the standard library only. The duplicated contract
vocabulary matches exactly.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import nvs_observation.capture_check as capture_check
import nvs_observation.capture_extra as capture_extra
import nvs_observation.contracts.capture_record as capture_record
import nvs_observation.contracts.constants as observation_constants
import nvs_observation.origin_map as origin_map
import pytest

import nvs_runtime.hidden_trajectory.payload as payload
import nvs_runtime.hidden_trajectory.record as record
import nvs_runtime.hidden_trajectory.record_schema as record_schema

_SHARED = (
    "REASON_EMPTY_GENERATION",
    "REASON_STEPS_OUT_OF_RANGE",
    "REASON_DIMENSION_OUT_OF_RANGE",
    "REASON_CAPTURE_INVALID",
    "REASON_SPACE_COLLISION",
    "REASON_ORIGIN_MISMATCH",
    "REASON_ORIGIN_FACTS_INVALID",
    "REASON_CONTRACT_NOT_BUILDABLE",
    "REASON_CAPTURE_EXTRA_OVERFLOW",
    "REASON_EDGE_UNAVAILABLE",
    "REASON_EDGE_INCOMPATIBLE",
    "CAPTURE_KEYS",
    "capture_block",
)

_KERNEL_VALUE_SKIP = (
    "nvs-kernel（nvs_edge.contract_v2）が無いので、"
    "公開の契約の値と kernel の内部の値の一致は非公開の結合の CI で確かめる"
)


def _import_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add(".")
            elif node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _imports_record(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "nvs_runtime.hidden_trajectory.record"
                or alias.name.endswith(".hidden_trajectory.record")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "nvs_runtime.hidden_trajectory.record":
                return True
            if node.module == "nvs_runtime.hidden_trajectory" and any(
                alias.name == "record" for alias in node.names
            ):
                return True
    return False


@pytest.mark.parametrize("name", _SHARED)
def test_record_reexports_the_shared_object(name: str) -> None:
    assert getattr(record, name) is getattr(capture_record, name)


def test_shared_modules_import_the_standard_library_only() -> None:
    for module in (capture_record, observation_constants):
        path = Path(module.__file__)
        roots = _import_roots(path)
        assert roots <= sys.stdlib_module_names
        assert "nvs_edge" not in roots


def test_capture_check_does_not_import_record() -> None:
    path = Path(capture_check.__file__)
    assert not _imports_record(path)


def test_public_contract_values_match_the_kernel() -> None:
    try:
        import nvs_edge.contract_v2 as contract_v2
    except ImportError:
        pytest.skip(_KERNEL_VALUE_SKIP)
    assert observation_constants.MAX_DIMENSION == contract_v2.MAX_DIMENSION
    assert observation_constants.POINT_PATTERN == contract_v2.POINT_PATTERN


def test_capture_extra_version_matches_the_record_schema() -> None:
    assert capture_extra.VERSION == record_schema.CAPTURE_EXTRA_VERSION


def test_capture_extra_reasons_match_the_record_schema() -> None:
    # "failed" is written by capture_map.map_capture when edge.source_build_id()
    # raises: the reason on the origin.edge_version line.
    four = {
        capture_extra.REASON_NOT_JSON_VALUE,
        capture_extra.REASON_NOT_JSON_KEY,
        capture_extra.REASON_TOO_DEEP,
        capture_extra.REASON_TOO_LARGE,
    }
    expected = four | {"failed"}
    assert set(record_schema.EXTRA_UNKNOWN_REASONS) == expected
    assert len(record_schema.EXTRA_UNKNOWN_REASONS) == len(expected)


def test_unknown_reasons_match() -> None:
    assert set(origin_map.UNKNOWN_REASONS) == set(payload._UNKNOWN_REASONS)


_O = frozenset(
    {
        "capture_record",
        "observation_constants",
        "adapter_session",
        "capture_check",
        "capture_extra",
        "origin_map",
        "capture_edge",
        "capture_map",
        "contract_build",
        "edge_observe",
    }
)


def _outside_o(path: Path) -> list[str]:
    """``nvs_runtime`` modules this file imports that are not in ``_O``."""
    outside: list[str] = []
    prefix = "nvs_runtime.hidden_trajectory."
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "nvs_runtime" or name.startswith("nvs_runtime."):
                    rest = name[len(prefix) :] if name.startswith(prefix) else None
                    if rest is None or rest not in _O:
                        outside.append(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if node.level:
                if module is None:
                    for alias in node.names:
                        if alias.name not in _O:
                            outside.append("." * node.level + alias.name)
                elif module.split(".", 1)[0] not in _O:
                    outside.append("." * node.level + module)
                continue
            if not module:
                continue
            if module == "nvs_runtime.hidden_trajectory":
                for alias in node.names:
                    if alias.name not in _O:
                        outside.append(f"{module}.{alias.name}")
            elif module.startswith("nvs_runtime."):
                rest = module[len(prefix) :] if module.startswith(prefix) else None
                if rest is None or rest not in _O:
                    outside.append(module)
    return outside


def test_observation_modules_import_only_inside_o() -> None:
    package = Path(capture_record.__file__).parents[1]  # nvs_observation (was hidden_trajectory)
    moved = {
        "capture_record": "contracts/capture_record.py",
        "observation_constants": "contracts/constants.py",
    }
    for name in sorted(_O):
        assert _outside_o(package / moved.get(name, f"{name}.py")) == []


def test_normalize_does_not_import_k_path_or_rerun() -> None:
    path = Path(record.__file__).parent / "phase4" / "normalize.py"
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [module, *[alias.name for alias in node.names]]
        assert "k_path" not in names
        assert "rerun" not in names
        for name in names:
            assert not name.endswith(".k_path")
            assert not name.endswith(".rerun")


def test_canonical_imports_the_standard_library_only() -> None:
    import nvs_runtime.hidden_trajectory.canonical as canonical

    path = Path(canonical.__file__)
    roots = _import_roots(path)
    assert roots <= sys.stdlib_module_names
    assert "nvs_edge" not in roots
