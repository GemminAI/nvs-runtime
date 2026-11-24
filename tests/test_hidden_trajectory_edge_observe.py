"""The observation under ``EdgeProvider`` is ``edge_observe``, and it does not import the runtime.

The kernel tests use the in-process kernel. Without it they skip: a real ``Edge`` cannot be made.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import nvs_observation.edge_observe as edge_observe
import pytest
from nvs_observation.edge_observe import EdgeObserver

import nvs_runtime.providers.edge as edge_module

_NOT_INSTALLED = "nvs_edge is not installed in this environment"
_NO_BASE_URL = "no kernel base_url configured"
_EARLIER = "kernel was unreachable earlier in this session"


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _string_bindings(tree: ast.AST) -> dict[str, str]:
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            found[target.id] = node.value.value
    return found


def test_edge_observe_imports_nothing_of_nvs_runtime() -> None:
    path = Path(edge_observe.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = ["." * node.level + (node.module or "")]
        else:
            continue
        for module in modules:
            assert not module.startswith("nvs_runtime")
            assert module != "nvs_runtime"
    bindings = _string_bindings(tree)
    imported: list[str] = []
    attributes: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) == "import_module":
            imported.append(_literal(node.args[0]) or "" if node.args else "")
        elif _call_name(node) == "getattr" and len(node.args) >= 2:
            arg = node.args[1]
            if isinstance(arg, ast.Name):
                attributes.append(bindings.get(arg.id, ""))
            else:
                attributes.append(_literal(arg) or "")
    assert sorted(imported) == ["nvs_edge.edge", "nvs_edge.raw_store"]
    assert all("nvs_runtime" not in name for name in imported)
    assert sorted(attributes) == ["Edge", "EdgeTransportError", "RawCapture"]


def test_a_missing_nvs_edge_is_the_same_absence() -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    edge, error, raw = edge_observe._load_nvs_edge(missing)
    assert edge is None
    assert error is Exception
    assert raw is None

    def unnamed(name: str) -> Any:
        return object()

    edge, error, raw = edge_observe._load_nvs_edge(unnamed)
    assert edge is None
    assert error is Exception
    assert raw is None


def test_the_provider_names_the_same_nvs_edge_classes() -> None:
    if not edge_module._NVS_EDGE_AVAILABLE:
        pytest.skip("nvs_edge が無いので、同じクラスかどうかはこの試験では比べられない")
    assert edge_module._Edge is edge_observe._Edge
    assert edge_module._EdgeTransportError is edge_observe._EdgeTransportError
    assert edge_module._RawCapture is edge_observe._RawCapture


def test_the_three_deferred_reasons_have_the_same_wording() -> None:
    observer = EdgeObserver()
    missing = observer.observe(
        session_id="s",
        token_index=0,
        hidden_state=[0.0],
        available=False,
        base_url="http://unused",
        in_process=False,
        edge_for=lambda session_id: None,
    )
    assert missing["reason"] == _NOT_INSTALLED
    assert missing["status"] == "DEFERRED"

    no_url = observer.observe(
        session_id="s",
        token_index=0,
        hidden_state=[0.0],
        available=True,
        base_url=None,
        in_process=False,
        edge_for=lambda session_id: None,
    )
    assert no_url["reason"] == _NO_BASE_URL

    class _Unreachable:
        def observe_token_full(self, token_index: int, hidden_state: Any) -> tuple[Any, Any]:
            raise edge_observe._EdgeTransportError("connection refused")

    first = observer.observe(
        session_id="down",
        token_index=0,
        hidden_state=[0.0],
        available=True,
        base_url="http://unused",
        in_process=False,
        edge_for=lambda session_id: _Unreachable(),
    )
    assert first["reason"] == "kernel unreachable: connection refused"
    again = observer.observe(
        session_id="down",
        token_index=1,
        hidden_state=[0.0],
        available=True,
        base_url="http://unused",
        in_process=False,
        edge_for=lambda session_id: _Unreachable(),
    )
    assert again["reason"] == _EARLIER


def test_a_key_and_a_tier_are_refused_with_the_same_words_before_a_token() -> None:
    observer = EdgeObserver()
    with pytest.raises(ValueError, match="non-empty bytes") as key_error:
        observer.set_session_key("s", b"")
    assert str(key_error.value) == "the session key must be non-empty bytes"
    with pytest.raises(ValueError, match="basic") as tier_error:
        observer.set_session_tier("s", "other")
    assert str(tier_error.value) == "tier must be 'basic' or 'pro', got 'other'"
