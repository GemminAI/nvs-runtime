"""The Gate child's imports stay in the child and come from a fixed allow-list (U6T-1).

``_gate_child.py`` loads the sensos contract modules by name (``importlib.import_module``) so that
the static test of the package (``test_e09``, which forbids ``contracts`` imports in the modules of
``nvs_runtime``) is not touched. Four conditions make that safe; each is tested here:

1. the parent process never loads a sensos contract module, never gets ``SENSOS_TOOLS`` on its
   ``sys.path``, and no module of it imports ``_gate_child`` (AST and ``sys.modules``);
2. the child does load the modules the check needs (they are in its ``checker_files``);
3. after the child has ended (and after several children) the parent's Python state -- modules,
   ``sys.path``, environment -- is as it was;
4. the names given to ``import_module`` come only from the constant tuple ``_ALLOWED_MODULES`` of
   the file; nothing of a request is used to make one; an unknown ``op`` ends the child
   (``GateUnavailable`` in the client).

The AST tests run everywhere; the ones with the real child need ``GATE_PY`` and ``SENSOS_TOOLS``.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_hidden_trajectory_gate import EXAMPLE, Session
from test_hidden_trajectory_gate_client import needs_real

import nvs_runtime
from nvs_runtime.hidden_trajectory.gate_client import CHILD_SCRIPT, GateClient, GateUnavailable

PACKAGE = Path(nvs_runtime.__file__).resolve().parent


@pytest.fixture(scope="module")
def door() -> Any:
    sys.path.insert(0, str(EXAMPLE))
    try:
        from sensos_door import SensosDoor
    finally:
        sys.path.remove(str(EXAMPLE))
    return SensosDoor.from_environment()


@pytest.fixture(scope="module")
def client() -> GateClient:
    return GateClient.from_environment()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _allowed_modules() -> tuple[str, ...]:
    for node in _tree(CHILD_SCRIPT).body:
        targets = (
            [node.target]
            if isinstance(node, ast.AnnAssign)
            else node.targets
            if isinstance(node, ast.Assign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "_ALLOWED_MODULES" for t in targets):
            value = node.value  # type: ignore[union-attr]
            assert isinstance(value, ast.Tuple), "_ALLOWED_MODULES must be a tuple literal"
            names = []
            for item in value.elts:
                assert isinstance(item, ast.Constant) and isinstance(item.value, str)
                names.append(item.value)
            return tuple(names)
    raise AssertionError("_ALLOWED_MODULES is not defined in _gate_child.py")


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


# ---- condition 4: the names come from a fixed tuple ----------------------------------------


def test_c4_the_allow_list_is_a_constant_tuple_of_contract_modules() -> None:
    names = _allowed_modules()
    assert names and len(set(names)) == len(names)
    assert all(n.startswith("contracts.") and n.count(".") == 1 for n in names)
    assert all(n.split(".", 1)[1].isidentifier() for n in names)


def test_c4_import_module_is_called_with_an_element_of_the_allow_list_only() -> None:
    tree = _tree(CHILD_SCRIPT)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _call_name(n) in {"import_module", "__import__"}
    ]
    assert len(calls) == 1, "exactly one dynamic import in the child"
    (call,) = calls
    assert _call_name(call) == "import_module" and len(call.args) == 1 and not call.keywords
    argument = call.args[0]
    # either a literal ...
    if isinstance(argument, ast.Constant):
        assert argument.value in _allowed_modules()
        return
    # ... or the loop variable of ``for allowed in _ALLOWED_MODULES``
    assert isinstance(argument, ast.Name)
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Name)
        and n.iter.id == "_ALLOWED_MODULES"
        and isinstance(n.target, ast.Name)
        and n.target.id == argument.id
        and any(c is call for c in ast.walk(n))
    ]
    assert len(loops) == 1, "the argument must be the variable of a loop over _ALLOWED_MODULES"


def test_c4_nothing_of_a_request_reaches_the_name_of_an_import() -> None:
    tree = _tree(CHILD_SCRIPT)
    function = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sensos"
    )
    used = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    assert "request" not in used and "op" not in used and "sys" not in used
    # nothing else evaluates or imports by name
    forbidden = {"eval", "exec", "compile", "__import__", "run_module", "run_path", "load_module"}
    called = {_call_name(n) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not (called & forbidden), called & forbidden


def test_c4_every_op_the_child_runs_is_compared_with_a_literal() -> None:
    tree = _tree(CHILD_SCRIPT)
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run")
    comparisons = [
        n
        for n in ast.walk(run)
        if isinstance(n, ast.Compare)
        and isinstance(n.left, ast.Name)
        and n.left.id == "op"
        and len(n.ops) == 1
        and isinstance(n.ops[0], ast.Eq)
    ]
    assert comparisons
    assert all(
        isinstance(c.comparators[0], ast.Constant) and isinstance(c.comparators[0].value, str)
        for c in comparisons
    )


@needs_real
@pytest.mark.parametrize(
    "request_",
    [
        {"op": "no_such_op"},
        {"op": "os.system"},
        {"op": "contracts.gate"},
        {"op": "../../x"},
        {"op": ""},
        {"op": None},
        {"op": ["gate"]},
        {"module": "os"},
        {},
        [],
        "gate",
    ],
)
def test_c4_an_unknown_op_or_a_module_name_is_refused_and_the_child_ends(request_: Any) -> None:
    client = GateClient.from_environment()
    with pytest.raises(GateUnavailable):
        client.process.call(request_)


@needs_real
def test_c4_extra_keys_of_a_request_are_not_used_to_load_anything() -> None:
    process = GateClient.from_environment().process
    plain = process.call({"op": "catalog", "trace": True})
    padded = process.call(
        {
            "op": "catalog",
            "trace": True,
            "module": "subprocess",
            "name": "socket",
            "import": "contracts.not_there",
            "modules": ["os", "contracts.not_there"],
        }
    )
    assert padded == plain
    assert set(plain["_trace"]["modules"]) <= {
        f"tools/{n.replace('.', '/')}.py" for n in _allowed_modules()
    } | set(plain["_trace"]["modules"])


@needs_real
def test_c4_the_loader_refuses_a_name_that_is_not_in_the_list() -> None:
    code = (
        "import runpy, sys\n"
        "namespace = runpy.run_path(sys.argv[1], run_name='gate_child_under_test')\n"
        "try:\n"
        "    namespace['_sensos']('os')\n"
        "except SystemExit as stop:\n"
        "    print('refused:', stop)\n"
        "    sys.exit(0)\n"
        "sys.exit(7)\n"
    )
    done = subprocess.run(
        [os.environ["GATE_PY"], "-c", code, str(CHILD_SCRIPT)],
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "SENSOS_TOOLS": os.environ["SENSOS_TOOLS"]},
    )
    assert done.returncode == 0, done.stderr
    assert "refused: module not allowed: 'os'" in done.stdout


# ---- condition 1 (static part): nobody in the parent imports the child --------------------


def _package_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if p.name != "_gate_child.py")


def test_c1_no_module_of_the_package_imports_the_child_or_a_sensos_module() -> None:
    for path in _package_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(f"{node.module}.{a.name}" for a in node.names)]
            else:
                continue
            for name in names:
                assert "_gate_child" not in name, (path, name)
                assert name != "contracts" and not name.startswith("contracts."), (path, name)
            if isinstance(node, ast.ImportFrom):
                assert all(a.name != "_gate_child" for a in node.names), path


def test_c1_the_client_reaches_the_child_only_as_a_file_it_runs() -> None:
    source = (PACKAGE / "hidden_trajectory" / "gate_client.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # the child is spelt only as a file name or a path (CHILD_SCRIPT; the bundle's file list), never
    # as a dotted module name ...
    spelt = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "_gate_child" in n.value
        and not any(c.isspace() or c == "`" for c in n.value)  # not a docstring
    ]
    assert spelt and all(n.value.endswith("_gate_child.py") for n in spelt), [
        n.value for n in spelt
    ]
    assert not any("." in n.value[: -len(".py")].replace("/", "") for n in spelt)
    # ... that no import call (import_module, __import__) is given anywhere in the package
    for path in [*_package_files(), CHILD_SCRIPT]:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and _call_name(node) in {"import_module", "__import__"}:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        assert "_gate_child" not in argument.value, path
    # ... and it is run by an interpreter, as ``[gate_py, CHILD_SCRIPT]``
    assert "[str(self.gate_py), str(CHILD_SCRIPT)]" in source


def test_c1_the_child_is_not_in_sys_modules() -> None:
    assert not [m for m in sys.modules if m.endswith("_gate_child")]


# ---- conditions 1 and 3 (the parent's state, with the real child, again and again) ---------


def _state(tools: Path) -> dict[str, Any]:
    root = str(tools.resolve())

    def under_tools(name: Any) -> bool:
        return isinstance(name, str) and os.path.realpath(name).startswith(root + os.sep)

    return {
        "contracts": sorted(
            m for m in sys.modules if m == "contracts" or m.startswith("contracts.")
        ),
        "from_tools": sorted(
            name
            for name, module in list(sys.modules.items())
            if under_tools(getattr(module, "__file__", None))
        ),
        "path": list(sys.path),
        "path_has_tools": [p for p in sys.path if os.path.realpath(p) == root],
        "child_module": [m for m in sys.modules if m.endswith("_gate_child")],
        "environ": dict(os.environ),
        "meta_path": len(sys.meta_path),
        "path_hooks": len(sys.path_hooks),
    }


@needs_real
def test_c1_c3_the_parent_is_as_it_was_after_the_child_has_run_again_and_again(
    door: Any,
    client: GateClient,
) -> None:
    tools = Path(os.environ["SENSOS_TOOLS"])
    before = _state(tools)
    assert before["from_tools"] == [] and before["path_has_tools"] == []
    session = Session(door, item="isolation")
    check = session.check()
    result = session.build(check)
    for round_ in range(3):
        for envelope in result.envelopes.values():
            verdict = client.verdict(envelope, blobs=result.blobs, contract_facts=check.facts)
            assert verdict["result"] == "PASS", verdict["violations"]
        fresh = GateClient.from_environment()
        assert fresh.bundle_sha256() == client.bundle_sha256()  # a child ran for the bundle
        after = _state(tools)
        assert after == before, (
            round_,
            {k: (before[k], after[k]) for k in before if before[k] != after[k]},
        )
        assert after["contracts"] == [] and after["from_tools"] == []
        assert after["path_has_tools"] == [] and after["child_module"] == []


# ---- condition 2: the child does load what the check needs --------------------------------


NEEDED = (
    "gate",
    "gate_nvs",
    "hidden_trajectory",
    "ic00_versions",
    "jcs",
    "ids",
    "schemas",
    "contract_facts",
    "hid_space",
    "hidden_trajectory_window",
)


@needs_real
def test_c2_the_child_loads_the_modules_the_check_needs(client: GateClient) -> None:
    answer = client.process.call({"op": "checker_files"})
    in_bundle = {f["path"] for f in answer["files"] if f["repo"] == "sensos"}
    for name in NEEDED:
        assert f"tools/contracts/{name}.py" in in_bundle, name
    # every module of the allow-list is among them: the list is neither longer nor shorter than use
    assert {f"tools/{n.replace('.', '/')}.py" for n in _allowed_modules()} <= in_bundle
    assert {f"contracts.{n}" for n in NEEDED} == set(_allowed_modules())
    # the envelope rule (IC-00) is a module the version table reaches
    assert any(p in in_bundle for p in ("tools/contracts/check.py", "tools/contracts/ic00_v110.py"))
    # the bundle of the client has the same files
    bundle_paths = {
        (f["repo"], f["path"]) for f in __import__("json").loads(client.checker_bundle())["files"]
    }
    assert {("sensos", p) for p in in_bundle} <= bundle_paths
