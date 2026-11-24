"""The observe CLI: preflight, fake observe / log / verify. No mlx and no real model."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fake_observe_adapter import LlamaAdapter, LlamaProvider
from nvs_observation import capture_edge
from test_hidden_trajectory_envelope import Ports
from test_hidden_trajectory_formal_writer import ListStore
from test_hidden_trajectory_run import clock as run_clock
from test_hidden_trajectory_save import BUNDLE, verdict_for

from nvs_runtime.hidden_trajectory import formal_writer
from nvs_runtime.hidden_trajectory.cli import (
    EXIT_ERROR,
    EXIT_FAIL,
    EXIT_OK,
    _cli_error,
    _declaration,
    _parser,
    cmd_log,
    cmd_observe,
    cmd_verify,
    display_path,
    main,
)
from nvs_runtime.hidden_trajectory.formal_writer import FormalWriter
from nvs_runtime.hidden_trajectory.model_catalog import GREEDY, MAX_TOKENS
from nvs_runtime.integrations import hekb_formal

EDGE, _ = capture_edge.current()
needs_edge = pytest.mark.skipif(EDGE is None, reason="nvs_edge contract v2 is not usable here")
PROMPT = "SECRET_PROMPT_TOKEN_XYZ"
CLOCK = datetime(2026, 11, 18, 12, 0, 0, tzinfo=timezone(timedelta(hours=9)))


class World:
    def __init__(self) -> None:
        self.stores: dict[str, ListStore] = {}

    def open(self, root: str, data_dir: str) -> ListStore:
        return self.stores[data_dir]

    def create(self, root: str, data_dir: str, declaration: Any) -> ListStore:
        self.stores[data_dir] = ListStore()
        return self.stores[data_dir]


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> World:
    made = World()
    monkeypatch.setattr(formal_writer, "_open_store", made.open)
    monkeypatch.setattr(formal_writer, "_create_store", made.create)
    monkeypatch.setattr(hekb_formal, "_claim_checkout", lambda root: str(root))
    monkeypatch.setattr(hekb_formal, "_release_checkout", lambda: None)
    return made


@pytest.fixture(autouse=True)
def _no_writer_left() -> Any:
    yield
    for writer in list(formal_writer._registry.values()):
        writer.close()


class FakeProcess:
    def call(self, request: Any) -> dict[str, Any]:
        return {"rows": [], "id": "x"}


class FakeClient:
    def __init__(self) -> None:
        self.process = FakeProcess()

    def checker_bundle(self) -> bytes:
        return BUNDLE


class FakeDoor:
    def __init__(self, ports: Ports, gate: Any) -> None:
        self.ports = ports
        self._gate = gate
        self.sensos_tools = Path(".")
        self.client = FakeClient()

    def session_id(self, **kwargs: Any) -> str:
        payload = json.dumps(kwargs, sort_keys=True, separators=(",", ":"))
        return "ses:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def observation_id(self, session_id: str, ordinal: int = 0) -> str:
        payload = json.dumps(
            {"ordinal": ordinal, "session_id": session_id}, sort_keys=True, separators=(",", ":")
        )
        return "obs:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def space_id_of(self, materials: Any) -> str:
        return self.ports.space_id_of(materials)

    def seal(self, envelope: dict[str, Any], lookup: Any) -> dict[str, Any]:
        return self.ports.seal(envelope, lookup)

    def verdict(self, envelope: Any, **kwargs: Any) -> Any:
        return self._gate(envelope, **kwargs)


class PassGate:
    def __call__(self, envelope: Any, **_kwargs: Any) -> dict[str, Any]:
        return verdict_for(envelope)


class FailGate:
    def __call__(self, envelope: Any, **_kwargs: Any) -> dict[str, Any]:
        return verdict_for(envelope, result="FAIL", violations=[{"code": "x", "message": "no"}])


def _layout(tmp_path: Path) -> dict[str, Path]:
    snap = tmp_path / "snap"
    snap.mkdir()
    gate = tmp_path / "py"
    gate.write_text("x", encoding="utf-8")
    tools = tmp_path / "tools"
    (tools / "contracts").mkdir(parents=True)
    (tools / "contracts" / "gate_nvs.py").write_text("#\n", encoding="utf-8")
    hekb = tmp_path / "hekb"
    (hekb / "graph").mkdir(parents=True)
    (hekb / "graph" / "formal.py").write_text("#\n", encoding="utf-8")
    out = tmp_path / "run"
    return {"snap": snap, "gate": gate, "tools": tools, "hekb": hekb, "out": out}


def _ns(paths: dict[str, Path], **changes: Any) -> Any:
    values = {
        "model": "llama31",
        "snapshot": str(paths["snap"]),
        "prompt": PROMPT,
        "prompt_file": None,
        "out": str(paths["out"]),
        "max_tokens": MAX_TOKENS,
        "gate_python": str(paths["gate"]),
        "sensos_tools": str(paths["tools"]),
        "hekb_root": str(paths["hekb"]),
        "replicate": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _observe(paths: dict[str, Path], door: FakeDoor, **kw: Any) -> tuple[int, str]:
    buf = io.StringIO()
    ns_changes = dict(kw.pop("ns_changes", {}) or {})
    max_tokens = int(ns_changes.get("max_tokens", MAX_TOKENS))
    adapter = kw.pop("adapter", None) or LlamaAdapter(LlamaProvider(snapshot_dir=paths["snap"]))
    request = SimpleNamespace(points=("final_norm",), max_tokens=max_tokens, sampling=dict(GREEDY))
    code = cmd_observe(
        _ns(paths, **ns_changes),
        adapter=adapter,
        door=door,
        declaration={"declaration": "formal-store/1"},
        request=request,
        clock=lambda: CLOCK,
        out=buf,
        **kw,
    )
    return code, buf.getvalue()


def _observation_id(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("observation_id:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(text)


def _tree_has_prompt(root: Path, prompt: str) -> bool:
    needle = prompt.encode("utf-8")
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if needle in path.read_bytes():
            return True
    return False


def test_observe_preflight_lists_what_is_missing_and_exits_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("GATE_PY", "SENSOS_TOOLS", "HEKB_FORMAL_ROOT", "LLAMA31_SNAPSHOT"):
        monkeypatch.delenv(name, raising=False)
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    code = main(["observe", "--model", "llama31"])
    text = buf.getvalue()
    assert code == EXIT_ERROR
    assert "missing:" in text
    assert "--prompt or --prompt-file" in text
    assert "GATE_PY" in text
    assert "SENSOS_TOOLS" in text
    assert "HEKB_FORMAL_ROOT" in text
    assert "LLAMA31_SNAPSHOT" in text
    assert PROMPT not in text


@needs_edge
def test_fake_observe_log_verify_exit_0(world: World, tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    door = FakeDoor(Ports(), PassGate())
    code, text = _observe(paths, door)
    assert code == EXIT_OK
    assert PROMPT not in text
    assert os.path.isabs(str(tmp_path))
    shown = display_path(paths["out"])
    assert shown.startswith("~") or shown.startswith("/")
    assert "/" in shown
    assert f"out: {shown}" in text
    assert f"log --data-dir {shown}" in text
    assert f"verify --data-dir {shown}" in text
    assert "--data-dir run" not in text
    assert "run_id:" in text
    assert "observation_id:" in text
    assert "gate final_norm: PASS" in text
    log_buf = io.StringIO()
    writer = FormalWriter.open(
        paths["out"],
        hekb_vnext_root=str(paths["hekb"]),
        gate=door.verdict,
        space_id_of=door.space_id_of,
        seal=door.seal,
        checker_bundle=BUNDLE,
        clock=run_clock,
    )
    try:
        log_ns = SimpleNamespace(
            data_dir=str(paths["out"]), observation=None, json=False, hekb_root=str(paths["hekb"])
        )
        assert cmd_log(log_ns, writer=writer, out=log_buf) == EXIT_OK
        log_text = log_buf.getvalue()
        assert PROMPT not in log_text
        assert str(tmp_path) not in log_text
        json_buf = io.StringIO()
        log_ns.json = True
        assert cmd_log(log_ns, writer=writer, out=json_buf) == EXIT_OK
        assert "observation_id" in json_buf.getvalue()
        verify_buf = io.StringIO()
        verify_ns = SimpleNamespace(
            data_dir=str(paths["out"]), rerun_k=False, hekb_root=str(paths["hekb"])
        )
        verify_code = cmd_verify(verify_ns, writer=writer, out=verify_buf)
        assert verify_code == EXIT_OK
        assert "archive: PASS" in verify_buf.getvalue()
        assert not _tree_has_prompt(paths["out"], PROMPT)
    finally:
        writer.close()


@needs_edge
def test_fake_observe_exits_1_when_the_gate_fails(world: World, tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    door = FakeDoor(Ports(), FailGate())
    code, text = _observe(paths, door)
    assert code == EXIT_FAIL
    assert "gate final_norm: FAIL" in text
    assert PROMPT not in text
    shown = display_path(paths["out"])
    assert f"out: {shown}" in text
    assert f"--data-dir {shown}" in text


def _help(argv: list[str]) -> str:
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        with pytest.raises(SystemExit) as raised:
            _parser().parse_args(argv)
        assert raised.value.code == 0
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_src_cli_help_mentions_the_three_commands() -> None:
    top = _help(["--help"])
    assert "observe" in top and "log" in top and "verify" in top
    observe = _help(["observe", "--help"])
    assert "--model" in observe and "--snapshot" in observe and "--prompt" in observe
    assert "--debug" in observe and "--replicate" in observe
    log = _help(["log", "--help"])
    assert "--data-dir" in log and "--json" in log and "--debug" in log
    verify = _help(["verify", "--help"])
    assert "--data-dir" in verify and "--rerun-k" in verify and "--debug" in verify


def test_display_path_replaces_home_prefix_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    dest = home / "sensos-runs" / "third"
    dest.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda *_args: home.resolve())
    assert display_path(dest) == "~/sensos-runs/third"
    assert display_path(home) == "~"


def test_display_path_keeps_absolute_path_outside_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "elsewhere" / "run"
    outside.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda *_args: home.resolve())
    shown = display_path(outside)
    assert shown == outside.resolve().as_posix()
    assert shown.startswith("/")
    assert not shown.startswith("~")


@needs_edge
def test_observe_prints_tilde_data_dir_not_a_relative_name(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda *_args: home.resolve())
    paths = _layout(tmp_path)
    paths["out"] = home / "sensos-runs" / "third"
    door = FakeDoor(Ports(), PassGate())
    code, text = _observe(paths, door)
    assert code == EXIT_OK
    assert "out: ~/sensos-runs/third" in text
    assert "--data-dir ~/sensos-runs/third" in text
    assert "--data-dir third" not in text
    assert PROMPT not in text


def test_declaration_uses_hekb_graph_not_a_cwd_decoy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    (cwd / "graph").mkdir(parents=True)
    (cwd / "graph" / "__init__.py").write_text("", encoding="utf-8")
    (cwd / "graph" / "formal.py").write_text(
        "def formal_declaration(**_k):\n    raise RuntimeError('cwd graph')\n",
        encoding="utf-8",
    )
    hekb = tmp_path / "hekb"
    (hekb / "graph").mkdir(parents=True)
    (hekb / "graph" / "formal.py").write_text(
        "def formal_declaration(**_k):\n    return {'from': 'hekb'}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd)
    monkeypatch.syspath_prepend(str(cwd))
    kept = {n: sys.modules[n] for n in list(sys.modules) if n == "graph" or n.startswith("graph.")}
    import graph.formal as decoy

    assert decoy.formal_declaration
    try:
        door = FakeDoor(Ports(), PassGate())
        got = _declaration(door, str(hekb))
        assert got == {"from": "hekb"}
    finally:
        for name in list(sys.modules):
            if name == "graph" or name.startswith("graph."):
                del sys.modules[name]
        sys.modules.update(kept)


_WATCHED_TOP = ("graph", "vendor", "storage")


def _forget_hekb_names() -> dict[str, Any]:
    kept: dict[str, Any] = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _WATCHED_TOP:
            kept[name] = sys.modules.pop(name)
    return kept


def _restore_hekb_names(kept: dict[str, Any]) -> None:
    for name in list(sys.modules):
        if name.split(".")[0] in _WATCHED_TOP:
            del sys.modules[name]
    sys.modules.update(kept)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_hekb(root: Path) -> Path:
    _write(
        root / "graph" / "formal.py",
        "from vendor.x.y import z\n"
        "def formal_declaration(**_k):\n"
        "    return {'from': 'hekb', 'z': z}\n",
    )
    _write(root / "vendor" / "x" / "y.py", 'z = "hekb-vendor"\n')
    return root


def _cwd_folders(root: Path, *, regular_vendor: bool = False) -> Path:
    (root / "vendor").mkdir(parents=True)
    (root / "graph").mkdir(parents=True)
    (root / "storage").mkdir(parents=True)
    if regular_vendor:
        (root / "vendor" / "__init__.py").write_text("", encoding="utf-8")
    return root


def _declaration_then_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    regular_vendor: bool,
    extra_path: list[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    hekb = _fake_hekb(tmp_path / "hekb")
    cwd = _cwd_folders(tmp_path / "cwd", regular_vendor=regular_vendor)
    monkeypatch.chdir(cwd)
    saved_path = list(sys.path)
    sys.path[:] = ["", *(extra_path or []), *saved_path]
    monkeypatch.setattr(hekb_formal, "_claims", 0)
    monkeypatch.setattr(hekb_formal, "_claimed_root", None)
    kept = _forget_hekb_names()
    try:
        door = FakeDoor(Ports(), PassGate())
        got = _declaration(door, str(hekb))
        hekb_formal._claim_checkout(str(hekb))
        hekb_formal._release_checkout()
        return got, list(sys.path)
    finally:
        sys.path[:] = saved_path
        _restore_hekb_names(kept)


def test_declaration_then_claim_with_cwd_namespace_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / "keepme.py").write_text("VALUE = 1\n", encoding="utf-8")
    got, path_after = _declaration_then_claim(
        tmp_path, monkeypatch, regular_vendor=False, extra_path=[str(site)]
    )
    assert got == {"from": "hekb", "z": "hekb-vendor"}
    assert str(site) in path_after
    sys.modules.pop("keepme", None)
    sys.path.insert(0, str(site))
    try:
        import keepme

        assert keepme.VALUE == 1
    finally:
        sys.modules.pop("keepme", None)
        if str(site) in sys.path:
            sys.path.remove(str(site))


def test_declaration_then_claim_with_cwd_regular_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    got, _path = _declaration_then_claim(tmp_path, monkeypatch, regular_vendor=True)
    assert got == {"from": "hekb", "z": "hekb-vendor"}


def test_declaration_keeps_site_packages_without_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site-packages"
    site.mkdir()
    _got, path_after = _declaration_then_claim(
        tmp_path, monkeypatch, regular_vendor=False, extra_path=[str(site)]
    )
    assert str(site) in path_after


def test_declaration_drops_namespace_vendor_path_not_plain_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tmp_path / "site-packages"
    plain.mkdir()
    with_vendor = tmp_path / "other-root"
    (with_vendor / "vendor").mkdir(parents=True)
    _got, path_after = _declaration_then_claim(
        tmp_path,
        monkeypatch,
        regular_vendor=False,
        extra_path=[str(plain), str(with_vendor)],
    )
    assert str(plain) in path_after
    assert str(with_vendor) not in path_after


def test_declaration_then_claim_on_real_hekb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hekb = os.environ.get("HEKB_FORMAL_ROOT")
    if not hekb or not (Path(hekb) / "graph" / "formal.py").is_file():
        pytest.skip("HEKB_FORMAL_ROOT has no graph/formal.py")
    cwd = _cwd_folders(tmp_path / "cwd")
    monkeypatch.chdir(cwd)
    saved_path = list(sys.path)
    sys.path[:] = ["", *saved_path]
    monkeypatch.setattr(hekb_formal, "_claims", 0)
    monkeypatch.setattr(hekb_formal, "_claimed_root", None)
    kept = _forget_hekb_names()
    try:
        door = FakeDoor(Ports(), PassGate())
        got = _declaration(door, hekb)
        assert got.get("declaration") == "formal-store/1"
        hekb_formal._claim_checkout(hekb)
        hekb_formal._release_checkout()
    finally:
        sys.path[:] = saved_path
        _restore_hekb_names(kept)


def test_cli_error_includes_the_message() -> None:
    assert _cli_error(RuntimeError("disk full")) == "error: RuntimeError: disk full\n"
    assert _cli_error(RuntimeError()) == "error: RuntimeError\n"


def test_cli_error_replaces_home_prefix_with_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "Users" / "dev"
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda *_args: home.resolve())
    err = RuntimeError(f"loaded from {home / 'vendor'}")
    assert _cli_error(err) == "error: RuntimeError: loaded from ~/vendor\n"


def test_debug_prints_traceback_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("archive missing")

    buf = io.StringIO()
    ns = SimpleNamespace(data_dir=str(tmp_path), rerun_k=False, hekb_root=None, debug=True)
    assert cmd_verify(ns, rerun=boom, out=buf) == EXIT_ERROR
    assert buf.getvalue() == "error: RuntimeError: archive missing\n"
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "archive missing" in err


def test_without_debug_there_is_no_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("archive missing")

    buf = io.StringIO()
    ns = SimpleNamespace(data_dir=str(tmp_path), rerun_k=False, hekb_root=None, debug=False)
    assert cmd_verify(ns, rerun=boom, out=buf) == EXIT_ERROR
    assert "Traceback" not in capsys.readouterr().err


def test_main_drops_cwd_from_sys_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    marker = str(tmp_path / "not-site-packages")
    saved = list(sys.path)
    cwd = tmp_path.resolve()
    sys.path[:] = ["", ".", str(cwd), marker, *[p for p in saved if p not in ("", ".")]]
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    try:
        assert main(["observe", "--model", "llama31"]) == EXIT_ERROR
        assert "" not in sys.path
        assert "." not in sys.path
        resolved = []
        for entry in sys.path:
            try:
                resolved.append(Path(entry).resolve())
            except (OSError, ValueError):
                continue
        assert cwd not in resolved
        assert marker in sys.path
        import nvs_runtime

        assert nvs_runtime.__file__ is not None
        assert Path(nvs_runtime.__file__).resolve() != cwd / "nvs_runtime" / "__init__.py"
    finally:
        sys.path[:] = saved


def test_log_prints_the_error_message(tmp_path: Path) -> None:
    class Boom:
        def records(self) -> list[Any]:
            raise RuntimeError("store is locked")

        def close(self) -> None:
            return None

    buf = io.StringIO()
    ns = SimpleNamespace(
        data_dir=str(tmp_path), observation=None, json=False, hekb_root=str(tmp_path)
    )
    assert cmd_log(ns, writer=Boom(), out=buf) == EXIT_ERROR
    text = buf.getvalue()
    assert text == "error: RuntimeError: store is locked\n"
    assert PROMPT not in text


@needs_edge
def test_observe_prints_the_error_message(world: World, tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    door = FakeDoor(Ports(), PassGate())

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("disk full")

    code, text = _observe(paths, door, writer_factory=boom)
    assert code == EXIT_ERROR
    assert "error: RuntimeError: disk full" in text
    assert PROMPT not in text


def test_verify_prints_the_error_message(tmp_path: Path) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("archive missing")

    buf = io.StringIO()
    ns = SimpleNamespace(data_dir=str(tmp_path), rerun_k=False, hekb_root=None)
    assert cmd_verify(ns, rerun=boom, out=buf) == EXIT_ERROR
    assert buf.getvalue() == "error: RuntimeError: archive missing\n"


@needs_edge
def test_different_prompts_get_different_observation_ids(world: World, tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    door = FakeDoor(Ports(), PassGate())
    first = tmp_path / "a"
    second = tmp_path / "b"
    code_a, text_a = _observe(
        paths, door, ns_changes={"prompt": "prompt alpha", "out": str(first)}
    )
    code_b, text_b = _observe(
        paths, door, ns_changes={"prompt": "prompt beta", "out": str(second)}
    )
    assert code_a == EXIT_OK and code_b == EXIT_OK
    assert _observation_id(text_a) != _observation_id(text_b)
    assert "prompt alpha" not in text_a and "prompt beta" not in text_b
    assert not _tree_has_prompt(first, "prompt alpha")
    assert not _tree_has_prompt(second, "prompt beta")


@needs_edge
def test_same_prompt_and_settings_repeat_the_observation_id(world: World, tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    door = FakeDoor(Ports(), PassGate())
    first = tmp_path / "one"
    second = tmp_path / "two"
    code_a, text_a = _observe(paths, door, ns_changes={"out": str(first)})
    code_b, text_b = _observe(paths, door, ns_changes={"out": str(second)})
    assert code_a == EXIT_OK and code_b == EXIT_OK
    assert _observation_id(text_a) == _observation_id(text_b)
    assert PROMPT not in text_a and PROMPT not in text_b
    assert not _tree_has_prompt(first, PROMPT)
    assert not _tree_has_prompt(second, PROMPT)


@needs_edge
def test_replicate_changes_the_observation_id(world: World, tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    door = FakeDoor(Ports(), PassGate())
    code_a, text_a = _observe(
        paths, door, ns_changes={"out": str(tmp_path / "r0"), "replicate": 0}
    )
    code_b, text_b = _observe(
        paths, door, ns_changes={"out": str(tmp_path / "r1"), "replicate": 1}
    )
    assert code_a == EXIT_OK and code_b == EXIT_OK
    assert _observation_id(text_a) != _observation_id(text_b)
    assert PROMPT not in text_a and PROMPT not in text_b


@needs_edge
def test_max_tokens_changes_the_observation_id(world: World, tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    door = FakeDoor(Ports(), PassGate())
    code_a, text_a = _observe(
        paths, door, ns_changes={"out": str(tmp_path / "t48"), "max_tokens": MAX_TOKENS}
    )
    code_b, text_b = _observe(
        paths, door, ns_changes={"out": str(tmp_path / "t24"), "max_tokens": 24}
    )
    assert code_a == EXIT_OK and code_b == EXIT_OK
    assert _observation_id(text_a) != _observation_id(text_b)
    assert PROMPT not in text_a and PROMPT not in text_b


def test_negative_replicate_is_rejected_at_parse() -> None:
    with pytest.raises(SystemExit) as raised:
        _parser().parse_args(["observe", "--model", "llama31", "--replicate", "-1"])
    assert raised.value.code == 2


def test_log_and_verify_reject_a_missing_directory() -> None:
    log_buf, verify_buf = io.StringIO(), io.StringIO()
    log_ns = SimpleNamespace(data_dir="no-such-run", observation=None, json=False, hekb_root=None)
    verify_ns = SimpleNamespace(data_dir="no-such-run", rerun_k=False, hekb_root=None)
    assert cmd_log(log_ns, out=log_buf) == EXIT_ERROR
    assert cmd_verify(verify_ns, out=verify_buf) == EXIT_ERROR


@needs_edge
def test_next_uses_python_module_when_sensos_prog_is_unset(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SENSOS_PROG", raising=False)
    paths = _layout(tmp_path)
    code, text = _observe(paths, FakeDoor(Ports(), PassGate()))
    shown = display_path(paths["out"])
    assert code == EXIT_OK
    assert (
        f"next: python -m nvs_runtime.hidden_trajectory.cli log --data-dir {shown}" in text
    )
    assert (
        f"next: python -m nvs_runtime.hidden_trajectory.cli verify --data-dir {shown}" in text
    )


@needs_edge
def test_next_uses_sensos_prog_under_home(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prog = home / "sensos-bundle-test" / "sensos"
    prog.parent.mkdir(parents=True)
    prog.write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda *_args: home.resolve())
    monkeypatch.setenv("SENSOS_PROG", str(prog))
    paths = _layout(tmp_path)
    paths["out"] = home / "sensos-runs" / "x"
    code, text = _observe(paths, FakeDoor(Ports(), PassGate()))
    assert code == EXIT_OK
    assert "next: ~/sensos-bundle-test/sensos log --data-dir ~/sensos-runs/x" in text
    assert "next: ~/sensos-bundle-test/sensos verify --data-dir ~/sensos-runs/x" in text
    assert "python -m nvs_runtime.hidden_trajectory.cli" not in text


@needs_edge
def test_next_quotes_sensos_prog_with_spaces_as_home(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prog = home / "my bundle" / "sensos"
    prog.parent.mkdir(parents=True)
    prog.write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda *_args: home.resolve())
    monkeypatch.setenv("SENSOS_PROG", str(prog))
    paths = _layout(tmp_path)
    paths["out"] = home / "sensos-runs" / "x"
    code, text = _observe(paths, FakeDoor(Ports(), PassGate()))
    assert code == EXIT_OK
    assert 'next: "$HOME/my bundle/sensos" log --data-dir ~/sensos-runs/x' in text
    assert 'next: "$HOME/my bundle/sensos" verify --data-dir ~/sensos-runs/x' in text


@needs_edge
def test_next_uses_absolute_sensos_prog_outside_home(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    prog = tmp_path / "elsewhere" / "sensos"
    prog.parent.mkdir()
    prog.write_text("", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda *_args: home.resolve())
    monkeypatch.setenv("SENSOS_PROG", str(prog))
    paths = _layout(tmp_path)
    code, text = _observe(paths, FakeDoor(Ports(), PassGate()))
    shown = display_path(paths["out"])
    prog_shown = prog.resolve().as_posix()
    assert code == EXIT_OK
    assert f"next: {prog_shown} log --data-dir {shown}" in text
    assert f"next: {prog_shown} verify --data-dir {shown}" in text
    assert not prog_shown.startswith("~")
    assert "python -m nvs_runtime.hidden_trajectory.cli" not in text


_EMPTY_CAUSAL = "causal log: empty (this build records no causal log)"


def _wide_adapter(paths: dict[str, Path], n_gen: int = 2) -> LlamaAdapter:
    """A fake adapter whose hidden states are 4096-D, so route K actually runs."""
    from u65_support import make_capture

    adapter = LlamaAdapter(LlamaProvider(snapshot_dir=paths["snap"]))
    adapter._capture = make_capture(kind="llama", dimension=4096, n_gen=n_gen)
    return adapter


def _stored_causal(paths: dict[str, Path], door: FakeDoor, text: str) -> bytes:
    writer = FormalWriter.open(
        paths["out"],
        hekb_vnext_root=str(paths["hekb"]),
        gate=door.verdict,
        space_id_of=door.space_id_of,
        seal=door.seal,
        checker_bundle=BUNDLE,
        clock=run_clock,
    )
    try:
        got = writer.observation(_observation_id(text))
        blob = got["positions"]["final_norm"]["k"]["causal_log"]
        assert isinstance(blob, bytes)
        return blob
    finally:
        writer.close()


def _log_text(paths: dict[str, Path], door: FakeDoor) -> str:
    buf = io.StringIO()
    writer = FormalWriter.open(
        paths["out"],
        hekb_vnext_root=str(paths["hekb"]),
        gate=door.verdict,
        space_id_of=door.space_id_of,
        seal=door.seal,
        checker_bundle=BUNDLE,
        clock=run_clock,
    )
    try:
        log_ns = SimpleNamespace(
            data_dir=str(paths["out"]),
            observation=None,
            json=False,
            hekb_root=str(paths["hekb"]),
        )
        assert cmd_log(log_ns, writer=writer, out=buf) == EXIT_OK
        return buf.getvalue()
    finally:
        writer.close()


def test_empty_causal_line_is_only_for_an_empty_saved_log() -> None:
    from nvs_runtime.hidden_trajectory.cli import _empty_causal_line

    line = _EMPTY_CAUSAL + "\n"
    assert _empty_causal_line([b"[]"]) == line
    assert _empty_causal_line([b"[{},{}]"]) == line
    assert _empty_causal_line([b"[{}]"]) == line
    assert _empty_causal_line([b"[{}, {}]"]) == line
    assert _empty_causal_line([b'[{"ports": {}}]']) == ""
    assert _empty_causal_line([b"[{}, {}]", b'[{"a": 1}]']) == ""
    assert _empty_causal_line([]) == ""
    assert _empty_causal_line([b"not json"]) == ""
    assert _empty_causal_line([b'{"a": {}}']) == ""
