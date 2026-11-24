"""The observe CLI: model selection, capture, Formal Store, then log and verify.

``python -m nvs_runtime.hidden_trajectory.cli observe|log|verify``. Src never imports
``sensos_inference`` (by design, no cross-imports): the Adapter and ``CaptureRequest`` are
loaded by name. Nothing here
imports mlx.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, TextIO, cast

from nvs_observation.adapter_session import AdapterSession

from nvs_runtime.hidden_trajectory.adapter_loader import (
    _MISSING_MESSAGE,
    AdapterLoadError,
    load_adapter,
)
from nvs_runtime.hidden_trajectory.envelope import EnvelopeIds
from nvs_runtime.hidden_trajectory.formal_writer import FormalWriter
from nvs_runtime.hidden_trajectory.gate_client import GateClient, GateProcess, git_head
from nvs_runtime.hidden_trajectory.model_catalog import GREEDY, MAX_TOKENS, MODELS
from nvs_runtime.hidden_trajectory.rerun import rerun_from_store

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


class ObserveDoor:
    """The Gate and id operations the FormalWriter needs. Thin wrap of ``GateClient``."""

    def __init__(self, gate_py: Path, sensos_tools: Path) -> None:
        self.gate_py = Path(gate_py)
        self.sensos_tools = Path(sensos_tools)

    @classmethod
    def from_paths(cls, gate_py: str | Path, sensos_tools: str | Path) -> ObserveDoor:
        return cls(Path(gate_py), Path(sensos_tools))

    @cached_property
    def client(self) -> GateClient:
        return GateClient(self.gate_py, self.sensos_tools)

    def call(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return GateProcess(self.gate_py, self.sensos_tools).call(request)

    def space_id_of(self, materials: Mapping[str, Any]) -> str:
        return str(self.call({"op": "hid_space_id", "materials": dict(materials)})["space_id"])

    def seal(self, envelope: dict[str, Any], lookup: Mapping[str, Any]) -> dict[str, Any]:
        sealed = self.call({"op": "seal", "envelope": envelope, "lookup": dict(lookup)})["envelope"]
        assert isinstance(sealed, dict)
        return sealed

    def session_id(
        self, *, spec: str, item_id: str, condition: str, replicate: int, rng_seed: int
    ) -> str:
        request = {"op": "session_id", "spec": spec, "item_id": item_id, "condition": condition}
        return str(self.call({**request, "replicate": replicate, "rng_seed": rng_seed})["id"])

    def observation_id(self, session_id: str, ordinal: int = 0) -> str:
        request = {"op": "observation_id", "session_id": session_id, "ordinal": ordinal}
        return str(self.call(request)["id"])

    def verdict(
        self,
        envelope: Mapping[str, Any],
        *,
        lookup: Mapping[str, Any] | None = None,
        stored: Mapping[str, Any] | None = None,
        blobs: Mapping[str, bytes] | None = None,
        contract_facts: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        return self.client.verdict(
            envelope, lookup=lookup, stored=stored, blobs=blobs, contract_facts=contract_facts
        )


_DEFAULT_PROG = "python -m nvs_runtime.hidden_trajectory.cli"


def display_path(path: Path) -> str:
    """Printable dest: home as ``~``, otherwise the absolute path. Copyable from any cwd."""
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(Path.home().resolve())
    except ValueError:
        return resolved.as_posix()
    posix = relative.as_posix()
    return "~" if posix == "." else f"~/{posix}"


def _shell_display(shown: str) -> str:
    """Quote a printable path so it pastes into a shell. Spaces only.

    A leading ``~`` is left as-is when there is no space (the shell expands it).
    Quoted ``~`` does not expand, so a tilde path with a space becomes ``"$HOME/..."``.
    """
    if shown.startswith("~"):
        if " " not in shown and "\t" not in shown:
            return shown
        return '"$HOME' + shown[1:] + '"'
    if " " in shown or "\t" in shown:
        return shlex.quote(shown)
    return shown


def _cli_prog() -> str:
    raw = os.environ.get("SENSOS_PROG")
    if not raw:
        return _DEFAULT_PROG
    return _shell_display(display_path(Path(raw)))


def _next_line(verb: str, data_dir: Path) -> str:
    return f"next: {_cli_prog()} {verb} --data-dir {_shell_display(display_path(data_dir))}\n"


def _home_as_tilde(text: str) -> str:
    home = str(Path.home().resolve())
    if text == home:
        return "~"
    return text.replace(home + "/", "~/").replace(home, "~")


def _cli_error(error: BaseException) -> str:
    """One line: the type and the message. Home as ``~``. No traceback."""
    name = type(error).__name__
    detail = str(error).strip()
    if detail:
        detail = _home_as_tilde(detail.splitlines()[0].strip())
    if not detail or detail == name:
        return f"error: {name}\n"
    return f"error: {name}: {detail}\n"


def _emit_error(stream: TextIO, error: BaseException, ns: Any) -> None:
    stream.write(_cli_error(error))
    if getattr(ns, "debug", False):
        traceback.print_exc(file=sys.stderr)


_HEKB_PACKAGES = ("graph", "vendor")


def _is_under(place: str, root: Path) -> bool:
    try:
        return Path(place).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _drop_foreign_hekb_modules(root: Path) -> None:
    """Forget ``graph`` / ``vendor`` loaded from somewhere other than ``root`` (cwd included)."""
    for name in list(sys.modules):
        if name.split(".")[0] not in _HEKB_PACKAGES:
            continue
        module = sys.modules[name]
        file = getattr(module, "__file__", None) or ""
        paths = [str(p) for p in getattr(module, "__path__", [])]
        places = [p for p in (file, *paths) if p]
        if not places or not any(_is_under(p, root) for p in places):
            del sys.modules[name]


def _is_cwd_path_entry(entry: str, cwd: Path | None = None) -> bool:
    if entry in ("", "."):
        return True
    here = cwd if cwd is not None else Path.cwd().resolve()
    try:
        return Path(entry).resolve() == here
    except (OSError, ValueError):
        return False


def _drop_cwd_from_sys_path() -> None:
    """Same effect as ``python -P``: cwd is not searched for later imports."""
    cwd = Path.cwd().resolve()
    sys.path[:] = [p for p in sys.path if not _is_cwd_path_entry(p, cwd)]


def _has_namespace_hekb(entry: str) -> bool:
    """True if this path holds ``graph`` or ``vendor`` as a namespace package (no
    ``__init__.py``)."""
    try:
        base = Path(entry)
        for name in _HEKB_PACKAGES:
            folder = base / name
            if folder.is_dir() and not (folder / "__init__.py").is_file():
                return True
    except OSError:
        return False
    return False


def _path_for_hekb_import(saved: list[str], root: Path) -> list[str]:
    """``root`` first; drop cwd and namespace ``graph`` / ``vendor`` paths. Keep site-packages."""
    root_text = str(root)
    cwd = Path.cwd().resolve()
    out = [root_text]
    for entry in saved:
        if entry == root_text:
            continue
        if _is_cwd_path_entry(entry, cwd):
            continue
        try:
            if Path(entry).resolve() == root:
                continue
        except (OSError, ValueError):
            pass
        if _has_namespace_hekb(entry):
            continue
        out.append(entry)
    return out


def _now() -> datetime:
    return datetime.now().astimezone().replace(microsecond=0)


def _run_id(model: str, clock: datetime) -> str:
    return f"{model}-{clock.strftime('%Y%m%d-%H%M%S')}"


def _item_id_for_prompt(prompt: str) -> str:
    """First 16 hex digits of the prompt sha256. The prompt text is not kept."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _run_spec_hash(spec: Mapping[str, Any], max_tokens: int) -> str:
    """sha256 of the run settings as sorted JSON with no spaces (64 hex digits)."""
    payload = {
        "build_id": spec["build_id"],
        "max_tokens": max_tokens,
        "points": list(spec["points"]),
        "sampling": {"seed": GREEDY["seed"], "top_k": GREEDY["top_k"]},
    }
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replicate_value(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("replicate must be an integer >= 0") from error
    if value < 0:
        raise argparse.ArgumentTypeError("replicate must be an integer >= 0")
    return value


def _read_prompt(text: str | None, path: str | None) -> str:
    if text is not None and path is not None:
        raise ValueError("give --prompt or --prompt-file, not both")
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    if text is None:
        raise ValueError("give --prompt or --prompt-file")
    return text


def missing_observe(ns: argparse.Namespace) -> list[str]:
    """What observe still needs, before any Adapter or model is loaded."""
    missing: list[str] = []
    if ns.model not in MODELS:
        missing.append(f"--model {ns.model!r} is not llama31 or qwen35")
        return missing
    spec = MODELS[ns.model]
    if not ns.prompt and not ns.prompt_file:
        missing.append("--prompt or --prompt-file")
    if ns.prompt and ns.prompt_file:
        missing.append("give only one of --prompt and --prompt-file")
    if ns.prompt_file and not Path(ns.prompt_file).is_file():
        missing.append(f"--prompt-file is not a file ({display_path(Path(ns.prompt_file))})")
    snapshot = ns.snapshot or os.environ.get(str(spec["snapshot_env"]))
    if not snapshot:
        missing.append(f"--snapshot or {spec['snapshot_env']}")
    elif not Path(snapshot).is_dir():
        missing.append("--snapshot is not a directory")
    gate_py = ns.gate_python or os.environ.get("GATE_PY")
    if not gate_py:
        missing.append("--gate-python or GATE_PY")
    elif not Path(gate_py).is_file():
        missing.append("GATE_PY is not a file")
    tools = ns.sensos_tools or os.environ.get("SENSOS_TOOLS")
    if not tools:
        missing.append("--sensos-tools or SENSOS_TOOLS")
    elif not (Path(tools) / "contracts" / "gate_nvs.py").is_file():
        missing.append("SENSOS_TOOLS has no contracts/gate_nvs.py")
    hekb = ns.hekb_root or os.environ.get("HEKB_FORMAL_ROOT")
    if not hekb:
        missing.append("--hekb-root or HEKB_FORMAL_ROOT")
    elif not (Path(hekb) / "graph" / "formal.py").is_file():
        missing.append("HEKB_FORMAL_ROOT has no graph/formal.py")
    return missing


def _capture_request(
    spec: Mapping[str, Any],
    max_tokens: int,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Any:
    try:
        observation = import_module("sensos_inference.observation")
    except ModuleNotFoundError as error:
        raise AdapterLoadError(_MISSING_MESSAGE) from error
    return observation.CaptureRequest(
        points=tuple(spec["points"]), max_tokens=max_tokens, sampling=dict(GREEDY)
    )


def _declaration(door: Any, hekb_root: str, created_by: str = "sensos-observe") -> dict[str, Any]:
    catalog = door.client.process.call({"op": "catalog"})
    tools_parent = Path(getattr(door, "sensos_tools", hekb_root)).resolve().parent
    version = git_head(tools_parent) or "unknown"
    root = Path(hekb_root).resolve()
    _drop_foreign_hekb_modules(root)
    sys.path[:] = _path_for_hekb_import(list(sys.path), root)
    formal = importlib.import_module("graph.formal")
    declaration: dict[str, Any] = formal.formal_declaration(
        catalog_rows=catalog["rows"], catalog_version=version, created_by=created_by
    )
    return declaration


_MEANING_SCHEMAS: tuple[dict[str, str], ...] = (
    {"id": "nvs.token-temporal", "version": "0.1.0"},
    {"id": "nvs.meaning-frame", "version": "0.1.0"},
    {"id": "nvs.meaning-measurement", "version": "0.1.0"},
)
_FRAME_NOT_SUPPLIED: dict[str, str] = {
    "check": "frame_binding",
    "reason": "frame_not_supplied",
}


def meaning_declaration(
    door: Any, hekb_root: str, created_by: str = "sensos-observe"
) -> dict[str, Any]:
    """A Formal Store declaration that accepts a meaning measurement and its frame.

    ``catalog_rows`` are the live catalogue of the Gate child. ``catalog_version`` is the
    full commit of the sensos checkout that child reads. ``_declaration`` is unchanged and
    still uses the four default schemas.
    """
    catalog = door.client.process.call({"op": "catalog"})
    tools = getattr(door, "sensos_tools", None)
    if not isinstance(tools, str) or not tools:
        raise RuntimeError("sensos tools path is missing; catalog_version cannot be read")
    version = git_head(Path(tools).resolve().parent)
    if version is None:
        raise RuntimeError("sensos HEAD is not a full commit; catalog_version cannot be unknown")
    root = Path(hekb_root).resolve()
    _drop_foreign_hekb_modules(root)
    sys.path[:] = _path_for_hekb_import(list(sys.path), root)
    formal = importlib.import_module("graph.formal")
    schemas = [dict(item) for item in formal.DEFAULT_PAYLOAD_SCHEMAS]
    schemas.extend(dict(item) for item in _MEANING_SCHEMAS)
    allow = [dict(item) for item in formal.DEFAULT_ALLOW_UNVERIFIED]
    allow.append(dict(_FRAME_NOT_SUPPLIED))
    declaration: dict[str, Any] = formal.formal_declaration(
        catalog_rows=catalog["rows"],
        catalog_version=version,
        created_by=created_by,
        accepted_payload_schemas=schemas,
        allow_unverified=allow,
    )
    return declaration


def _gate_rows(record: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for point, entry in (record.get("positions") or {}).items():
        gate = entry.get("gate") if isinstance(entry, dict) else None
        status = str((gate or {}).get("status") or "not_called")
        reason = ""
        if isinstance(gate, dict):
            violations = gate.get("violations") or []
            if violations and isinstance(violations[0], dict):
                reason = str(violations[0].get("message") or violations[0].get("code") or "")
        rows.append((str(point), status, reason))
    return rows


def _k_sha(record: Mapping[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for point, entry in (record.get("positions") or {}).items():
        k = entry.get("k_path") if isinstance(entry, dict) else None
        if isinstance(k, dict):
            out[str(point)] = k.get("causal_log_sha256")
        else:
            out[str(point)] = None
    return out


_EMPTY_CAUSAL_LINE = "causal log: empty (this build records no causal log)\n"


def _causal_blob_is_empty(blob: bytes) -> bool:
    """True when the stored causal log is ``[]`` or every entry is ``{}``."""
    try:
        entries = json.loads(blob)
    except (json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(entries, list):
        return False
    return all(entry == {} for entry in entries)


def _observation_causal_blobs(got: Mapping[str, Any]) -> list[bytes]:
    """The saved causal blobs of one observation (``FormalWriter.observation``)."""
    blobs: list[bytes] = []
    for item in (got.get("positions") or {}).values():
        held = item.get("k") if isinstance(item, dict) else None
        if not isinstance(held, dict):
            continue
        blob = held.get("causal_log")
        if isinstance(blob, (bytes, bytearray)) and blob:
            blobs.append(bytes(blob))
    return blobs


def _empty_causal_line(blobs: Sequence[bytes]) -> str:
    """One line when every saved causal log is empty; nothing otherwise."""
    if blobs and all(_causal_blob_is_empty(blob) for blob in blobs):
        return _EMPTY_CAUSAL_LINE
    return ""


def cmd_observe(
    ns: argparse.Namespace,
    *,
    adapter: Any = None,
    door: Any = None,
    declaration: Mapping[str, Any] | None = None,
    writer_factory: Callable[..., FormalWriter] | None = None,
    request: Any = None,
    clock: Callable[[], datetime] | None = None,
    import_module: Callable[[str], Any] = importlib.import_module,
    out: TextIO | None = None,
) -> int:
    """Run one capture into a new Formal Store. 0 all Gate PASS, 1 a FAIL, 2 an error."""
    stream = out or sys.stdout
    missing = missing_observe(ns)
    if missing:
        stream.write("missing:\n")
        for item in missing:
            stream.write(f"  {item}\n")
        return EXIT_ERROR
    spec = MODELS[ns.model]
    try:
        prompt = _read_prompt(ns.prompt, ns.prompt_file)
    except (OSError, ValueError) as error:
        stream.write(f"error: {error}\n")
        return EXIT_ERROR
    messages = [{"role": "user", "content": prompt}]
    snapshot = Path(ns.snapshot or os.environ[str(spec["snapshot_env"])])
    gate_py = ns.gate_python or os.environ["GATE_PY"]
    tools = ns.sensos_tools or os.environ["SENSOS_TOOLS"]
    hekb = ns.hekb_root or os.environ["HEKB_FORMAL_ROOT"]
    tick = clock or _now
    started = tick()
    run_id = _run_id(ns.model, started)
    data_dir = Path(ns.out) if ns.out else Path("sensos-runs") / run_id
    empty_causal = ""
    try:
        data_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        stream.write(f"error: output already exists ({display_path(data_dir)})\n")
        return EXIT_ERROR
    except OSError as error:
        stream.write(f"error: cannot create output ({type(error).__name__}: {error})\n")
        return EXIT_ERROR
    try:
        if adapter is None:
            adapter = load_adapter(spec, snapshot, import_module=import_module)
        if request is None:
            request = _capture_request(spec, ns.max_tokens, import_module=import_module)
        if door is None:
            door = ObserveDoor.from_paths(gate_py, tools)
        produced_at = started.strftime("%Y-%m-%dT%H:%M:%SZ")
        session_id = door.session_id(
            spec=_run_spec_hash(spec, ns.max_tokens),
            item_id=_item_id_for_prompt(prompt),
            condition="run",
            replicate=int(getattr(ns, "replicate", 0)),
            rng_seed=0,
        )
        observation_id = door.observation_id(session_id, 0)
        sequence = iter(range(1, 10**6))
        ids = EnvelopeIds(session_id, observation_id, produced_at, lambda: next(sequence))
        captured = AdapterSession(adapter).capture_with_origin(
            messages,
            request,
            model_build_id=str(spec["build_id"]),
            compute_environment=str(spec["compute"]),
        )
        decl = declaration if declaration is not None else _declaration(door, hekb)
        factory = writer_factory or FormalWriter.create
        writer = factory(
            data_dir,
            decl,
            hekb_vnext_root=hekb,
            gate=door.verdict,
            space_id_of=door.space_id_of,
            seal=door.seal,
            checker_bundle=door.client.checker_bundle,
            clock=lambda: started,
            k_path=True,
        )
        try:
            saved = writer.save(
                captured, contract_session_id=f"observe-{ns.model}", synthetic=False, ids=ids
            )
            empty_causal = _empty_causal_line(
                _observation_causal_blobs(
                    writer.observation(str(saved.record.get("observation_id")))
                )
            )
        finally:
            writer.close()
    except AdapterLoadError as error:
        stream.write(f"error: {error}\n")
        if getattr(ns, "debug", False):
            traceback.print_exc(file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:  # noqa: BLE001 -- the CLI turns any failure into exit 2
        _emit_error(stream, error, ns)
        return EXIT_ERROR
    record = saved.record
    gates = _gate_rows(record)
    k_sha = _k_sha(record)
    failed = [point for point, status, _reason in gates if status == "FAIL"]
    stream.write(f"run_id: {run_id}\n")
    stream.write(f"out: {display_path(data_dir)}\n")
    stream.write(f"observation_id: {record.get('observation_id')}\n")
    for point, status, _reason in gates:
        stream.write(f"gate {point}: {status}\n")
    for point, digest in k_sha.items():
        stream.write(f"k {point}: {digest or 'none'}\n")
    if empty_causal:
        stream.write(empty_causal)
    stream.write(_next_line("log", data_dir))
    stream.write(_next_line("verify", data_dir))
    return EXIT_FAIL if failed else EXIT_OK


def _log_body(got: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    record = got.get("record") or {}
    capture = record.get("capture") or {}
    gates = _gate_rows(record) if record else []
    k_sha = _k_sha(record) if record else {}
    ports: dict[str, Any] = {}
    for point, entry in (record.get("positions") or {}).items():
        k = entry.get("k_path") if isinstance(entry, dict) else None
        if isinstance(k, dict):
            ports[str(point)] = {
                "status": k.get("status"),
                "n_records": k.get("n_records"),
                "integrity": (k.get("integrity") or {}).get("chain_and_crc"),
            }
    return {
        "observation_id": got.get("observation_id"),
        "model_build_id": capture.get("model_build_id"),
        "adapter": capture.get("adapter"),
        "points": capture.get("points"),
        "n_in": capture.get("n_in"),
        "n_gen": capture.get("n_gen"),
        "finish_reason": capture.get("finish_reason"),
        "gate": [{"point": p, "status": s, "reason": r} for p, s, r in gates],
        "k_sha256": k_sha,
        "ports": ports,
        "records": len(records),
        "missing": list(got.get("missing") or []),
    }


def cmd_log(
    ns: argparse.Namespace,
    *,
    writer: FormalWriter | None = None,
    out: TextIO | None = None,
) -> int:
    stream = out or sys.stdout
    data_dir = Path(ns.data_dir)
    if not data_dir.is_dir():
        stream.write("error: --data-dir is not a directory\n")
        return EXIT_ERROR
    hekb = getattr(ns, "hekb_root", None) or os.environ.get("HEKB_FORMAL_ROOT")
    if writer is None and not hekb:
        stream.write("missing:\n  --hekb-root or HEKB_FORMAL_ROOT\n")
        return EXIT_ERROR
    try:
        opened = writer or FormalWriter.open(
            data_dir, hekb_vnext_root=cast(str | os.PathLike[str], hekb)
        )
        try:
            records = opened.records()
            if not records:
                stream.write("error: no capture records\n")
                return EXIT_ERROR
            observation_id = ns.observation or records[-1]["observation_id"]
            got = opened.observation(observation_id)
        finally:
            if writer is None:
                opened.close()
    except Exception as error:  # noqa: BLE001
        _emit_error(stream, error, ns)
        return EXIT_ERROR
    body = _log_body(got, records)
    if ns.json:
        stream.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return EXIT_OK
    stream.write(f"observation_id: {body['observation_id']}\n")
    stream.write(f"model: {body['model_build_id']}\n")
    stream.write(f"adapter: {body['adapter']}\n")
    stream.write(f"points: {', '.join(body['points'] or [])}\n")
    stream.write(
        f"tokens: n_in={body['n_in']} n_gen={body['n_gen']} finish={body['finish_reason']}\n"
    )
    for row in body["gate"]:
        extra = f" ({row['reason']})" if row["reason"] else ""
        stream.write(f"gate {row['point']}: {row['status']}{extra}\n")
    for point, digest in (body["k_sha256"] or {}).items():
        stream.write(f"k {point}: {digest or 'none'}\n")
    empty_causal = _empty_causal_line(_observation_causal_blobs(got))
    if empty_causal:
        stream.write(empty_causal)
    for point, ports in (body["ports"] or {}).items():
        stream.write(
            f"ports {point}: status={ports['status']} n={ports['n_records']} "
            f"chain={ports['integrity']}\n"
        )
    stream.write(f"records.jsonl: {body['records']} record(s)\n")
    return EXIT_OK


def cmd_verify(
    ns: argparse.Namespace,
    *,
    rerun: Callable[..., dict[str, Any]] = rerun_from_store,
    writer: FormalWriter | None = None,
    out: TextIO | None = None,
) -> int:
    stream = out or sys.stdout
    data_dir = Path(ns.data_dir)
    if not data_dir.is_dir():
        stream.write("error: --data-dir is not a directory\n")
        return EXIT_ERROR
    try:
        report = rerun(data_dir, rerun_k=bool(ns.rerun_k))
    except Exception as error:  # noqa: BLE001
        _emit_error(stream, error, ns)
        return EXIT_ERROR
    items: list[tuple[str, str]] = []
    items.append(("archive", "PASS" if report.get("ok") else "FAIL"))
    hekb = getattr(ns, "hekb_root", None) or os.environ.get("HEKB_FORMAL_ROOT")
    gate_fail = False
    try:
        if writer is not None or hekb:
            opened = writer or FormalWriter.open(
            data_dir, hekb_vnext_root=cast(str | os.PathLike[str], hekb)
        )
            try:
                for record in opened.records():
                    for point, status, _reason in _gate_rows(record):
                        items.append((f"gate {point}", status))
                        if status == "FAIL":
                            gate_fail = True
                    for point, entry in (record.get("positions") or {}).items():
                        k = entry.get("k_path") if isinstance(entry, dict) else None
                        if isinstance(k, dict) and k.get("status") == "measured":
                            items.append((f"k {point}", "PASS"))
            finally:
                if writer is None:
                    opened.close()
    except Exception as error:  # noqa: BLE001
        _emit_error(stream, error, ns)
        return EXIT_ERROR
    if ns.rerun_k:
        for capture in report.get("captures") or []:
            k_report = capture.get("k") or {}
            for point, info in k_report.items():
                if isinstance(info, dict) and "rerun_k" in info:
                    same = bool((info.get("rerun_k") or {}).get("same_as_stored"))
                    items.append((f"rerun_k {point}", "PASS" if same else "FAIL"))
    for name, status in items:
        stream.write(f"{name}: {status}\n")
    unchecked = int(report.get("k_unchecked") or 0)
    if unchecked >= 1:
        stream.write(f"K: {unchecked} 位置は検査していません（Developer Tools が入っていません）\n")
    failed = [name for name, status in items if status == "FAIL"]
    if not report.get("ok") or gate_fail or failed:
        return EXIT_FAIL
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nvs_runtime.hidden_trajectory.cli",
        description="Observe a model, then read and verify the stored run.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    observe = sub.add_parser("observe", help="capture, Gate, Formal Store, path K")
    observe.add_argument("--model", required=True, choices=sorted(MODELS))
    observe.add_argument("--snapshot", default=None)
    observe.add_argument("--prompt", default=None)
    observe.add_argument("--prompt-file", default=None)
    observe.add_argument("--out", default=None)
    observe.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    observe.add_argument("--gate-python", default=None)
    observe.add_argument("--sensos-tools", default=None)
    observe.add_argument("--hekb-root", default=None)
    observe.add_argument("--replicate", type=_replicate_value, default=0)
    observe.add_argument("--debug", action="store_true")

    log = sub.add_parser("log", help="read a stored observation")
    log.add_argument("--data-dir", required=True)
    log.add_argument("--observation", default=None)
    log.add_argument("--json", action="store_true")
    log.add_argument("--hekb-root", default=None)
    log.add_argument("--debug", action="store_true")

    verify = sub.add_parser("verify", help="check a stored run from its archive")
    verify.add_argument("--data-dir", required=True)
    verify.add_argument("--rerun-k", action="store_true")
    verify.add_argument("--hekb-root", default=None)
    verify.add_argument("--debug", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _drop_cwd_from_sys_path()
    parser = _parser()
    ns = parser.parse_args(argv)
    if ns.command == "observe":
        return cmd_observe(ns)
    if ns.command == "log":
        return cmd_log(ns)
    if ns.command == "verify":
        return cmd_verify(ns)
    return EXIT_ERROR


def entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entry()
