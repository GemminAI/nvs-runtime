"""The child process of the Gate (design U6T-1 v0.1 4): the sensos contracts behind a JSON door.

**Never imported by nvs-runtime's own process** (by design, no cross-imports). It is a
script, run by ``GateClient`` in ``GATE_PY`` (an interpreter with ``jsonschema``), and it imports
the sensos checkout named by ``SENSOS_TOOLS`` (its ``tools/`` directory). It is a file of the
package so that it travels with it and enters the checker bundle (``gate-checker-bundle/1``);
nothing in it imports ``nvs_runtime``. Standard library and sensos only.

stdin: one JSON object with an ``op``; stdout: one JSON object.

* ``hid_space_id`` ``{materials}`` -> ``{space_id}``                       (contracts.hid_space)
* ``seal`` ``{envelope, lookup}`` -> ``{envelope}`` with its two ids         (contracts.ids)
* ``session_id`` / ``observation_id`` -> ``{id}``                          (contracts.ids)
* ``check`` ``{envelope, lookup, stored, blobs (base64), contract_facts}`` ->
  ``{l1_codes, l1_messages, payload, window}`` (the form of the U6-4 door, unchanged)
* ``gate``: the same request; ``{l1: [{code, message}], payload, window, envelope_sha256,
  lookup_sha256, contract_facts_sha256}``. The three hashes are SHA-256 of the JCS
  (``contracts.jcs``, RFC 8785) of the envelope, of the lookup and of the contract facts (``null``
  for none): sensos's JCS is the only one, nothing is copied (design N-2).
* ``catalog`` -> ``{rows: [[payload_schema, check, reason], ...], sha256}``: the UNVERIFIABLE
  catalogue (``contracts.hidden_trajectory.UNVERIFIABLE_CATALOG``, DC-7), rows in the order of
  the code points; ``sha256`` is that of the JCS of the rows as ``{payload_schema, check, reason}``.
* ``jcs_sha256`` ``{value}`` -> ``{sha256}``.
* ``checker_files`` ``{extra_files: [{repo, path, b64}]}``: makes the checker bundle. It first runs
  a fixed warm-up that loads, through the public functions of the checker, every file it can
  read (the schemas of ``common/``, ``IC-00/1.1.0/``, ``msr/``, ``nvs/``, the row rules); the
  bundle holds the ``.py`` files of the modules loaded from ``SENSOS_TOOLS`` and the other files
  the process opened below the sensos root, plus ``extra_files`` (nvs-runtime's own). The answer:
  ``{bundle_b64, bundle_sha256, files: [{repo, path, sha256}], runtime}``.

``check`` and ``gate`` run the envelope rule first (IC-00 by ``contract.version``). Only if it
reports nothing is the payload checked, by ``gate_nvs.check_nvs_payload``: ``payload`` is then
``{kind, status, violations, unverifiable}``, else the string ``"not_run"``. For
``msr.trajectory-window`` 0.10.0 the space of the window is checked as well and reported under
``window``. Any request may carry ``"trace": true``; the answer then has ``_trace`` (the files
below the sensos root that the process opened or imported), for the test of the bundle.

A ``contract_facts`` of the wrong shape is an error of the caller, not a violation of the
envelope: the exit status is 2 and stdout is ``{"error": "gate_internal_error", "message": ...}``.
"""

from __future__ import annotations

import base64
import builtins
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import sys
from typing import Any, Final

_TOOLS = os.environ.get("SENSOS_TOOLS") or sys.exit(
    "SENSOS_TOOLS is not set (tools/ of a sensos checkout with contracts/gate_nvs.py)"
)
sys.path.insert(0, _TOOLS)

_OPENED: list[str] = []
_real_open = io.open


def _recording_open(file: Any, *args: Any, **kwargs: Any) -> Any:
    if isinstance(file, str | os.PathLike):
        _OPENED.append(os.path.realpath(file))
    return _real_open(file, *args, **kwargs)


io.open = _recording_open
builtins.open = _recording_open


#: The only modules this script ever imports from sensos: the names are fixed here, in the file.
#: Nothing of the request (``op`` or anything else) is used to make a name for ``import_module``.
_ALLOWED_MODULES: Final = (
    "contracts.contract_facts",
    "contracts.gate",
    "contracts.gate_nvs",
    "contracts.hid_space",
    "contracts.hidden_trajectory",
    "contracts.hidden_trajectory_window",
    "contracts.ic00_versions",
    "contracts.ids",
    "contracts.jcs",
    "contracts.schemas",
)


def _sensos(name: str) -> Any:
    """A module of the sensos contracts, by its short name; only a name of ``_ALLOWED_MODULES``.

    Loaded by name, not by an ``import`` statement: this file is a script of the package that runs
    in another interpreter, and the static test of the package (``test_e09``) keeps every
    ``import`` of its modules free of sensos names. The argument of ``import_module`` is always an
    element of the constant tuple above, never a string built from ``name``."""
    for allowed in _ALLOWED_MODULES:
        if allowed == f"contracts.{name}":
            return importlib.import_module(allowed)
    raise SystemExit(f"module not allowed: {name!r}")


gate = _sensos("gate")
gate_nvs = _sensos("gate_nvs")
ids = _sensos("ids")
schemas = _sensos("schemas")
ContractFactsError = _sensos("contract_facts").ContractFactsError
hid_space_id = _sensos("hid_space").hid_space_id
UNVERIFIABLE_CATALOG = _sensos("hidden_trajectory").UNVERIFIABLE_CATALOG
HiddenTrajectoryResult = _sensos("hidden_trajectory").HiddenTrajectoryResult
row_rule_registry = _sensos("hidden_trajectory").row_rule_registry
check_window_space = _sensos("hidden_trajectory_window").check_window_space
RULES = _sensos("ic00_versions").RULES
check_envelope_versioned = _sensos("ic00_versions").check_envelope_versioned
jcs = _sensos("jcs").jcs

WINDOW_SCHEMA = ("msr.trajectory-window", "0.10.0")
BUNDLE_VERSION = "gate-checker-bundle/1"
SENSOS = "sensos"
_ROOT = os.path.dirname(os.path.realpath(_TOOLS))


def _sha(value: Any) -> str:
    return hashlib.sha256(jcs(value)).hexdigest()


def _violations(found: Any) -> list[dict[str, str]]:
    return [{"code": v.code, "message": v.message} for v in found]


def _result(kind: str, result: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "status": result.status,
        "violations": _violations(result.violations),
        "unverifiable": [{"check": u.check, "reason": u.reason} for u in result.unverifiable],
    }


# A HiddenTrajectoryResult is not one kind. The id of the payload schema names it. An id
# that is not here must not be reported as hidden_trajectory.
_RESULT_KIND = {
    "nvs.hidden-trajectory": "hidden_trajectory",
    "nvs.meaning-measurement": "meaning_measurement",
    "nvs.meaning-frame": "meaning_frame",
}


def _payload_part(schema_id: str, found: Any) -> dict[str, Any]:
    """The payload layer. A list (``nvs.token-temporal`` and the other ic02 checkers) stays
    ``ic02``. A ``HiddenTrajectoryResult`` takes its kind from the schema id, and keeps the
    status the checker returned (``PASS``, ``PASS_UNVERIFIED`` or ``FAIL``)."""
    if isinstance(found, HiddenTrajectoryResult):
        kind = _RESULT_KIND.get(schema_id)
        if kind is None:
            raise RuntimeError(
                f"no kind for HiddenTrajectoryResult of payload_schema {schema_id!r}"
            )
        return _result(kind, found)
    return {
        "kind": "ic02",
        "status": "FAIL" if found else "PASS",
        "violations": _violations(found),
        "unverifiable": [],
    }


def _parts(request: dict[str, Any]) -> dict[str, Any]:
    """``{l1: [violations], payload, window}``: the three layers, L1 first (a failed L1 stops the
    rest)."""
    envelope, lookup = request["envelope"], request.get("lookup") or {}
    stored = request.get("stored") or {}
    blobs = {uri: base64.b64decode(data) for uri, data in (request.get("blobs") or {}).items()}
    l1 = _violations(check_envelope_versioned(envelope, lookup))
    out: dict[str, Any] = {"l1": l1, "payload": "not_run", "window": None}
    if l1:
        return out
    schema = envelope["payload_schema"]
    # the parent contents of a calibration, as examples/pb_path/gate_ic02_cli.py passes them
    parent_contents = (
        {a: e["content_id"] for a, e in lookup.items()}
        if schema["id"] == "msr.calibration"
        else None
    )
    found = gate_nvs.check_nvs_payload(
        envelope, stored, blobs, parent_contents, contract_facts=request.get("contract_facts")
    )
    out["payload"] = _payload_part(schema["id"], found)
    if (schema["id"], schema["version"]) == WINDOW_SCHEMA:
        out["window"] = _result("window_space", check_window_space(envelope, stored))
    return out


def check(request: dict[str, Any]) -> dict[str, Any]:
    parts = _parts(request)
    return {
        "l1_codes": sorted({v["code"] for v in parts["l1"]}),
        "l1_messages": [v["message"] for v in parts["l1"]],
        "payload": parts["payload"],
        "window": parts["window"],
    }


def gate_op(request: dict[str, Any]) -> dict[str, Any]:
    parts = _parts(request)
    facts = request.get("contract_facts")
    return {
        **parts,
        "envelope_sha256": _sha(request["envelope"]),
        "lookup_sha256": _sha(request.get("lookup") or {}),
        "contract_facts_sha256": None if facts is None else _sha(facts),
    }


def catalog(_request: dict[str, Any]) -> dict[str, Any]:
    rows = sorted([list(row) for row in UNVERIFIABLE_CATALOG])
    as_objects = [{"payload_schema": s, "check": c, "reason": r} for s, c, r in rows]
    return {"rows": rows, "sha256": _sha(as_objects)}


def _warm_up() -> None:
    """Load, through public functions, every file of the checker that a check can read. The same
    calls every time, so the set of files is the same every time."""
    schemas.registry()  # common/, msr/ (with the archives)
    for path in gate.PAYLOAD_SCHEMAS.values():
        schemas.validator(path)
    for version in sorted(RULES):  # the envelope schema of each IC-00 version
        check_envelope_versioned({"contract": {"version": version}}, {})
    row_rule_registry()  # tools/contracts/hid_row_rules.json
    try:  # loads nvs/hidden-trajectory.schema.json; the answer is of no interest
        gate_nvs.check_nvs_payload(
            {"payload_schema": {"id": "nvs.hidden-trajectory", "version": "0.1.0"}, "payload": {}},
            {},
            {},
            None,
            contract_facts=None,
        )
    except Exception:  # noqa: BLE001, S110 -- reading the files is what counts here
        pass
    try:  # loads nvs/meaning-frame.schema.json; the answer is of no interest
        gate_nvs.check_nvs_payload(
            {"payload_schema": {"id": "nvs.meaning-frame", "version": "0.1.0"}, "payload": {}},
            {},
            {},
            None,
            contract_facts=None,
        )
    except Exception:  # noqa: BLE001, S110 -- reading the files is what counts here
        pass


def _below_root(path: str) -> str | None:
    real = os.path.realpath(path)
    if real != _ROOT and not real.startswith(_ROOT + os.sep):
        return None
    return os.path.relpath(real, _ROOT).replace(os.sep, "/")


def _loaded_py() -> list[str]:
    found = set()
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if isinstance(name, str) and name.endswith(".py"):
            rel = _below_root(name)
            if rel is not None and rel.startswith("tools/"):
                found.add(rel)
    return sorted(found)


def _opened_other() -> list[str]:
    found = set()
    for name in _OPENED:
        rel = _below_root(name)
        if rel is None or "/__pycache__/" in f"/{rel}" or rel.endswith((".py", ".pyc")):
            continue
        if os.path.isfile(name):
            found.add(rel)
    return sorted(found)


def checker_files(request: dict[str, Any]) -> dict[str, Any]:
    _warm_up()
    files: list[dict[str, str]] = []
    for rel in sorted({*_loaded_py(), *_opened_other()}):
        data = _real_open(os.path.join(_ROOT, rel), "rb").read()
        files.append({"repo": SENSOS, "path": rel, "b64": base64.b64encode(data).decode("ascii")})
    for extra in request.get("extra_files") or []:
        files.append({"repo": extra["repo"], "path": extra["path"], "b64": extra["b64"]})
    files.sort(key=lambda f: (f["repo"], f["path"]))
    bundle = jcs(
        {
            "version": BUNDLE_VERSION,
            "files": [
                {
                    "repo": f["repo"],
                    "path": f["path"],
                    "sha256": _bytes_sha(f["b64"]),
                    "bytes_b64": f["b64"],
                }
                for f in files
            ],
        }
    )
    return {
        "bundle_b64": base64.b64encode(bundle).decode("ascii"),
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
        "files": [
            {"repo": f["repo"], "path": f["path"], "sha256": _bytes_sha(f["b64"])} for f in files
        ],
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "packages": {
                "jsonschema": importlib.metadata.version("jsonschema"),
                "referencing": importlib.metadata.version("referencing"),
            },
            "platform": {"system": platform.system(), "machine": platform.machine()},
        },
    }


def _bytes_sha(b64: str) -> str:
    return hashlib.sha256(base64.b64decode(b64)).hexdigest()


def run(request: dict[str, Any]) -> dict[str, Any]:
    op = request.get("op") if isinstance(request, dict) else None
    if op == "check":
        return check(request)
    if op == "gate":
        return gate_op(request)
    if op == "catalog":
        return catalog(request)
    if op == "checker_files":
        return checker_files(request)
    if op == "jcs_sha256":
        return {"sha256": _sha(request["value"])}
    if op == "hid_space_id":
        return {"space_id": hid_space_id(request["materials"])}
    if op == "seal":
        envelope, lookup = request["envelope"], request.get("lookup") or {}
        envelope["content_id"] = ids.content_id(envelope, lookup)
        envelope["artifact_id"] = ids.artifact_id(envelope, lookup)
        return {"envelope": envelope}
    if op == "session_id":
        return {
            "id": ids.session_id_experiment(
                request["spec"],
                request["item_id"],
                request["condition"],
                request["replicate"],
                request["rng_seed"],
            )
        }
    if op == "observation_id":
        return {"id": ids.observation_id(request["session_id"], request["ordinal"])}
    raise SystemExit(f"unknown op: {op!r}")  # an op is only ever compared, never imported


def main() -> int:
    request = json.load(sys.stdin)
    try:
        answer = run(request)
    except ContractFactsError as exc:
        print(json.dumps({"error": exc.code, "message": exc.message}))
        return 2
    if request.get("trace"):
        answer["_trace"] = {"modules": _loaded_py(), "opened": _opened_other()}
    print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
