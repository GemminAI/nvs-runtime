"""``GateClient``: the formal Gate of nvs-runtime, which builds a ``GateVerdict``
(``gate-verdict/1``; design U6T-1 v0.1 3-7, DC-T1-1..4).

The rules are sensos's (by design, no cross-imports): nvs-runtime never
imports them. ``GateClient`` starts
``_gate_child.py`` in ``GATE_PY`` (an interpreter with ``jsonschema``) for each call and only
JSON crosses. What nvs-runtime adds is what the sensos checker does not say:

* the layers: L1 (the IC-00 envelope rule), then L2 (the payload), then L3 (the window space, only
  for a ``msr.trajectory-window`` payload). A failed L1 leaves L2 and L3 ``not_run``;
* the ``result``: ``FAIL`` if any layer has a violation, else ``PASS_UNVERIFIED`` if anything is
  unverifiable, else ``PASS``. ``WAIVED`` is not made (``waived`` is always empty);
* the check of every ``unverifiable`` row against sensos's catalogue (DC-7, DC-T1-4);
* the lookup of the parents (DC-T1-3): an envelope with parents is not sent to the Gate without
  one;
* the checker bundle (``gate-checker-bundle/1``) and its hash, the environment of the child, and
  ``verdict_sha256``.

Not in a verdict: eligibility or admission to a chain (a verdict does not say it, by design)
and any time (DC-T1-2): the
same inputs give the same bytes. The bundle's bytes are given by ``checker_bundle``; keeping them
is U6T-2's.

This module imports the standard library and ``nvs_runtime.__version__`` only.
``examples/hidden_trajectory/sensos_door.py`` is a thin layer over ``GateProcess`` and
``GateClient``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from nvs_runtime.__version__ import __version__

#: The script that runs in ``GATE_PY``. Located, never imported (by design, no cross-imports).
CHILD_SCRIPT: Final = Path(__file__).with_name("_gate_child.py")
VERDICT_VERSION: Final = "gate-verdict/1"
BUNDLE_VERSION: Final = "gate-checker-bundle/1"
GATE_INTERNAL_ERROR: Final = "gate_internal_error"
PRODUCER_NAME: Final = "nvs-runtime"
REPO_NVS_RUNTIME: Final = "nvs-runtime"

LAYER_L1: Final = "L1"
LAYER_L2: Final = "L2"
LAYER_L3: Final = "L3"
NOT_RUN: Final = "not_run"
NOT_APPLICABLE: Final = "not_applicable"

#: A ``GateVerdict`` is a plain dictionary of the form ``gate-verdict/1``.
GateVerdict = dict[str, Any]

# The files of nvs-runtime that enter the checker bundle: their package paths.
_OWN_FILES: Final = (
    "nvs_runtime/hidden_trajectory/_gate_child.py",
    "nvs_runtime/hidden_trajectory/gate_client.py",
)


class GateError(Exception):
    """Base of what the Gate raises instead of making a verdict."""


class GateUnavailable(GateError):
    """The child process failed or its output could not be read: no verdict (T-T1-13). Not a
    judgment on the envelope."""


class GateInternalError(GateUnavailable):
    """The Gate could not judge because the caller's input was wrong (a ``contract_facts`` of the
    wrong shape): not a violation of the envelope."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class CatalogMismatch(GateError):
    """The checker returned an ``unverifiable`` row that sensos's catalogue does not have: the
    checker and the catalogue are of different versions, and no verdict is made (DC-T1-4)."""


class LookupMissing(GateError):
    """The envelope has parents (or a calibration) that the lookup does not name: raised before
    the Gate is called (DC-T1-3), so that a mistake of the caller is not recorded as a failure of
    the envelope."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        super().__init__(f"the lookup lacks: {', '.join(missing)}")
        self.missing = missing


class ParentEnvelopeMissing(GateError):
    """The envelope has a parent, the caller gave no parent envelope, and did not say the parent is
    absent (``ParentAbsent``): raised before the Gate is called, as ``LookupMissing`` is (DC-T1-3).
    "Not fetched" is the caller's mistake, not a finding about the envelope."""

    def __init__(self, artifact_id: str) -> None:
        super().__init__(f"the parent envelope of {artifact_id} was not given")
        self.artifact_id = artifact_id


class ParentMismatch(GateError):
    """The parent envelope the caller gave is not the one ``ParentRef`` names: the data of the
    caller disagree. ``fields`` are the names that differ (``artifact_id``, ``content_id``,
    ``seq``, ``observation_id``); ``differences`` has ``(expected, found)`` for each."""

    def __init__(self, differences: Mapping[str, tuple[Any, Any]]) -> None:
        shown = "; ".join(f"{k}: expected {e!r}, found {f!r}" for k, (e, f) in differences.items())
        super().__init__(f"the parent envelope does not match the parent: {shown}")
        self.differences = dict(differences)
        self.fields = tuple(differences)


@dataclass(frozen=True, slots=True)
class ParentAbsent:
    """The caller looked the parent up in ``source`` (the store it asked) and it is not there.
    This is a fact the caller states; ``None`` instead means "not fetched" and is never read as
    absence."""

    artifact_id: str
    source: str


def required_lookup_ids(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    """The ``artifact_id`` of every parent and of the calibration of ``envelope`` (the ones the
    IC-00 rule reads from the lookup), once each, in the order of the code points."""
    found: set[str] = set()
    parents = envelope.get("parents")
    if isinstance(parents, list):
        for role in parents:
            ids = role.get("artifact_ids") if isinstance(role, Mapping) else None
            if isinstance(ids, list):
                found.update(i for i in ids if isinstance(i, str))
    calibration = envelope.get("calibration_ref")
    if isinstance(calibration, Mapping) and isinstance(calibration.get("artifact_id"), str):
        found.add(calibration["artifact_id"])
    return tuple(sorted(found))


def missing_lookup(
    envelope: Mapping[str, Any], lookup: Mapping[str, Any] | None
) -> tuple[str, ...]:
    """The ids of ``required_lookup_ids`` that ``lookup`` does not have."""
    have = lookup or {}
    return tuple(i for i in required_lookup_ids(envelope) if i not in have)


@dataclass(frozen=True)
class GateProcess:
    """One call of ``_gate_child.py`` in ``gate_py``, for ``sensos_tools`` (``SENSOS_TOOLS``)."""

    gate_py: Path
    sensos_tools: Path

    @classmethod
    def from_environment(cls) -> GateProcess:
        missing = [name for name in ("GATE_PY", "SENSOS_TOOLS") if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"environment variables not set: {', '.join(missing)}")
        return cls(Path(os.environ["GATE_PY"]), Path(os.environ["SENSOS_TOOLS"]))

    def call(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            done = subprocess.run(
                [str(self.gate_py), str(CHILD_SCRIPT)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                env={"PYTHONDONTWRITEBYTECODE": "1", "SENSOS_TOOLS": str(self.sensos_tools)},
            )
        except OSError as error:
            raise GateUnavailable(f"the child process could not be started: {error}") from error
        if done.returncode == 2:
            try:
                answer = json.loads(done.stdout)
                code, message = str(answer["error"]), str(answer["message"])
            except (ValueError, KeyError, TypeError) as error:
                raise GateUnavailable("the child answered 2 with no readable error") from error
            raise GateInternalError(code, message)
        if done.returncode:
            last = done.stderr.strip().splitlines()[-1:] or ["no output"]
            raise GateUnavailable(f"the child exited {done.returncode}: {last[0]}")
        try:
            answer = json.loads(done.stdout)
        except ValueError as error:
            raise GateUnavailable("the child's output is not JSON") from error
        if not isinstance(answer, dict):
            raise GateUnavailable("the child's output is not a JSON object")
        return answer


def git_head(root: Path) -> str | None:
    """The commit of the repository whose top is ``root``, or ``None`` (not a checkout, no git)."""
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if top.returncode or Path(top.stdout.strip()).resolve() != root.resolve():
            return None
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = head.stdout.strip()
    ok = head.returncode == 0 and len(value) == 40 and all(c in "0123456789abcdef" for c in value)
    return value if ok else None


def _matches_catalog(
    rows: set[tuple[str, str, str]], schema_id: str, check: str, reason: str
) -> bool:
    """A row of the catalogue names ``(payload_schema, check, reason)``; a ``check`` that ends in
    ``.<field>`` stands for any field name after its prefix (``contract_binding.<field>``)."""
    if (schema_id, check, reason) in rows:
        return True
    for row_schema, row_check, row_reason in rows:
        if row_schema != schema_id or row_reason != reason or not row_check.endswith(".<field>"):
            continue
        prefix = row_check[: -len("<field>")]
        if check.startswith(prefix) and len(check) > len(prefix):
            return True
    return False


class GateClient:
    """``verdict(envelope, ...)`` -> ``GateVerdict``. One client builds its checker bundle once."""

    def __init__(self, gate_py: Path, sensos_tools: Path) -> None:
        self.process = GateProcess(Path(gate_py), Path(sensos_tools))
        self._prepared: dict[str, Any] | None = None

    @classmethod
    def from_environment(cls) -> GateClient:
        process = GateProcess.from_environment()
        return cls(process.gate_py, process.sensos_tools)

    # ---- the bundle and the catalogue: asked once --------------------------------------------

    def _prepare(self) -> dict[str, Any]:
        if self._prepared is None:
            package_root = CHILD_SCRIPT.parents[2]
            extra = [
                {
                    "repo": REPO_NVS_RUNTIME,
                    "path": relative,
                    "b64": base64.b64encode((package_root / relative).read_bytes()).decode("ascii"),
                }
                for relative in _OWN_FILES
            ]
            bundle = self.process.call({"op": "checker_files", "extra_files": extra})
            catalog = self.process.call({"op": "catalog"})
            sensos_commit = git_head(self.process.sensos_tools.resolve().parent)
            runtime_commit = git_head(CHILD_SCRIPT.parents[3])
            self._prepared = {
                "bundle": base64.b64decode(bundle["bundle_b64"]),
                "bundle_sha256": bundle["bundle_sha256"],
                "files": bundle["files"],
                "runtime": bundle["runtime"],
                "rows": {tuple(row) for row in catalog["rows"]},
                "catalog_sha256": catalog["sha256"],
                "sensos_commit": sensos_commit,
                "nvs_runtime_commit": runtime_commit,
            }
        return self._prepared

    def checker_bundle(self) -> bytes:
        """The bytes of ``gate-checker-bundle/1`` (the JCS of ``{version, files}``), the same for
        the life of this client. Its SHA-256 is ``bundle_sha256`` of every verdict."""
        bundle = self._prepare()["bundle"]
        assert isinstance(bundle, bytes)
        return bundle

    def bundle_sha256(self) -> str:
        return str(self._prepare()["bundle_sha256"])

    # ---- the verdict -------------------------------------------------------------------------

    def verdict(
        self,
        envelope: Mapping[str, Any],
        *,
        lookup: Mapping[str, Any] | None = None,
        stored: Mapping[str, Any] | None = None,
        blobs: Mapping[str, bytes] | None = None,
        contract_facts: Mapping[str, Any] | None = None,
    ) -> GateVerdict:
        """The ``GateVerdict`` of ``envelope``.

        ``LookupMissing`` before anything is run if a parent is not in ``lookup``;
        ``GateUnavailable`` if the child fails or the envelope cannot be identified;
        ``CatalogMismatch`` for an ``unverifiable`` row the catalogue does not have. Nothing is
        a verdict then.
        """
        missing = missing_lookup(envelope, lookup)
        if missing:
            raise LookupMissing(missing)
        subject, rules = self._identify(envelope)
        prepared = self._prepare()
        lookup_given = {k: dict(v) for k, v in (lookup or {}).items()}
        blobs_given = dict(blobs or {})
        answer = self.process.call(
            {
                "op": "gate",
                "envelope": dict(envelope),
                "lookup": lookup_given,
                "stored": dict(stored or {}),
                "blobs": {u: base64.b64encode(d).decode("ascii") for u, d in blobs_given.items()},
                "contract_facts": None if contract_facts is None else dict(contract_facts),
            }
        )
        layers, violations, unverifiable = _layers(answer)
        schema_id = rules["payload_schema"]["id"]
        for row in unverifiable:
            if not _matches_catalog(prepared["rows"], schema_id, row["check"], row["reason"]):
                raise CatalogMismatch(
                    f"({schema_id}, {row['check']}, {row['reason']}) is not in the UNVERIFIABLE "
                    "catalogue of sensos"
                )
        subject["envelope_sha256"] = answer["envelope_sha256"]
        commit = prepared["sensos_commit"]
        runtime_commit = prepared["nvs_runtime_commit"]
        body: dict[str, Any] = {
            "verdict_version": VERDICT_VERSION,
            "subject": subject,
            "result": _result(violations, unverifiable),
            "layers": layers,
            "codes": sorted({v["code"] for v in violations}),
            "violations": violations,
            "unverifiable": unverifiable,
            "waived": [],
            "rules": rules,
            "inputs": {
                "lookup": lookup_given,
                "lookup_sha256": answer["lookup_sha256"],
                "stored": sorted(stored or {}),
                "blobs_checked": sorted(
                    {hashlib.sha256(d).hexdigest() for d in blobs_given.values()}
                ),
                "contract_facts_sha256": answer["contract_facts_sha256"],
            },
            "gate": {
                "checker": {
                    "bundle_sha256": prepared["bundle_sha256"],
                    "sensos_commit": commit,
                    "nvs_runtime_commit": runtime_commit,
                    "files": prepared["files"],
                },
                "runtime": prepared["runtime"],
                "unverifiable_catalog": {
                    "version": commit or "unknown",
                    "sha256": prepared["catalog_sha256"],
                },
            },
            "produced_by": {
                "name": PRODUCER_NAME,
                "version": __version__,
                "build": runtime_commit or f"bundle:{prepared['bundle_sha256'][:16]}",
            },
        }
        body["verdict_sha256"] = self.process.call({"op": "jcs_sha256", "value": body})["sha256"]
        return body

    @staticmethod
    def _identify(envelope: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(subject, rules)`` from the envelope, or ``GateUnavailable`` if it cannot say what it
        is (then no document of the form ``gate-verdict/1`` can be made about it)."""
        try:
            contract_version = envelope["contract"]["version"]
            schema = envelope["payload_schema"]
            subject = {
                "artifact_id": envelope["artifact_id"],
                "content_id": envelope["content_id"],
            }
            rules = {
                "ic00_contract_version": contract_version,
                "payload_schema": {"id": schema["id"], "version": schema["version"]},
            }
        except (KeyError, TypeError) as error:
            raise GateUnavailable(
                f"the envelope cannot be identified (missing {error}); no verdict is made"
            ) from error
        texts = (
            contract_version,
            subject["artifact_id"],
            subject["content_id"],
            rules["payload_schema"]["id"],
            rules["payload_schema"]["version"],
        )
        if not all(isinstance(t, str) and t for t in texts):
            raise GateUnavailable("the envelope cannot be identified (not text); no verdict")
        return subject, rules


def _layers(
    answer: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    """``(layers, violations, unverifiable)`` from the child's ``gate`` answer."""
    violations: list[dict[str, str]] = []
    unverifiable: list[dict[str, str]] = []

    def take(layer: str, found: list[Mapping[str, str]], unknown: list[Mapping[str, str]]) -> None:
        violations.extend(
            {"layer": layer, "code": v["code"], "message": v["message"]} for v in found
        )
        unverifiable.extend(
            {"layer": layer, "check": u["check"], "reason": u["reason"]} for u in unknown
        )

    l1 = list(answer["l1"])
    take(LAYER_L1, l1, [])
    if l1:
        layers = {
            LAYER_L1: {"status": "FAIL", "codes": sorted({v["code"] for v in l1})},
            LAYER_L2: {"status": NOT_RUN, "kind": None, "codes": []},
            LAYER_L3: {"status": NOT_RUN, "codes": []},
        }
        return layers, violations, unverifiable
    payload = answer["payload"]
    take(LAYER_L2, payload["violations"], payload["unverifiable"])
    window = answer.get("window")
    if window is None:
        l3: dict[str, Any] = {"status": NOT_APPLICABLE, "codes": []}
    else:
        take(LAYER_L3, window["violations"], window["unverifiable"])
        l3 = {
            "status": window["status"],
            "codes": sorted({v["code"] for v in window["violations"]}),
        }
    layers = {
        LAYER_L1: {"status": "PASS", "codes": []},
        LAYER_L2: {
            "status": payload["status"],
            "kind": payload["kind"],
            "codes": sorted({v["code"] for v in payload["violations"]}),
        },
        LAYER_L3: l3,
    }
    return layers, violations, unverifiable


def _result(violations: list[dict[str, str]], unverifiable: list[dict[str, str]]) -> str:
    if violations:
        return "FAIL"
    return "PASS_UNVERIFIED" if unverifiable else "PASS"
