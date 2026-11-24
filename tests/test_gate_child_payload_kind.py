"""The payload kind chosen by the Gate child.

The child is a script (by design, no cross-imports). This file never imports it. A ``GATE_PY``
process loads it, with
``SENSOS_TOOLS`` pointing at the sensos checkout, and reports what ``_payload_part`` does with
each return of ``check_nvs_payload``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

CHILD = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "nvs_runtime"
    / "hidden_trajectory"
    / "_gate_child.py"
)

_MISSING = [name for name in ("SENSOS_TOOLS", "GATE_PY") if not os.environ.get(name)]
pytestmark = pytest.mark.skipif(
    bool(_MISSING) or not Path(os.environ.get("GATE_PY", "")).is_file(),
    reason=(
        f"environment variables not set: {', '.join(_MISSING)}"
        if _MISSING
        else f"GATE_PY is not a file: {os.environ.get('GATE_PY')}"
    ),
)

_PROBE = r"""
import importlib.util
import json
import sys

path = sys.argv[1]
spec = importlib.util.spec_from_file_location("gate_child_kind_probe", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from contracts.check import Violation
from contracts.hidden_trajectory import Unverifiable

Result = mod.HiddenTrajectoryResult

def part(schema_id, found):
    return mod._payload_part(schema_id, found)

out = {}
out["measurement"] = part(
    "nvs.meaning-measurement",
    Result(unverifiable=(Unverifiable("frame_binding", "frame_not_supplied"),)),
)
out["frame"] = part("nvs.meaning-frame", Result())
out["frame_fail"] = part(
    "nvs.meaning-frame",
    Result(violations=(Violation("meaning_frame_id_mismatch", "id"),)),
)
out["hidden"] = part("nvs.hidden-trajectory", Result())
out["token"] = part("nvs.token-temporal", [])
out["token_fail"] = part("nvs.token-temporal", [Violation("schema", "no")])
try:
    part("nvs.not-a-payload", Result())
except RuntimeError as exc:
    out["unknown"] = {"raised": True, "message": str(exc)}
else:
    out["unknown"] = {"raised": False, "message": ""}
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def kinds() -> dict[str, Any]:
    proc = subprocess.run(
        [os.environ["GATE_PY"], "-c", _PROBE, str(CHILD)],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert proc.returncode == 0, proc.stderr
    loaded = json.loads(proc.stdout)
    assert isinstance(loaded, dict)
    return loaded


def test_a_measurement_result_is_meaning_measurement(kinds: dict[str, Any]) -> None:
    assert kinds["measurement"] == {
        "kind": "meaning_measurement",
        "status": "PASS_UNVERIFIED",
        "violations": [],
        "unverifiable": [{"check": "frame_binding", "reason": "frame_not_supplied"}],
    }


def test_a_frame_result_is_meaning_frame(kinds: dict[str, Any]) -> None:
    assert kinds["frame"] == {
        "kind": "meaning_frame",
        "status": "PASS",
        "violations": [],
        "unverifiable": [],
    }
    assert kinds["frame_fail"] == {
        "kind": "meaning_frame",
        "status": "FAIL",
        "violations": [{"code": "meaning_frame_id_mismatch", "message": "id"}],
        "unverifiable": [],
    }


def test_a_token_temporal_list_stays_ic02(kinds: dict[str, Any]) -> None:
    assert kinds["token"] == {
        "kind": "ic02",
        "status": "PASS",
        "violations": [],
        "unverifiable": [],
    }
    assert kinds["token_fail"] == {
        "kind": "ic02",
        "status": "FAIL",
        "violations": [{"code": "schema", "message": "no"}],
        "unverifiable": [],
    }


def test_a_hidden_trajectory_result_keeps_its_kind(kinds: dict[str, Any]) -> None:
    assert kinds["hidden"] == {
        "kind": "hidden_trajectory",
        "status": "PASS",
        "violations": [],
        "unverifiable": [],
    }


def test_an_unknown_schema_that_returns_the_result_raises(kinds: dict[str, Any]) -> None:
    assert kinds["unknown"] == {
        "raised": True,
        "message": "no kind for HiddenTrajectoryResult of payload_schema 'nvs.not-a-payload'",
    }
