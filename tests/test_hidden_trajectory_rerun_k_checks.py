"""``rerun_from_store(k_checks=...)`` hands the K part on to the checks it is given.

The K route itself (what the checks do with a stored K) is tested in the Developer Tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nvs_runtime.hidden_trajectory import rerun

BOARD = json.dumps([{"seq": 0, "hmac": "a"}, {"seq": 1, "hmac": "b"}]).encode()


CAUSAL = b"[]"


class RecordingChecks:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def check_chain(self, wires: Any, *, session_id: Any, head: Any) -> dict[str, Any]:
        seen = {"n": len(wires), "session_id": session_id, "head": head}
        self.calls.append(("check_chain", seen))
        return {"ok": True, "records": len(wires), "problems": []}

    def rerun(self, **kwargs: Any) -> Any:
        self.calls.append(("rerun", kwargs))
        return SimpleNamespace(k_path={"status": "measured"}, board=BOARD, causal=CAUSAL)


def test_rerun_from_store_hands_k_checks_to_every_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records = [{"observation_id": "o1", "contract": {"contract_sha256": "c1"}}]
    monkeypatch.setattr(
        rerun, "CaptureArchive", lambda data_dir: SimpleNamespace(records=lambda: records)
    )
    got: list[Any] = []

    def check_capture(archive: Any, record: Any, *, rerun_k: bool, k_checks: Any) -> dict[str, Any]:
        got.append((rerun_k, k_checks))
        return {"observation_id": record["observation_id"], "ok": True}

    monkeypatch.setattr(rerun, "_check_capture", check_capture)
    checks = RecordingChecks()
    report = rerun.rerun_from_store(tmp_path, rerun_k=True, k_checks=checks)
    assert report["ok"] is True and got == [(True, checks)]
    sentinel = object()
    monkeypatch.setattr(rerun, "default_k_checks", lambda: sentinel)
    rerun.rerun_from_store(tmp_path)
    assert got[1] == (False, sentinel)  # the default is what the entry point offers
