"""The 12 main series of a Phase 4 acquisition, read from its capture archive (never written)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.phase4 import summary as sm
from nvs_runtime.hidden_trajectory.replay import risk_control as rcx


def prompt_groups(prompt_set: dict[str, Any]) -> dict[str, str]:
    """prompt_id -> group, in the order of the file."""
    return {e["prompt_id"]: g for g, entries in prompt_set["groups"].items() for e in entries}


def load_series(store_dir: str | Path, groups: dict[str, str]) -> list[sm.Series]:
    """One ``Series`` per prompt: the stored ``cycle_raw`` of each generated token, the anchor
    counts of its board record, and the finish reason of the capture contract."""
    archive = CaptureArchive(store_dir)
    by_session = {str(r.get("contract_session_id")): r for r in archive.records()}
    out: list[sm.Series] = []
    for prompt_id, group in groups.items():
        record = by_session[f"p4-{prompt_id}"]
        k = record["positions"]["final_norm"]["k_path"]
        cycles, _ = rcx.read_cycles(store_dir, str(record["observation_id"]))
        wires = sorted(
            json.loads(archive.read_k(k["board_records_sha256"]) or b"[]"),
            key=lambda w: int(w["token_index"]),
        )
        contract = archive.read_contract(record["contract"]["contract_sha256"]) or {}
        binding = contract.get("binding", {})
        out.append(
            sm.Series(
                prompt_id,
                group,
                cycles,
                [dict(w["anchor_counts"]) for w in wires],
                str(binding.get("finish_reason")),
                None,
                len(cycles),
            )
        )
    return out
