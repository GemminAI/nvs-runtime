"""Check what a capture archive holds, from the archive alone (U6-6a; design U6-6 v0.1, 6 and 14;
AC-66-8, AC-66-3).

No Formal Store, no model, no MLX and no key are used: the contract, the RAW files and the id
lists are read from ``<data_dir>/capture/`` and

* the contract is checked again (``check_contract_v2``: structure, G1, G2 with ``raw-nd-v1``
  recomputed from the RAW files) and its hash recomputed;
* the id lists hash to the contract's ``prompt_ids_sha256`` / ``gen_token_ids_sha256``;
* for each position with a K path, the K checks (``k_checks``, or those of the installed
  ``nvs_runtime.k_measurer`` entry point) check the stored board records without a key: structure,
  CRC-8, ``record_hash`` and the chain, the session id and the chain head the record names. The HMAC
  is NOT checked: it cannot be, and the record says only that it was checked when the run was made
  (``hmac: verified_at_run_time``). With no checks installed the K part is reported as not checked
  (``edge_unavailable``) and counted in ``k_unchecked``;
* with ``rerun_k``, the same checks run route K again from the stored RAW with a new random key and
  compare the parts that do not depend on the key (everything of a board record except ``hmac``)
  with the stored ones.

``rerun_from_store(data_dir)`` is the function; ``python -m nvs_runtime.hidden_trajectory.rerun
<data_dir>`` is the command (JSON on stdout, exit 0 only if every check passed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from nvs_observation import capture_edge
from nvs_observation.contract_check import (
    EDGE_NOT_INSTALLED,
    ContractUnavailable,
    check_contract_v2,
    recompute_contract_sha256,
)

from nvs_runtime.hidden_trajectory.capture_archive import CaptureArchive
from nvs_runtime.hidden_trajectory.k_measure import KOutcome, default_k_checks
from nvs_runtime.hidden_trajectory.record import REASON_K_EDGE_UNAVAILABLE


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class KChecks(Protocol):
    """The checks of route K that need ``nvs_edge``'s own record format and Edge.

    ``rerun_from_store`` hands the whole K part of a capture to ``check_k_route``; ``check_chain``
    (the stored board records, no key) and ``rerun`` (route K run again) are what an
    implementation builds it from. ``rerun_from_store`` takes them as ``k_checks``, or finds them
    through ``k_measure``'s entry point; with neither the K part is reported as not checked."""

    def check_k_route(
        self,
        archive: CaptureArchive,
        record: Mapping[str, Any],
        *,
        raw: Mapping[str, bytes],
        trajectory: Mapping[str, str],
        contract: Mapping[str, Any],
        rerun_k: bool,
    ) -> dict[str, Any]:
        """The K part of one capture's report: point -> what was checked."""
        ...

    def check_chain(
        self, wires: Sequence[Mapping[str, Any]], *, session_id: str | None, head: str | None
    ) -> dict[str, Any]:
        """Structure, CRC-8, ``record_hash`` and the chain of board records, with no key."""
        ...

    def rerun(
        self,
        *,
        observation_id: str,
        point: str,
        raw_bytes: bytes,
        trajectory_id: str | None,
        model_build_id: str | None,
        compute_environment: str | None,
    ) -> KOutcome:
        """Route K run again for one position (with the default key source)."""
        ...


def _k_route_not_checked(record: Mapping[str, Any]) -> dict[str, Any]:
    """The K part of a report when nothing can check it (no ``k_checks``, no entry point): every
    position that has a measured K is listed as not checked, with the reason a record uses when
    ``nvs_edge`` is not there."""
    report: dict[str, Any] = {}
    for point, entry in record["positions"].items():
        k = entry.get("k_path")
        if k is not None and k["status"] == "measured":
            report[point] = {"checked": False, "reason": REASON_K_EDGE_UNAVAILABLE}
    return report


def _k_part(
    k_checks: KChecks | None,
    archive: CaptureArchive,
    record: Mapping[str, Any],
    *,
    raw: Mapping[str, bytes],
    trajectory: Mapping[str, str],
    contract: Mapping[str, Any],
    rerun_k: bool,
) -> dict[str, Any]:
    """The K part of one capture's report: ``k_checks`` does all of it; with none it is reported as
    not checked."""
    if k_checks is None:
        return _k_route_not_checked(record)
    return k_checks.check_k_route(
        archive,
        record,
        raw=raw,
        trajectory=trajectory,
        contract=contract,
        rerun_k=rerun_k,
    )


def _k_unchecked(record: Mapping[str, Any], capture: Mapping[str, Any]) -> int:
    """How many positions of ``record`` have a measured K that ``capture`` (its report) does not
    show as checked: reported as ``checked: False``, or not reported at all (the capture could not
    be checked). A position whose K was not measured is not counted."""
    reported = capture.get("k") or {}
    count = 0
    for point, entry in (record.get("positions") or {}).items():
        k = entry.get("k_path")
        if k is None or k["status"] != "measured":
            continue
        done = reported.get(point)
        if not isinstance(done, Mapping) or done.get("checked") is False:
            count += 1
    return count


def _check_capture(
    archive: CaptureArchive,
    record: Mapping[str, Any],
    *,
    rerun_k: bool,
    k_checks: KChecks | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"observation_id": record["observation_id"], "ok": False}
    sha = record["contract"]["contract_sha256"]
    contract = archive.read_contract(sha) if sha else None
    if contract is None:
        out["problem"] = "the contract is not in the archive"
        return out
    out["contract_sha256_recomputed_matches"] = recompute_contract_sha256(contract) == sha
    # the id lists
    ids: dict[str, Any] = {}
    for name, key in (("input", "prompt_ids_sha256"), ("generated", "gen_token_ids_sha256")):
        wanted = contract["binding"][key]
        data = archive.read_ids(wanted)
        ids[name] = {"sha256": wanted, "present": data is not None}
        if data is not None:
            count = len(data) // 8
            ids[name]["hash_matches"] = _sha(data) == wanted
            ids[name]["count"] = count
            struct.unpack(f"<{count}q", data)  # readable as int64
    out["ids"] = ids
    # RAW and the contract, checked again
    trajectory = {p["point"]: p["trajectory_id"] for p in contract["positions"]}
    raw: dict[str, bytes] = {}
    for point, tid in trajectory.items():
        data = archive.raw_for_trajectory(tid)
        if data is not None:
            raw[point] = data
    edge, failure = capture_edge.current()
    if edge is None:
        raise ContractUnavailable(failure.detail if failure is not None else EDGE_NOT_INSTALLED)
    instruments: dict[str, dict[str, Any]] = {}
    for point, data in raw.items():
        states, header = edge.decode_raw(data, dimension=None)
        instruments[point] = edge.instrument_record(states, header.trajectory_id)
    check = check_contract_v2(contract, raw_bytes=raw, instrument_records=instruments)
    report = check.report or {}
    g1, g2 = report.get("G1", {}), report.get("G2", {})
    out["contract_check"] = {
        "status": check.status,
        "contract_sha256": check.contract_sha256,
        "G1": g1.get("result"),
        "G1_failures": g1.get("failures", []),
        "G2": g2.get("result"),
        "raw_nd_output_sha256": {
            p: (instruments[p].get("output_sha256")) for p in sorted(instruments)
        },
    }
    out["raw_present"] = {p: p in raw for p in trajectory}
    # route K, without the key
    k_report = _k_part(
        k_checks,
        archive,
        record,
        raw=raw,
        trajectory=trajectory,
        contract=contract,
        rerun_k=rerun_k,
    )
    out["k"] = k_report
    ok = (
        out["contract_sha256_recomputed_matches"]
        and all(i.get("present") and i.get("hash_matches") for i in ids.values())
        and check.status == "VERIFIED"
        and check.contract_sha256 == sha
        and not g1.get("failures")
        and g2.get("result") == "PASS"
        and all(out["raw_present"].values())
        and all(
            i.get("checked") is False
            or (
                i["board_present"]
                and i["causal_present"]
                and i["chain_and_crc"]["ok"]
                and i["n_records_matches"]
                and ("rerun_k" not in i or i["rerun_k"]["same_as_stored"])
            )
            for i in k_report.values()
        )
    )
    out["ok"] = bool(ok)
    return out


def rerun_from_store(
    data_dir: str | Path,
    *,
    observation_id: str | None = None,
    rerun_k: bool = False,
    k_checks: KChecks | None = None,
) -> dict[str, Any]:
    """Check every capture (or one observation) of the archive in ``data_dir`` from the archive
    alone. ``{"ok": bool, "captures": [...], "k_unchecked": int}``; ``ok`` only if every capture
    passed. ``k_checks`` does the K part (see ``KChecks``); without it, the checks of the installed
    ``nvs_runtime.k_measurer`` entry point are used, and with none installed the K part is
    reported as ``{"checked": False, "reason": "edge_unavailable"}`` per position.
    ``k_unchecked`` is the number of positions with a measured K that were not checked (it does not
    change ``ok``)."""
    if k_checks is None:
        k_checks = default_k_checks()
    archive = CaptureArchive(data_dir)
    latest: dict[str, dict[str, Any]] = {}
    for record in archive.records():
        if observation_id is None or record.get("observation_id") == observation_id:
            latest[record["observation_id"]] = record
    captures: list[dict[str, Any]] = []
    for record in latest.values():
        if record["contract"]["contract_sha256"] is None:
            captures.append(
                {"observation_id": record["observation_id"], "ok": True, "skipped": "no contract"}
            )
            continue
        try:
            captures.append(_check_capture(archive, record, rerun_k=rerun_k, k_checks=k_checks))
        except Exception as error:  # noqa: BLE001 -- a file that is not what its name says is a finding
            captures.append(
                {
                    "observation_id": record["observation_id"],
                    "ok": False,
                    "problem": f"{type(error).__name__}: {error}"[:300],
                }
            )
    unchecked = sum(
        _k_unchecked(record, capture)
        for record, capture in zip(latest.values(), captures, strict=True)
    )
    return {
        "ok": bool(captures) and all(c["ok"] for c in captures),
        "captures": captures,
        "k_unchecked": unchecked,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m nvs_runtime.hidden_trajectory.rerun",
        description="Check a capture archive from the archive alone (no key is used).",
    )
    parser.add_argument("data_dir")
    parser.add_argument("--observation-id", default=None)
    parser.add_argument(
        "--rerun-k", action="store_true", help="run route K again from the stored RAW"
    )
    args = parser.parse_args(argv)
    report = rerun_from_store(
        args.data_dir, observation_id=args.observation_id, rerun_k=args.rerun_k
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
