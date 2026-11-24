#!/usr/bin/env python3
"""Export the files of a public profile of nvs-runtime into a new folder (no git history).

* The profile ``core`` is ``allowlist.txt``; ``measured-k`` adds ``allowlist-measured-k.txt`` (the
  stored measured-K data and the tests that read it).
* The files are read from the commit ``HEAD`` (not from the working tree), the working tree must be
  clean, and the target folder must be new (or empty) and outside the repository. Nothing is
  committed and no git data is copied.
* ``public-sync/sync-manifest.json`` is written into the export: the profile, the sha256 of every
  exported file (the manifest cannot hold its own), and the sha256 of the audit rules and of the
  audit tool that are exported with it. With ``--reference-rules`` it also says whether the
  exported rules file is byte-identical to that one. It holds no commit id: nothing in the export
  names the commit it was made from, and the export fails (and writes nothing) if any file or the
  manifest contains that commit's id, in full or as its first 7 characters.
* ``--internal-record PATH`` writes, outside the export and outside the repository, a JSON record
  of the export: the source commit, the branch, the profile, the number of exported files, the
  sha256 of the public manifest and the time (UTC). PATH inside either folder is refused.
* A file of the profile that refers to a tracked file the profile does not publish (an import of a
  test module, a path of a data folder) is reported, and the export is not made.
* ``--audit`` applies the audit rules (``--rules``, repeatable; the exported public rules by
  default) to the exported files and prints the number of hits by layer and rule. Layer-3 hits are
  named by file and line only.

Standard library only. Exit status 0 = done, 1 = a problem (on stderr), 2 = bad rules.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_artifacts  # noqa: E402
import check_allowlist  # noqa: E402

PROFILES = ("core", "measured-k")
MANIFEST_REL = "public-sync/sync-manifest.json"
RULES_REL = "public-sync/audit-rules.toml"
TOOL_REL = "public-sync/audit_artifacts.py"


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


def selection(allowlist: Path, extra: Path | None, profile: str) -> list[str]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; one of {PROFILES}")
    core = check_allowlist.read_list(allowlist)
    more = (
        check_allowlist.read_list(extra) if (extra is not None and profile == "measured-k") else []
    )
    return sorted([*core, *more])


def find_references(root: Path, chosen: list[str], tracked: list[str]) -> list[str]:
    """What a chosen Python file of ``src``, ``tests`` or ``examples`` refers to that the profile
    does not publish.

    A reference is an import of a module whose file is a tracked sibling that is not chosen, or a
    string that names a data folder (the highest folder whose tracked files are all unpublished)
    or a loose data file that is not chosen."""
    chosen_set = set(chosen)
    unpublished = sorted(set(tracked) - chosen_set)
    unpublished_set = set(unpublished)
    stems: dict[str, str] = {}
    names: dict[str, str] = {}
    for rel in unpublished:
        path = Path(rel)
        if path.suffix == ".py":
            stems.setdefault(path.stem, rel)
            continue
        top = None
        for parent in reversed(path.parents):
            if len(parent.parts) < 2 or parent.parts[0] not in ("tests", "examples"):
                continue
            below = [f for f in tracked if f.startswith(parent.as_posix() + "/")]
            if below and all(f in unpublished_set for f in below):
                top = parent
                break
        names.setdefault(top.name if top is not None else path.name, rel)
    found: list[str] = []
    for rel in chosen:
        if not rel.endswith(".py") or rel.split("/")[0] not in ("src", "tests", "examples"):
            continue
        tree = ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".")[0]]
            for module in modules:
                target = stems.get(module)
                if target is not None and Path(target).parent == Path(rel).parent:
                    found.append(f"{rel}: imports {module}, which is not published ({target})")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for name, target in names.items():
                    if len(name) > 3 and name in node.value:
                        found.append(f"{rel}: names {name!r}, which is not published ({target})")
    return sorted(set(found))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _names_commit(data: bytes, commit: str) -> bool:
    """Whether ``data`` contains ``commit`` in full or as its first 7 characters."""
    text = data.lower()
    return commit.lower().encode() in text or commit[:7].lower().encode() in text


def export(
    root: Path,
    out: Path,
    profile: str,
    allowlist: Path,
    extra: Path | None,
    reference_rules: Path | None = None,
    internal_record: Path | None = None,
) -> dict[str, object]:
    root = root.resolve()
    out = out.resolve()
    if out == root or root in out.parents:
        raise ValueError("the target folder must be outside the repository")
    if internal_record is not None:
        record = internal_record.resolve()
        for folder, name in ((root, "the repository"), (out, "the export")):
            if record == folder or folder in record.parents:
                raise ValueError(f"the internal record must be outside {name}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("the target folder must be new or empty")
    if _git(root, "status", "--porcelain").strip():
        raise ValueError("the working tree is not clean: commit first (files are read from HEAD)")
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    tracked = check_allowlist.tracked_files(root)
    chosen = selection(allowlist, extra, profile)
    problems, _ = check_allowlist.check(root, allowlist, extra, tracked)
    if problems:
        raise ValueError("allowlist problems: " + "; ".join(problems))
    references = find_references(root, chosen, tracked)
    if references:
        raise ValueError(
            "references to files that are not published:\n  " + "\n  ".join(references)
        )
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    contents: dict[str, bytes] = {rel: _git(root, "show", f"HEAD:{rel}") for rel in chosen}
    files: dict[str, str] = {
        rel: _sha256(data) for rel, data in contents.items() if rel != MANIFEST_REL
    }
    manifest: dict[str, object] = {
        "schema": 1,
        "package": "nvs-runtime",
        "profile": profile,
        "audit_rules_sha256": files.get(RULES_REL),
        "audit_tool_sha256": files.get(TOOL_REL),
        "files": dict(sorted(files.items())),
    }
    if reference_rules is not None:
        manifest["audit_rules_identical_to_reference"] = (
            reference_rules.read_bytes() == contents.get(RULES_REL)
        )
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    contents[MANIFEST_REL] = manifest_bytes
    named = [rel for rel, data in sorted(contents.items()) if _names_commit(data, commit)]
    if named:
        raise ValueError("the commit id is in the export (nothing written): " + ", ".join(named))
    for rel, data in contents.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if internal_record is not None:
        record_data = {
            "source_commit": commit,
            "branch": branch,
            "profile": profile,
            "file_count": len(contents),
            "manifest_sha256": _sha256(manifest_bytes),
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        internal_record.parent.mkdir(parents=True, exist_ok=True)
        internal_record.write_text(json.dumps(record_data, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit(
    out: Path, rule_files: list[Path]
) -> dict[tuple[int, str], tuple[int, int, list[tuple[str, int]]]]:
    """``(layer, rule id) -> (files, lines, [(file, line)])`` for the files under ``out``."""
    rules: list[audit_artifacts.Rule] = []
    for path in rule_files:
        rules.extend(audit_artifacts.load_rules(path))
    result: dict[tuple[int, str], tuple[int, int, list[tuple[str, int]]]] = {}
    binary: list[tuple[str, int]] = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            binary.append((path.relative_to(out).as_posix(), 0))
    if binary:  # a file the rules cannot read as text (the audit tool names it the same way)
        result[(3, "binary.not-decodable")] = (len(binary), len(binary), binary)
    for rule in rules:
        hits: list[tuple[str, int]] = []
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            rel = path.relative_to(out).as_posix()
            data = path.read_bytes()
            if rule.scope in ("name", "both") and audit_artifacts._matches(
                rule,
                audit_artifacts.normalize(rel, rule.fold_separators) if rule.normalize else rel,
            ):
                hits.append((rel, 0))
            if rule.scope in ("content", "both"):
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if rule.normalize:
                    subject, line_of = audit_artifacts.normalize_with_lines(
                        text, rule.fold_separators
                    )
                    found = [line_of[start] for start, _ in audit_artifacts._matches(rule, subject)]
                else:
                    found = [
                        text.count("\n", 0, start) + 1
                        for start, _ in audit_artifacts._matches(rule, text)
                    ]
                hits.extend((rel, line) for line in sorted(set(found)))
        if hits:
            result[(rule.layer, rule.id)] = (len({f for f, _ in hits}), len(hits), hits)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE.parent)
    ap.add_argument("--profile", choices=PROFILES, default="core")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allowlist", type=Path, default=HERE / "allowlist.txt")
    ap.add_argument("--extra", type=Path, default=HERE / "allowlist-measured-k.txt")
    ap.add_argument("--reference-rules", type=Path, default=None)
    ap.add_argument("--internal-record", type=Path, default=None)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--rules", type=Path, action="append", default=None)
    args = ap.parse_args(argv)
    try:
        manifest = export(
            args.root,
            args.out,
            args.profile,
            args.allowlist,
            args.extra,
            args.reference_rules,
            args.internal_record,
        )
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"export_public: {error}", file=sys.stderr)
        return 1
    files = manifest["files"]
    assert isinstance(files, dict)
    print(f"export_public: {len(files)} file(s) of profile {args.profile}")
    if args.audit:
        rules = args.rules or [args.out / RULES_REL]
        try:
            found = audit(args.out, rules)
        except audit_artifacts.RulesError as error:
            print(f"export_public: bad rules: {error}", file=sys.stderr)
            return 2
        for (layer, rule_id), (n_files, n_lines, hits) in sorted(found.items()):
            print(f"audit L{layer} {rule_id}: {n_files} file(s), {n_lines} line(s)")
            if layer == 3:  # file and line only, never the matched text
                by_file: dict[str, list[int]] = collections.defaultdict(list)
                for rel, line in hits:
                    by_file[rel].append(line)
                for rel, lines in sorted(by_file.items()):
                    print(f"  {rel}: {len(lines)} line(s), first {lines[:5]}")
        if not found:
            print("audit: no hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
