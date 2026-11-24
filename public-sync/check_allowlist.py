#!/usr/bin/env python3
"""Check the public allowlist of nvs-runtime (check only; it never edits anything).

``allowlist.txt`` lists the files that may be published (the core profile);
``allowlist-measured-k.txt`` lists the files that only the profile with the stored measured-K data
adds. Both are one repository-relative path per line; ``#`` starts a comment. The default is NOT
listed: a tracked file that is on neither list is not published, and this tool names it.

* every listed file exists (and, with ``--git``, is tracked); nothing is listed twice;
* no entry is absolute, climbs out of the repository, or points into a private location;
* with ``--git``: the tracked files that are on neither list are printed (they are not published).

Standard library only. Exit status 0 = clean, 1 = problems (listed on stderr).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent
PRIVATE_PARTS = {".local", ".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}
PRIVATE_NAMES = {".coverage"}


def read_list(path: Path) -> list[str]:
    if not path.is_file():
        return []
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def tracked_files(root: Path) -> list[str]:
    done = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, check=True
    )
    return [name for name in done.stdout.split("\0") if name]


def check(
    root: Path,
    allowlist: Path,
    extra: Path | None = None,
    tracked: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """``(problems, not_published)``. ``not_published`` is empty without ``tracked``."""
    core = read_list(allowlist)
    more = read_list(extra) if extra is not None else []
    problems: list[str] = []
    seen: set[str] = set()
    for entry in [*core, *more]:
        if entry in seen:
            problems.append(f"listed twice: {entry}")
        seen.add(entry)
        pure = PurePosixPath(entry)
        if pure.is_absolute() or ".." in pure.parts:
            problems.append(f"entry is not a relative path inside the repository: {entry}")
            continue
        if PRIVATE_PARTS & set(pure.parts) or pure.name in PRIVATE_NAMES:
            problems.append(f"entry points at a private location: {entry}")
        if not (root / entry).is_file():
            problems.append(f"listed file does not exist: {entry}")
        elif tracked is not None and entry not in set(tracked):
            problems.append(f"listed file is not tracked: {entry}")
    not_published = sorted(set(tracked) - seen) if tracked is not None else []
    return problems, not_published


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--allowlist", type=Path, default=HERE / "allowlist.txt")
    ap.add_argument("--extra", type=Path, default=HERE / "allowlist-measured-k.txt")
    ap.add_argument("--git", action="store_true", help="also compare with the tracked files")
    args = ap.parse_args(argv)
    tracked = tracked_files(args.root) if args.git else None
    problems, not_published = check(args.root, args.allowlist, args.extra, tracked)
    for name in not_published:
        print(f"not published: {name}")
    for problem in problems:
        print(f"check_allowlist: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"check_allowlist: ok ({len(not_published)} tracked file(s) not published)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
