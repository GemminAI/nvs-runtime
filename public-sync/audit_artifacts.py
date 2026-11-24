#!/usr/bin/env python3
"""Audit a built nvs-observation wheel (and sdist) before it can be published.

For each artifact it checks, without extracting to disk:
  1. the file list equals the expected list derived from allowlist.txt (no extra, none missing);
  2. every file name and every file content against the audit rules (TOML data, see --rules);
  3. METADATA / PKG-INFO: License Apache-2.0, Requires-Dist is numpy only, Requires-Python
     >=3.12, and no local URL or path.

The rules are data. This program holds no list of words. The default rules are the public ones
(audit-rules.toml next to this file); --rules may be given several times and then replaces the
default. A missing rules file or a malformed rule is an error, never a silent pass.

Standard library only. Exit status 0 = clean, 1 = problems (listed on stderr), 2 = bad rules.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tomllib
import unicodedata
import zipfile
from dataclasses import dataclass, field
from email.parser import Parser
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RULES = HERE / "audit-rules.toml"

KINDS = ("literal", "regex")
SCOPES = ("content", "name", "both")
REQUIRED = ("id", "layer", "kind", "pattern", "scope", "description")
PRIVATE_NOTE = "<非公開ルール>"
ZERO_WIDTH = re.compile("[​-‍⁠﻿­]")
SEPARATORS = re.compile(r"[_\-\s]+")
URL = re.compile(r"(?:https?|file|ssh|git)://[^\s)>\"']+")
LOCAL_URL_MARKERS = ("localhost", "127.0.0.1", "file://")


class RulesError(Exception):
    """The rules file is missing or malformed."""


# ---- normalization --------------------------------------------------------------------------


def normalize_with_lines(text: str, fold_separators: bool = True) -> tuple[str, list[int]]:
    """Normalize ``text`` and say, for every output character, the source line it came from.

    Steps: NFKC; casefold; hiragana (U+3041..U+3096) to katakana (+0x60); drop zero-width
    characters (U+200B..U+200D, U+2060, U+FEFF, U+00AD); and, only when ``fold_separators`` is
    true, every run of "_", "-" and white space becomes one space. Lines are normalized one by
    one so that the line numbers survive; the separator step then runs over the joined text, so a
    phrase broken across lines still matches. A rule with ``fold_separators = false`` keeps
    "_", "-" and white space as they are (so that, for example, "a_b" and "a-b" differ).
    """
    chars: list[str] = []
    lines: list[int] = []
    source_lines = text.split("\n")
    for number, line in enumerate(source_lines, start=1):
        piece = unicodedata.normalize("NFKC", line).casefold()
        piece = "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in piece)
        piece = ZERO_WIDTH.sub("", piece)
        if number < len(source_lines):
            piece += "\n"
        chars.extend(piece)
        lines.extend([number] * len(piece))
    if not fold_separators:
        return "".join(chars), lines
    out: list[str] = []
    out_lines: list[int] = []
    previous_separator = False
    for char, number in zip(chars, lines, strict=True):
        if SEPARATORS.fullmatch(char):
            if not previous_separator:
                out.append(" ")
                out_lines.append(number)
            previous_separator = True
        else:
            out.append(char)
            out_lines.append(number)
            previous_separator = False
    return "".join(out), out_lines


def normalize(text: str, fold_separators: bool = True) -> str:
    return normalize_with_lines(text, fold_separators)[0]


# ---- rules ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    id: str
    layer: int
    kind: str
    scope: str
    normalize: bool
    fold_separators: bool
    public: bool
    needles: tuple[str, ...] = ()
    regex: re.Pattern[str] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Finding:
    rule: Rule
    where: str
    line: int | None
    text: str

    def describe(self) -> str:
        place = self.where if self.line is None else f"{self.where}:{self.line}"
        shown = self.text if self.rule.public else PRIVATE_NOTE
        return f"{place}: rule {self.rule.id} (layer {self.rule.layer}): {shown}"


def _fail(path: Path, message: str) -> RulesError:
    return RulesError(f"{path}: {message}")


def load_rules(path: Path) -> list[Rule]:
    if not path.is_file():
        raise RulesError(f"rules file not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise _fail(path, f"not valid TOML: {error}") from error
    visibility = data.get("visibility", "private")
    if visibility not in ("public", "private"):
        raise _fail(path, f"visibility must be 'public' or 'private', got {visibility!r}")
    raw_rules = data.get("rule")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise _fail(path, "no [[rule]] entries")
    rules: list[Rule] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules, start=1):
        if not isinstance(raw, dict):
            raise _fail(path, f"rule #{index} is not a table")
        for key in REQUIRED:
            if key not in raw:
                raise _fail(path, f"rule #{index} lacks the field {key!r}")
        rule_id = raw["id"]
        if not isinstance(rule_id, str) or not rule_id:
            raise _fail(path, f"rule #{index}: id must be a non-empty string")
        if rule_id in seen:
            raise _fail(path, f"duplicate rule id {rule_id!r}")
        seen.add(rule_id)
        if raw["layer"] not in (1, 2, 3) or isinstance(raw["layer"], bool):
            raise _fail(path, f"rule {rule_id!r}: layer must be 1, 2 or 3")
        if raw["kind"] not in KINDS:
            raise _fail(path, f"rule {rule_id!r}: unknown kind {raw['kind']!r}")
        if raw["scope"] not in SCOPES:
            raise _fail(path, f"rule {rule_id!r}: unknown scope {raw['scope']!r}")
        pattern = raw["pattern"]
        if not isinstance(pattern, str) or not pattern:
            raise _fail(path, f"rule {rule_id!r}: pattern must be a non-empty string")
        do_normalize = raw.get("normalize", True)
        if not isinstance(do_normalize, bool):
            raise _fail(path, f"rule {rule_id!r}: normalize must be true or false")
        fold = raw.get("fold_separators", True)
        if not isinstance(fold, bool):
            raise _fail(path, f"rule {rule_id!r}: fold_separators must be true or false")
        aliases = raw.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) and a for a in aliases):
            raise _fail(path, f"rule {rule_id!r}: aliases must be a list of strings")
        needles: tuple[str, ...] = ()
        compiled: re.Pattern[str] | None = None
        if raw["kind"] == "literal":
            texts = [pattern, *aliases]
            needles = tuple(normalize(t, fold) if do_normalize else t for t in texts)
        else:
            if aliases:
                raise _fail(path, f"rule {rule_id!r}: aliases are for literal rules only")
            try:
                compiled = re.compile(pattern)
            except re.error as error:
                raise _fail(path, f"rule {rule_id!r}: bad regular expression: {error}") from error
        rules.append(
            Rule(
                id=rule_id,
                layer=raw["layer"],
                kind=raw["kind"],
                scope=raw["scope"],
                normalize=do_normalize,
                fold_separators=fold,
                public=visibility == "public",
                needles=needles,
                regex=compiled,
            )
        )
    return rules


def _matches(rule: Rule, text: str) -> list[tuple[int, str]]:
    """``(start offset, matched text)`` of every match of ``rule`` in ``text``."""
    found: list[tuple[int, str]] = []
    if rule.kind == "literal":
        for needle in rule.needles:
            start = text.find(needle)
            while start != -1:
                found.append((start, needle))
                start = text.find(needle, start + len(needle))
    else:
        assert rule.regex is not None
        found.extend((m.start(), m.group(0)) for m in rule.regex.finditer(text))
    return sorted(found)


def scan_text(where: str, text: str, rules: list[Rule], scope: str) -> list[Finding]:
    """Apply the rules whose scope covers ``scope`` ("content" or "name") to ``text``."""
    findings: list[Finding] = []
    # Two normalized views of the text: separators folded, and separators kept. Each rule uses
    # the one its fold_separators field asks for (built only when some rule needs it).
    views: dict[bool, tuple[str, list[int]]] = {}
    for rule in rules:
        if rule.scope not in (scope, "both"):
            continue
        if rule.normalize:
            if rule.fold_separators not in views:
                views[rule.fold_separators] = normalize_with_lines(text, rule.fold_separators)
            subject, line_of = views[rule.fold_separators]
        else:
            subject, line_of = text, []
        for start, hit in _matches(rule, subject):
            line: int | None = None
            if scope == "content":
                line = line_of[start] if rule.normalize else subject.count("\n", 0, start) + 1
            findings.append(Finding(rule, where, line, hit if len(hit) <= 40 else hit[:37] + "..."))
            break  # one finding per rule and file is enough to fail it
    return findings


def scan_member(name: str, data: bytes, rules: list[Rule]) -> list[Finding]:
    findings = scan_text(name, name, rules, "name")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is not None:
        return findings + scan_text(name, text, rules, "content")
    # Not text: say so, and run the layer-3 patterns over a byte-for-byte (latin-1) decoding.
    marker = Rule("binary.not-decodable", 3, "regex", "content", False, True, True)
    findings.append(Finding(marker, name, None, "content could not be decoded as UTF-8"))
    latin = data.decode("latin-1")
    secrets = [r for r in rules if r.layer == 3 and r.kind == "regex"]
    return findings + scan_text(name, latin, secrets, "content")


# ---- the artifact -----------------------------------------------------------------------------


def read_allowlist(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _dist_prefix(filename: str, suffixes: tuple[str, ...]) -> str:
    base = filename
    for suffix in suffixes:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def expected_wheel_files(allowlist: list[str], wheel_name: str) -> set[str]:
    parts = wheel_name[: -len(".whl")].split("-")
    dist_info = f"{parts[0]}-{parts[1]}.dist-info"
    expected = {e[len("src/") :] for e in allowlist if e.startswith("src/")}
    expected |= {f"{dist_info}/{n}" for n in ("METADATA", "RECORD", "WHEEL")}
    expected |= {f"{dist_info}/licenses/{e}" for e in allowlist if e in ("LICENSE", "NOTICE")}
    return expected


def expected_sdist_files(allowlist: list[str], sdist_name: str) -> set[str]:
    root = _dist_prefix(sdist_name, (".tar.gz",))
    expected = {f"{root}/{e}" for e in allowlist}
    expected |= {f"{root}/PKG-INFO", f"{root}/pyproject.toml"}
    return expected


def _members(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    files[info.filename] = zf.read(info)
    else:
        with tarfile.open(path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    extracted = tf.extractfile(member)
                    files[member.name] = extracted.read() if extracted else b""
    return files


def check_metadata(name: str, data: bytes) -> list[str]:
    msg = Parser().parsestr(data.decode("utf-8", errors="replace"))
    problems: list[str] = []
    lic = msg.get("License-Expression") or msg.get("License") or ""
    if "Apache-2.0" not in lic:
        problems.append(f"{name}: License is not Apache-2.0 (got {lic!r})")
    requires = msg.get_all("Requires-Dist") or []
    names = [re.split(r"[<>=!~;\[ (]", r, maxsplit=1)[0].strip().lower() for r in requires]
    if names != ["numpy"]:
        problems.append(f"{name}: Requires-Dist must be numpy only (got {requires!r})")
    if (msg.get("Requires-Python") or "").replace(" ", "") != ">=3.12":
        got = msg.get("Requires-Python")
        problems.append(f"{name}: Requires-Python must be >=3.12 (got {got!r})")
    header_text = "\n".join(f"{k}: {v}" for k, v in msg.items())
    for url in URL.findall(header_text):
        if any(marker in url for marker in LOCAL_URL_MARKERS):
            problems.append(f"{name}: local URL in metadata: {url}")
    return problems


def audit(path: Path, allowlist: list[str], rules: list[Rule]) -> list[str]:
    files = _members(path)
    if path.suffix == ".whl":
        expected = expected_wheel_files(allowlist, path.name)
        metadata_names = [n for n in files if n.endswith(".dist-info/METADATA")]
    else:
        expected = expected_sdist_files(allowlist, path.name)
        metadata_names = [n for n in files if n.endswith("/PKG-INFO")]
    problems: list[str] = []
    for extra in sorted(set(files) - expected):
        problems.append(f"{path.name}: file not expected (not on the allowlist): {extra}")
    for missing in sorted(expected - set(files)):
        problems.append(f"{path.name}: expected file is missing: {missing}")
    for name, data in sorted(files.items()):
        for finding in scan_member(f"{path.name}:{name}", data, rules):
            problems.append(finding.describe())
    if not metadata_names:
        problems.append(f"{path.name}: no METADATA / PKG-INFO found")
    for name in metadata_names:
        problems.extend(check_metadata(f"{path.name}:{name}", files[name]))
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts", nargs="+", type=Path, help="wheel (.whl) and/or sdist (.tar.gz)")
    ap.add_argument("--allowlist", type=Path, default=HERE / "allowlist.txt")
    ap.add_argument(
        "--rules",
        type=Path,
        action="append",
        default=None,
        help="TOML rules file (repeatable). Default: audit-rules.toml next to this script.",
    )
    args = ap.parse_args(argv)
    try:
        rules: list[Rule] = []
        for rules_path in args.rules or [DEFAULT_RULES]:
            rules.extend(load_rules(rules_path))
    except RulesError as error:
        print(f"audit_artifacts: bad rules: {error}", file=sys.stderr)
        return 2
    allowlist = read_allowlist(args.allowlist)
    problems: list[str] = []
    for artifact in args.artifacts:
        found = audit(artifact, allowlist, rules)
        problems.extend(found)
        print(
            f"audit_artifacts: {artifact.name}: {len(_members(artifact))} files, "
            f"{len(found)} problem(s), {len(rules)} rule(s)"
        )
    for p in problems:
        print(f"audit_artifacts: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
