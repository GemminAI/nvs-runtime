"""Tests of the public-sync tools: the allowlist check, the export and the reference check."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_allowlist  # noqa: E402
import export_public  # noqa: E402


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "public-sync").mkdir()
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "tests" / "test_a.py").write_text("import helper\n", encoding="utf-8")
    (root / "tests" / "helper.py").write_text("y = 2\n", encoding="utf-8")
    (root / "tests" / "data").mkdir()
    (root / "tests" / "data" / "one.json").write_text("{}\n", encoding="utf-8")
    (root / "tests" / "test_b.py").write_text('P = "data"\n', encoding="utf-8")
    (root / "private.txt").write_text("private\n", encoding="utf-8")
    (root / "public-sync" / "audit-rules.toml").write_text(
        'visibility = "public"\n', encoding="utf-8"
    )
    (root / "public-sync" / "audit_artifacts.py").write_text("# tool\n", encoding="utf-8")
    (root / "public-sync" / "allowlist.txt").write_text(
        "# c\nsrc/a.py\ntests/test_a.py\ntests/helper.py\npublic-sync/audit-rules.toml\n"
        "public-sync/audit_artifacts.py\n",
        encoding="utf-8",
    )
    (root / "public-sync" / "allowlist-measured-k.txt").write_text(
        "tests/data/one.json\ntests/test_b.py\n", encoding="utf-8"
    )
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"],
    ):
        subprocess.run(["git", *args], cwd=root, check=True)
    return root


def _paths(root: Path) -> tuple[Path, Path]:
    return root / "public-sync" / "allowlist.txt", root / "public-sync" / "allowlist-measured-k.txt"


def test_a_clean_list_names_what_is_not_published(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    problems, not_published = check_allowlist.check(
        root, allow, extra, check_allowlist.tracked_files(root)
    )
    assert problems == []
    assert "private.txt" in not_published and "src/a.py" not in not_published


def test_a_listed_file_that_is_missing_or_listed_twice_is_a_problem(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    allow.write_text(allow.read_text() + "src/gone.py\nsrc/a.py\n", encoding="utf-8")
    problems, _ = check_allowlist.check(root, allow, extra)
    assert "listed file does not exist: src/gone.py" in problems
    assert "listed twice: src/a.py" in problems


def test_an_entry_in_a_private_place_is_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    allow.write_text("../outside.txt\n.local/x.log\n", encoding="utf-8")
    problems, _ = check_allowlist.check(root, allow, extra)
    assert any("not a relative path" in p for p in problems)
    assert any("private location" in p for p in problems)


def test_the_export_holds_only_the_profile_and_a_manifest_without_a_commit(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    out = tmp_path / "out"
    manifest = export_public.export(
        root, out, "measured-k", allow, extra, root / "public-sync" / "audit-rules.toml"
    )
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert "private.txt" not in names and "tests/data/one.json" in names
    assert not (out / ".git").exists()
    written = json.loads((out / export_public.MANIFEST_REL).read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    )
    assert written == manifest and "source_commit" not in written
    assert head.stdout.strip() not in (out / export_public.MANIFEST_REL).read_text(encoding="utf-8")
    assert export_public.MANIFEST_REL not in written["files"]
    assert written["audit_rules_identical_to_reference"] is True


def test_the_core_profile_leaves_out_the_added_files_and_a_reference_to_them_is_reported(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    chosen = export_public.selection(allow, extra, "core")
    assert "tests/data/one.json" not in chosen and "tests/test_b.py" not in chosen
    # a chosen file that imports a module or names a folder that is not published
    (root / "tests" / "test_a.py").write_text(
        "import helper\nimport test_b\nQ = 'data'\n", encoding="utf-8"
    )
    refs = export_public.find_references(root, chosen, check_allowlist.tracked_files(root))
    assert any("imports test_b" in r for r in refs) and any("names 'data'" in r for r in refs)


def test_a_dirty_tree_or_a_target_inside_the_repository_is_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    with pytest.raises(ValueError, match="outside the repository"):
        export_public.export(root, root / "out", "core", allow, extra)
    (root / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not clean"):
        export_public.export(root, tmp_path / "out", "core", allow, extra)


def test_an_internal_record_inside_the_export_or_the_repository_is_refused(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    out = tmp_path / "out"
    for record in (out / "record.json", root / "record.json", out, root):
        with pytest.raises(ValueError, match="internal record must be outside"):
            export_public.export(root, out, "core", allow, extra, None, record)
    assert not out.exists()


def test_the_internal_record_holds_the_commit_and_the_manifest_digest(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    out, record = tmp_path / "out", tmp_path / "elsewhere" / "record.json"
    export_public.export(root, out, "core", allow, extra, None, record)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    data = json.loads(record.read_text(encoding="utf-8"))
    manifest_file = out / export_public.MANIFEST_REL
    assert set(data) == {
        "source_commit",
        "branch",
        "profile",
        "file_count",
        "manifest_sha256",
        "created_utc",
    }
    assert data["source_commit"] == head and data["profile"] == "core"
    assert data["manifest_sha256"] == hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    assert data["file_count"] == len([p for p in out.rglob("*") if p.is_file()])
    assert data["created_utc"].endswith("Z") and data["branch"]


def test_a_file_that_names_the_commit_stops_the_export_before_anything_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    allow, extra = _paths(root)
    (root / "src" / "a.py").write_text("# built from abcdef1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "d"],
        cwd=root,
        check=True,
    )
    real_git = export_public._git

    def git(root_: Path, *args: str) -> bytes:
        if args == ("rev-parse", "HEAD"):
            return b"abcdef1234\n"
        return real_git(root_, *args)

    monkeypatch.setattr(export_public, "_git", git)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="commit id is in the export.*src/a.py"):
        export_public.export(root, out, "core", allow, extra)
    assert not out.exists()


def test_a_commit_id_is_found_in_full_or_as_its_first_seven_characters() -> None:
    commit = "0123456789abcdef" * 2 + "01234567"
    assert export_public._names_commit(b"x 0123456 y", commit)
    assert export_public._names_commit(commit.upper().encode(), commit)
    assert not export_public._names_commit(b"x 012345 y", commit)
