"""The release workflow: manual, from main only, Trusted Publishing, no stored credentials."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"
MAIN = "github.ref == 'refs/heads/main'"


def _load() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_it_is_started_by_hand_only_and_asks_where_to_publish() -> None:
    workflow = _load()
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` as True
    assert set(triggers) == {"workflow_dispatch"}
    target = triggers["workflow_dispatch"]["inputs"]["target"]
    assert target["type"] == "choice" and target["options"] == ["testpypi", "pypi"]
    assert target["required"] is True and "default" not in target


def test_only_the_two_publish_jobs_may_get_an_id_token() -> None:
    workflow = _load()
    assert workflow["permissions"] == {}
    jobs = workflow["jobs"]
    assert set(jobs) == {"build", "publish-testpypi", "publish-pypi"}
    assert jobs["build"]["permissions"] == {"contents": "read"}
    for name in ("publish-testpypi", "publish-pypi"):
        assert jobs[name]["permissions"] == {"id-token": "write"}
        assert jobs[name]["needs"] == "build"


def test_every_job_runs_on_main_only_and_each_publish_job_has_its_target_and_environment() -> None:
    jobs = _load()["jobs"]
    assert jobs["build"]["if"] == MAIN
    assert jobs["publish-testpypi"]["if"] == f"{MAIN} && inputs.target == 'testpypi'"
    assert jobs["publish-pypi"]["if"] == f"{MAIN} && inputs.target == 'pypi'"
    assert jobs["publish-testpypi"]["environment"] == "testpypi"
    assert jobs["publish-pypi"]["environment"] == "pypi"


def test_the_publish_step_uses_the_official_action_without_a_token_or_skip_existing() -> None:
    jobs = _load()["jobs"]
    for name, repository_url in (
        ("publish-testpypi", "https://test.pypi.org/legacy/"),
        ("publish-pypi", None),
    ):
        steps = jobs[name]["steps"]
        (publish,) = [s for s in steps if "gh-action-pypi-publish" in s.get("uses", "")]
        assert publish["uses"] == "pypa/gh-action-pypi-publish@release/v1"
        assert publish.get("with", {}).get("repository-url") == repository_url
        assert set(publish.get("with", {})) <= {"repository-url"}
        assert "env" not in publish


def test_the_build_job_checks_the_distributions_and_hands_them_on() -> None:
    steps = _load()["jobs"]["build"]["steps"]
    runs = [s["run"] for s in steps if "run" in s]
    assert "python -m build" in runs and "twine check dist/*" in runs
    assert "sha256sum dist/*" in runs
    uploads = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact@")]
    assert [u["with"] for u in uploads] == [{"name": "dist", "path": "dist/"}]


def test_no_secret_or_token_is_referenced() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "secrets." not in text and "API_TOKEN" not in text.upper()
    assert "skip-existing" not in text and "password" not in text
