"""Exercise release guards and tagging against a disposable local remote."""

import os
import subprocess
from pathlib import Path

import yaml


def test_release_version_and_tag(tmp_path):
    workflow = yaml.safe_load(
        (Path(__file__).parents[1] / ".github/workflows/publish.yml").read_text()
    )
    steps = {step["name"]: step for step in workflow["jobs"]["run"]["steps"]}
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Release test",
        "GIT_COMMITTER_NAME": "Release test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "GITHUB_REF": "refs/heads/main",
        "REQUESTED_VERSION": "0.2.0",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "RELEASE_TAG": "v0.2.0",
    }

    def run(*args):
        return subprocess.run(args, cwd=tmp_path, env=env, capture_output=True)

    def check_version():
        return run("bash", "-e", "-c", steps["Check release version"]["run"]).returncode

    assert run("git", "init", "-b", "main").returncode == 0
    assert run("git", "init", "--bare", "remote.git").returncode == 0
    assert run("git", "remote", "add", "origin", "remote.git").returncode == 0
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.2.0"\n')
    assert run("git", "add", "pyproject.toml").returncode == 0
    assert run("git", "commit", "-m", "Initial version").returncode == 0
    assert check_version() == 0
    for field, bad in (
        ("GITHUB_REF", "refs/heads/feature"),
        ("REQUESTED_VERSION", "0.3.0"),
    ):
        original = env[field]
        env[field] = bad
        assert check_version() != 0
        env[field] = original
    for _ in range(2):
        assert (
            run("bash", "-e", "-c", steps["Create release tag"]["run"]).returncode == 0
        )
    assert b"refs/tags/v0.2.0" in run("git", "ls-remote", "--tags", "origin").stdout
    env.update(GITHUB_EVENT_NAME="push", GITHUB_REF="refs/tags/v0.2.0")
    assert check_version() == 0
    env["GITHUB_REF"] = "refs/tags/v0.3.0"
    assert check_version() != 0
    env["GITHUB_REF"] = "refs/tags/v0.2.0"
    assert run("git", "commit", "--allow-empty", "-m", "Another commit").returncode == 0
    assert check_version() != 0
