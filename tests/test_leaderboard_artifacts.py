"""Offline validation and portability of frozen leaderboard artifacts."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from test_leaderboard_cli import setup_path as setup_path

from judgearena.benchmarks.elo import artifacts
from judgearena.benchmarks.elo.leaderboard import entry_filename
from judgearena.cli import cli


@pytest.fixture
def submission(setup_path, tmp_path):
    board = tmp_path / "board"
    cli(["leaderboard", "create", str(setup_path), "--output", str(board)])
    model = "Dummy/first"
    cli(["leaderboard", "submit", str(board), "--model", model])
    battles = next((tmp_path / "results").rglob("battles.parquet"))
    entry = board / "entries" / entry_filename(model)
    return SimpleNamespace(board=board, battles=battles, entry=entry)


def test_export_validates_without_original_files(submission, tmp_path, capsys):
    output = tmp_path / "export"
    cli(["leaderboard", "export", str(submission.board), "--output", str(output)])
    target = output / "versions/test-elo-v0.01"
    filename = submission.entry.name
    shutil.rmtree(submission.board)
    shutil.rmtree(tmp_path / "results")
    cli(
        [
            "leaderboard",
            "validate-submission",
            str(target),
            "--entry",
            str(target / "entries" / filename),
            "--battles",
            str(target / "submissions" / Path(filename).stem / "battles.parquet"),
        ]
    )
    assert "Validated Dummy/first" in capsys.readouterr().out


def test_reject_forged_rating(submission):
    data = json.loads(submission.entry.read_text())
    data["overall"]["rating"] += 5
    for summary in data["by_language"].values():
        summary["rating"] += 5
    submission.entry.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="rating does not match recomputed score"):
        artifacts.validate_submission(
            submission.board, submission.entry, submission.battles
        )


def test_reject_external_prompt_path_before_reading(submission):
    path = submission.board / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["judge"]["prompt"]["system_file"] = "../../private.txt"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="relative prompt"):
        artifacts.validate_submission(
            submission.board, submission.entry, submission.battles
        )
