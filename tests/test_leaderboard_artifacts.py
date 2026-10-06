"""Offline validation and portability of frozen leaderboard artifacts."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from test_leaderboard_cli import setup_path as setup_path

from judgearena.benchmarks.elo import artifacts
from judgearena.benchmarks.elo.leaderboard import (
    collapse_swapped_rows,
    entry_filename,
    score_frozen_submission,
)
from judgearena.cli import cli


@pytest.fixture
def submission(setup_path, tmp_path):
    board = tmp_path / "board"
    cli(["leaderboard", "create", str(setup_path), "--output", str(board)])
    model = "Dummy/first"
    cli(["leaderboard", "submit", str(board), "--model", model])
    battles = next((tmp_path / "results").rglob("battles.parquet"))
    entry = board / "entries" / entry_filename(model)
    return SimpleNamespace(board=board, model=model, battles=battles, entry=entry)


def test_export_portable_tree_and_validate_cli(submission, tmp_path, capsys):
    output = tmp_path / "export"
    cli(["leaderboard", "export", str(submission.board), "--output", str(output)])
    target = output / "versions/test-elo-v0.01"
    assert {p.name for p in target.iterdir()} == {
        *artifacts.FROZEN_FILES,
        "leaderboard.json",
        "entries",
        "submissions",
    }
    board = json.loads((target / "leaderboard.json").read_text())
    assert all(entry["source"] == "anchor" for entry in board["entries"])
    cfg = yaml.safe_load((target / "config.yaml").read_text())
    assert "run" not in cfg
    filename = submission.entry.name
    exported_battles = target / "submissions" / Path(filename).stem / "battles.parquet"
    assert exported_battles.read_bytes() == submission.battles.read_bytes()
    cli(
        [
            "leaderboard",
            "validate-submission",
            str(target),
            "--entry",
            str(target / "entries" / filename),
            "--battles",
            str(exported_battles),
        ]
    )
    assert "Validated Dummy/first" in capsys.readouterr().out
    # Exported data remains usable without the original results directory.
    shutil.rmtree(tmp_path / "results")
    artifacts.export_leaderboard(target, tmp_path / "reexport")
    with pytest.raises(FileExistsError):
        artifacts.export_leaderboard(target, output)


@pytest.mark.parametrize("change", ["protocol", "score", "ci", "count", "anchor"])
def test_reject_forged_entries(submission, change):
    data = json.loads(submission.entry.read_text())
    if change == "protocol":
        data["protocol_id"] = "another-version"
    elif change == "score":
        data["overall"]["rating"] += 5
        for summary in data["by_language"].values():
            summary["rating"] += 5
    elif change == "ci":
        data["overall"]["ci_low"] -= 5
    elif change == "count":
        data["overall"]["n_battles"] -= 1
        data["by_language"]["en"]["n_battles"] -= 1
    else:
        data["model"] = "reference"
    submission.entry.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        artifacts.validate_submission(
            submission.board, submission.entry, submission.battles
        )


@pytest.mark.parametrize(
    "change",
    [
        "drop",
        "duplicate",
        "panel_id",
        "lang",
        "question_id",
        "model_a",
        "judge_model",
        "orientation",
        "extra",
        "infinite",
        "negative",
        "string",
        "hard",
        "winner",
    ],
)
def test_reject_invalid_battles(submission, change):
    battles = pd.read_parquet(submission.battles)
    if change == "drop":
        battles = battles.iloc[1:]
    elif change == "duplicate":
        battles = pd.concat([battles, battles.iloc[:1]], ignore_index=True)
    elif change == "extra":
        battles["private_log"] = "not allowed"
    elif change in {"infinite", "negative", "string"}:
        battles["pref"] = battles.pref.astype(object)
        battles.loc[0, "pref"] = {
            "infinite": float("inf"),
            "negative": -0.1,
            "string": "invalid",
        }[change]
        if change == "string":
            battles["pref"] = battles.pref.astype(str)
    elif change == "hard":
        battles.loc[0, "pref_hard"] = 1.0
    elif change == "winner":
        battles.loc[0, "winner"] = "model_a"
    else:
        battles.loc[0, change] = "wrong"
    battles.to_parquet(submission.battles, index=False)
    with pytest.raises(ValueError):
        artifacts.validate_submission(
            submission.board, submission.entry, submission.battles
        )


def test_nan_comparison_preserved_and_recomputed(submission, tmp_path):
    battles = pd.read_parquet(submission.battles)
    battles.loc[0, ["pref", "pref_hard"]] = float("nan")
    battles.loc[0, "winner"] = None
    battles.to_parquet(submission.battles, index=False)
    anchors, _, cfg = artifacts.load_frozen_artifacts(submission.board)
    entry = score_frozen_submission(
        collapse_swapped_rows(battles, cfg.judge.swap_mode),
        submission.model,
        anchors,
        soft_elo=cfg.elo.soft_elo,
        n_bootstraps=cfg.elo.n_bootstraps,
    )
    submission.entry.write_text(entry.model_dump_json())
    validated = artifacts.validate_submission(
        submission.board, submission.entry, submission.battles
    )
    assert validated.overall.n_battles == 7
    output = artifacts.export_leaderboard(submission.board, tmp_path / "partial")
    exported = pd.read_parquet(
        output / "submissions" / submission.entry.stem / "battles.parquet"
    )
    assert len(exported) == 16
    assert exported.pref.isna().sum() == 1


@pytest.mark.parametrize(
    "field,value", [("name", "../escape"), ("version", "../../v2")]
)
def test_safe_version_paths(submission, field, value):
    anchors, _, _ = artifacts.load_frozen_artifacts(submission.board)
    with pytest.raises(ValueError, match="safe path"):
        artifacts.version_path(anchors.model_copy(update={field: value}))
    assert "/" not in entry_filename("../../model/name")


def test_reject_external_prompt_path_before_reading(submission):
    path = submission.board / "config.yaml"
    config = yaml.safe_load(path.read_text())
    config["judge"]["prompt"]["system_file"] = "../../private.txt"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="relative prompt"):
        artifacts.validate_submission(
            submission.board, submission.entry, submission.battles
        )


def test_export_requires_unambiguous_validated_battles(submission, tmp_path):
    missing = tmp_path / "missing-results"
    with pytest.raises(ValueError, match="found 0"):
        artifacts.export_leaderboard(
            submission.board, tmp_path / "missing", results_dir=missing
        )
    duplicate = tmp_path / "results/duplicate/battles.parquet"
    duplicate.parent.mkdir()
    shutil.copyfile(submission.battles, duplicate)
    with pytest.raises(ValueError, match="found 2"):
        artifacts.export_leaderboard(
            submission.board, tmp_path / "ambiguous", results_dir=tmp_path / "results"
        )


@pytest.mark.parametrize("swap_mode", ["fixed", "random"])
def test_single_order_protocol(setup_path, tmp_path, swap_mode):
    config = yaml.safe_load(setup_path.read_text())
    config["evaluation"]["judge"]["swap_mode"] = swap_mode
    setup_path.write_text(yaml.safe_dump(config))
    board = tmp_path / "single"
    cli(["leaderboard", "create", str(setup_path), "--output", str(board)])
    cli(["leaderboard", "submit", str(board), "--model", "Dummy/single"])
    battles = next((tmp_path / "results").rglob("battles.parquet"))
    entry = board / "entries" / entry_filename("Dummy/single")
    result = artifacts.validate_submission(board, entry, battles)
    assert result.overall.n_battles == 8
    frame = pd.read_parquet(battles)
    frame.loc[0, "orientation"] = "direct"
    frame.to_parquet(battles, index=False)
    with pytest.raises(ValueError, match="orientations"):
        artifacts.validate_submission(board, entry, battles)
