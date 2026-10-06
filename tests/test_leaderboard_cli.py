"""Offline end-to-end creation and submission through the public CLI."""

import json

import pandas as pd
import pytest
import yaml

from judgearena.benchmarks.elo import freeze, runner
from judgearena.cli import cli
from judgearena.models import DummyModel


@pytest.fixture
def setup_path(tmp_path, monkeypatch):
    def conversation(prompt, answer):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

    battles = pd.DataFrame(
        [
            {
                "question_id": f"{language}-{index}",
                "lang": language,
                "model_a": "reference",
                "model_b": "anchor",
                "winner": "model_a" if index % 2 else "model_b",
                "conversation_a": conversation(
                    f"{language} prompt {index}", "answer A"
                ),
                "conversation_b": conversation(
                    f"{language} prompt {index}", "answer B"
                ),
            }
            for language in ("en", "fr")
            for index in range(8)
        ]
    )
    monkeypatch.setattr(freeze, "load_battles", lambda _task: battles)
    path = tmp_path / "setup.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "test-elo",
                "version": "0.01",
                "languages": ["en", "fr"],
                "anchor_models": ["anchor"],
                "battles_per_language": 4,
                "min_anchor_battles": 8,
                "evaluation": {
                    "task": "elo-comparia",
                    "model": {"name": "Dummy/placeholder", "max_out_tokens": 32},
                    "judge": {
                        "model": "Dummy/score A: 5 score B: 5",
                        "swap_mode": "both",
                    },
                    "elo": {"baseline_model": "reference", "n_bootstraps": 2},
                    "run": {
                        "store_root": str(tmp_path / "cache"),
                        "result_folder": str(tmp_path / "results"),
                        "no_log_file": True,
                    },
                },
            }
        )
    )
    return path


def test_adding_models_preserves_frozen_inputs_and_previous_entries(
    setup_path, tmp_path, monkeypatch
):
    output = tmp_path / "board"
    cli(["leaderboard", "create", str(setup_path), "--output", str(output)])
    frozen_names = (
        "config.yaml",
        "anchors.json",
        "panel.parquet",
        "judge-system-prompt.txt",
        "judge-user-prompt.txt",
    )
    frozen = {name: (output / name).read_bytes() for name in frozen_names}
    previous_entries = json.loads((output / "leaderboard.json").read_text())["entries"]

    def fail_arena_load(_task):
        raise AssertionError("submission must use the frozen panel, not the arena")

    monkeypatch.setattr(runner, "load_battles", fail_arena_load)
    for candidate in ("Dummy/first", "Dummy/second"):
        cli(["leaderboard", "submit", str(output), "--model", candidate])
        entries = json.loads((output / "leaderboard.json").read_text())["entries"]
        assert candidate in {row["model"] for row in entries}
        assert [row for row in entries if row["model"] != candidate] == previous_entries
        assert {name: (output / name).read_bytes() for name in frozen_names} == frozen
        previous_entries = entries


def test_submission_survives_one_unparseable_reversed_judgment(
    setup_path, tmp_path, monkeypatch
):
    output = tmp_path / "partial"
    cli(["leaderboard", "create", str(setup_path), "--output", str(output)])
    batch = DummyModel.batch
    judge_calls = 0

    def batch_with_one_bad_output(model, inputs, **kwargs):
        nonlocal judge_calls
        outputs = batch(model, inputs, **kwargs)
        if model.message == "score A: 5 score B: 5":
            judge_calls += 1
            if judge_calls == 2:
                outputs[0] = "No scores supplied."
        return outputs

    monkeypatch.setattr(DummyModel, "batch", batch_with_one_bad_output)
    cli(["leaderboard", "submit", str(output), "--model", "Dummy/partial"])
    board = json.loads((output / "leaderboard.json").read_text())
    entry = next(row for row in board["entries"] if row["source"] == "submission")
    assert entry["overall"]["n_battles"] == 7
    assert entry["by_language"]["en"]["n_battles"] == 3
    assert entry["by_language"]["fr"]["n_battles"] == 4
