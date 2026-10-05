"""Offline end-to-end creation and submission through the public CLI."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from judgearena.benchmarks.elo import freeze, runner
from judgearena.benchmarks.elo.cli import load_setup
from judgearena.benchmarks.elo.leaderboard import entry_filename
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


def test_create_submit_show_and_version_isolation(
    setup_path, tmp_path, capsys, monkeypatch
):
    output = tmp_path / "v0.01"
    cli(["leaderboard", "create", str(setup_path), "--output", str(output)])
    frozen_names = ("config.yaml", "anchors.json", "panel.parquet")
    frozen = {name: (output / name).read_bytes() for name in frozen_names}
    board = json.loads((output / "leaderboard.json").read_text())
    assert board["version"] == "0.01"
    original_entries = board["entries"]
    assert all(entry["source"] == "anchor" for entry in original_entries)

    def fail_arena_load(_task):
        raise AssertionError("submission must use the frozen panel, not the arena")

    monkeypatch.setattr(runner, "load_battles", fail_arena_load)
    for command, candidate in (("evaluate", "Dummy/first"), ("submit", "Dummy/second")):
        capsys.readouterr()
        cli(["leaderboard", command, str(output), "--model", candidate])
        saved_line = next(
            line
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("Saved run: ")
        )
        run_dir = Path(saved_line.removeprefix("Saved run: "))
        assert (run_dir / "battles.parquet").is_file()
        assert json.loads((run_dir / "entry.json").read_text()) == json.loads(
            (output / "entries" / entry_filename(candidate)).read_text()
        )
    board = json.loads((output / "leaderboard.json").read_text())
    assert len(board["entries"]) == 4
    assert [
        row for row in board["entries"] if row["source"] == "anchor"
    ] == original_entries
    assert {name: (output / name).read_bytes() for name in frozen_names} == frozen
    for row in board["entries"]:
        if row["source"] == "submission":
            assert row["overall"]["n_battles"] == 8  # Both orders are one comparison.
            assert row["overall"]["rating"] == pytest.approx(1000.0)
            assert set(row["by_language"]) == {"en", "fr"}

    capsys.readouterr()
    cli(["leaderboard", "show", str(output)])
    shown = capsys.readouterr().out
    assert "test-elo v0.01" in shown
    assert "Human anchor" in shown and "Judge estimate" in shown
    assert "Dummy/first" in shown and "Dummy/second" in shown
    with pytest.raises(SystemExit, match="2"):
        cli(["leaderboard", "submit", str(output), "--model", "Dummy/first"])
    with pytest.raises(SystemExit, match="2"):
        cli(["leaderboard", "create", str(setup_path), "--output", str(output)])

    data = yaml.safe_load(setup_path.read_text())
    data["version"] = "0.02"
    setup_path.write_text(yaml.safe_dump(data))
    next_version = tmp_path / "v0.02"
    cli(["leaderboard", "create", str(setup_path), "--output", str(next_version)])
    next_board = json.loads((next_version / "leaderboard.json").read_text())
    assert next_board["version"] == "0.02"
    assert next_board["protocol_id"] != board["protocol_id"]
    assert len(next_board["entries"]) == 2
    assert json.loads((output / "leaderboard.json").read_text()) == board


@pytest.mark.parametrize(
    "name,languages", [("english", ["en"]), ("multilingual", ["en", "fr", "de", "es"])]
)
def test_recommended_setups(name, languages):
    path = Path(__file__).parents[1] / "configs/leaderboards" / f"{name}-v0.01.yaml"
    setup = load_setup(path)
    assert setup.version == "0.01"
    assert setup.languages == languages
    assert setup.min_anchor_battles == setup.battles_per_language == 100
    assert setup.evaluation.elo.n_bootstraps == 1000
    assert setup.evaluation.judge.swap_mode == "both"
    assert setup.evaluation.generation.truncate_all_input_chars is None
    assert setup.evaluation.generation.truncate_judge_input_chars is None


def test_setup_resolves_relative_prompt_files(setup_path):
    (setup_path.parent / "system.txt").write_text("Judge the answers.")
    (setup_path.parent / "user.txt").write_text(
        "{user_prompt} {completion_A} {completion_B}"
    )
    data = yaml.safe_load(setup_path.read_text())
    data["evaluation"]["judge"]["prompt"] = {
        "system_file": "system.txt",
        "user_file": "user.txt",
        "parser": "score",
    }
    setup_path.write_text(yaml.safe_dump(data))
    setup = load_setup(setup_path)
    assert setup.evaluation.judge.prompt.system_file == setup_path.parent / "system.txt"


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
    result_path = next((tmp_path / "results").rglob("results-*.json"))
    result = json.loads(result_path.read_text())
    assert result["num_battles"] == 7
    assert result["sampling_metadata"]["attempted_battles"] == 8
    assert result["sampling_metadata"]["skipped_battles"] == 1
    assert json.loads((result_path.parent / "entry.json").read_text()) == entry
