from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from judgearena.benchmarks.elo import calibration, freeze
from judgearena.benchmarks.elo.leaderboard import (
    AnchorSet,
    comparable_config,
    load_frozen_files,
)
from judgearena.config import load_config


def _conversation(instruction: str, completion: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": completion},
    ]


def _write_config(tmp_path: Path, *, calibrate: bool = False) -> Path:
    (tmp_path / "system.txt").write_text("Judge both answers.")
    (tmp_path / "user.txt").write_text(
        "{user_prompt}\nA: {completion_A}\nB: {completion_B}"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "task": "elo-comparia",
                "model": {"name": "Dummy/candidate", "max_out_tokens": 64},
                "judge": {
                    "model": "Dummy/score A: 5 score B: 5",
                    "prompt": {
                        "system_file": "system.txt",
                        "user_file": "user.txt",
                        "parser": "score",
                    },
                },
                "generation": {"n_instructions": 3},
                "elo": {
                    "baseline_model": "reference",
                    "n_bootstraps": 0,
                    "languages": ["en"],
                    "calibrate_temperature": calibrate,
                    "calibration_size": 10 if calibrate else None,
                },
                "run": {"seed": 7, "store_root": None},
            }
        )
    )
    return config


def _battles(count: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question_id": f"q{index}",
                "lang": "en",
                "model_a": "reference" if index % 2 else "strong",
                "model_b": "strong" if index % 2 else "reference",
                "winner": "model_a" if index % 4 < 2 else "model_b",
                "conversation_a": _conversation(f"prompt {index}", f"a {index}"),
                "conversation_b": _conversation(f"prompt {index}", f"b {index}"),
            }
            for index in range(count)
        ]
    )


def test_freeze_writes_exact_deterministic_balanced_panel(monkeypatch, tmp_path):
    config = _write_config(tmp_path)
    monkeypatch.setattr(freeze, "load_battles", lambda _task: _battles())

    def fail_calibration(*_args, **_kwargs):
        raise AssertionError("fixed beta must not build a calibration judge")

    monkeypatch.setattr(calibration, "build_judge", fail_calibration)

    first = freeze.freeze_leaderboard(
        config,
        tmp_path / "first",
        ["strong", "reference"],
        ["en"],
        6,
        name="test",
        version="0.02",
        min_anchor_battles=8,
    )
    second = freeze.freeze_leaderboard(
        config,
        tmp_path / "second",
        ["strong", "reference"],
        ["en"],
        6,
        name="test",
        version="0.02",
        min_anchor_battles=8,
    )

    panel = pd.read_parquet(first / "panel.parquet")
    assert len(panel) == panel["question_id"].nunique() == 6
    assert set(panel["candidate_position"]) == {"A", "B"}
    assert panel["candidate_position"].value_counts().nunique() == 1
    opponent_counts = panel["opponent_model"].value_counts()
    assert opponent_counts.max() - opponent_counts.min() <= 2
    pd.testing.assert_frame_equal(panel, pd.read_parquet(second / "panel.parquet"))

    anchors = AnchorSet.load(first / "anchors.json")
    assert anchors.ratings_by_language["en"]["reference"] == 1000
    assert set(anchors.ratings_by_language["en"]) == {"reference", "strong"}
    assert anchors.human_battles_by_language == {"en": 8}
    assert anchors.battles_per_language == 6
    assert anchors.version == "0.02"
    assert anchors.min_anchor_battles == 8
    assert anchors.counts_by_language == {"en": {"reference": 8, "strong": 8}}
    task = freeze.get_packaged_task("elo-comparia")
    assert anchors.dataset_sources == {
        name: source.model_dump(mode="json")
        for name, source in task.spec.dataset.sources.items()
    }
    load_frozen_files(first)
    assert anchors.protocol_id == AnchorSet.load(second / "anchors.json").protocol_id
    saved = load_config(first / "config.yaml")
    assert saved.generation.n_instructions is None
    assert saved.elo.n_instructions_per_language is None
    assert saved.elo.elo_random_battles is None
    assert saved.elo.leaderboard_dir is None
    runtime = load_config(first / "config.yaml")
    runtime.model.name = "Dummy/new-submission"
    runtime.elo.leaderboard_dir = first
    assert comparable_config(runtime) == comparable_config(saved)
    system_prompt = first / "judge-system-prompt.txt"
    assert system_prompt.read_text() == "Judge both answers."
    assert "{completion_A}" in (first / "judge-user-prompt.txt").read_text()
    system_prompt.write_text("changed")
    with pytest.raises(ValueError, match="protocol ID"):
        load_frozen_files(first)
    system_prompt.write_text("Judge both answers.")
    assert (first / "entries").is_dir()
    leaderboard = json.loads((first / "leaderboard.json").read_text())
    assert {entry["source"] for entry in leaderboard["entries"]} == {"anchor"}
    assert leaderboard["name"] == "test"
    assert leaderboard["version"] == "0.02"


def test_freeze_calibrates_beta_once_without_mutating_config(monkeypatch, tmp_path):
    config = load_config(_write_config(tmp_path, calibrate=True))
    original = config.model_copy(deep=True)
    monkeypatch.setattr(freeze, "load_battles", lambda _task: _battles(24))
    calibrated = {}
    judge = calibration.judge_and_parse_prefs

    def capture_judging(**kwargs):
        calibrated.update(kwargs)
        return judge(**kwargs)

    monkeypatch.setattr(calibration, "judge_and_parse_prefs", capture_judging)
    monkeypatch.setattr(calibration, "fit_temperature", lambda *_args: 0.75)
    output = freeze.freeze_leaderboard(
        config,
        tmp_path / "calibrated",
        ["strong"],
        ["en"],
        6,
    )

    saved = load_config(output / "config.yaml")
    assert saved.elo.soft_elo_temperature == 0.75
    assert saved.elo.calibrate_temperature is False
    assert saved.elo.calibration_size is None
    assert config == original
    assert len(calibrated["instructions"]) == 10
    expected = _battles(24).sample(n=10, random_state=2029167941)
    assert calibrated["instructions"] == [
        row[0]["content"] for row in expected["conversation_a"]
    ]
    assert calibrated["cache_row_metadata"] == [
        {
            "instruction_id": f"ComparIA:{row.question_id}",
            "model_a": row.model_a,
            "model_b": row.model_b,
            "orientation": "direct",
        }
        for row in expected.itertuples()
    ]


def test_freeze_rejects_nonpositive_calibrated_beta(monkeypatch, tmp_path):
    config = _write_config(tmp_path, calibrate=True)
    monkeypatch.setattr(freeze, "load_battles", lambda _task: _battles(24))
    monkeypatch.setattr(calibration, "fit_temperature", lambda *_args, **_kwargs: 0.0)

    with pytest.raises(ValueError, match="finite positive soft-Elo beta"):
        freeze.freeze_leaderboard(
            config,
            tmp_path / "invalid-calibration",
            ["strong"],
            ["en"],
            6,
        )


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("existing", FileExistsError, "already exists"),
        ("disconnected", ValueError, "not connected"),
        ("count", ValueError, "at least 9 usable human battles"),
        ("panel", ValueError, "distinct anchor questions"),
        ("minimum", ValueError, "min_anchor_battles must be positive"),
        ("version", ValueError, "version must be non-empty"),
    ],
)
def test_freeze_preflight_precedes_calibration(
    monkeypatch, tmp_path, case, error, message
):
    config = _write_config(tmp_path, calibrate=True)
    monkeypatch.setattr(freeze, "load_battles", lambda _task: _battles())

    def fail_calibration(*_args, **_kwargs):
        pytest.fail("Invalid freezes must not reach calibration")

    monkeypatch.setattr(freeze, "calibrate_frozen_temperature", fail_calibration)
    output = tmp_path / "frozen"
    if case == "existing":
        output.mkdir()
    with pytest.raises(error, match=message):
        freeze.freeze_leaderboard(
            config,
            output,
            ["missing" if case == "disconnected" else "strong"],
            ["en"],
            9 if case == "panel" else 6,
            min_anchor_battles=0 if case == "minimum" else 9 if case == "count" else 1,
            version=" " if case == "version" else "0.01",
        )
    assert not output.exists() or case == "existing"


def test_anchor_counts_exclude_self_battles_and_invalid_human_labels():
    usable = _battles(4)
    unusable = pd.concat(
        [
            _battles().assign(model_a="strong", model_b="strong"),
            _battles().assign(winner="unknown", model_a=None, model_b=""),
        ],
        ignore_index=True,
    )
    battles = pd.concat([usable, unusable], ignore_index=True)
    language_battles, ratings, counts, human_battles = freeze._fit_language_anchors(
        battles, "en", ["reference", "strong"], "reference", 4
    )
    assert len(language_battles) == human_battles == 4
    assert counts == {"reference": 4, "strong": 4}
    assert (
        ratings
        == freeze._fit_language_anchors(
            usable, "en", ["reference", "strong"], "reference", 4
        )[1]
    )
    with pytest.raises(ValueError, match="at least 5 usable human battles"):
        freeze._fit_language_anchors(
            battles, "en", ["reference", "strong"], "reference", 5
        )


@pytest.mark.parametrize("column", ["model_a", "model_b"])
@pytest.mark.parametrize("model", [None, "", " \t", 17])
def test_anchor_fit_rejects_malformed_model_names(column, model):
    battles = _battles(4)
    battles[column] = battles[column].astype(object)
    battles.loc[0, column] = model

    with pytest.raises(
        ValueError, match=f"{column} must contain non-empty model names"
    ):
        freeze._fit_language_anchors(
            battles, "en", ["reference", "strong"], "reference", 4
        )


def test_anchor_fit_retains_connected_non_anchor_battles():
    battles = pd.concat(
        [
            _battles(4).assign(model_a="reference", model_b="bridge"),
            _battles(6).assign(model_a="bridge", model_b="strong"),
            _battles(8).assign(model_a="bridge", model_b="other"),
            _battles(2).assign(model_a="disconnected", model_b="island"),
        ],
        ignore_index=True,
    )
    _, ratings, counts, human_battles = freeze._fit_language_anchors(
        battles, "en", ["reference", "strong"], "reference", 4
    )
    connected = battles.iloc[:18].copy()
    connected["pref"] = connected["winner"].map(freeze.winner_to_pref)
    expected = freeze.fit_bradley_terry(connected, baseline_model="reference")
    assert ratings == {model: expected[model] for model in ("reference", "strong")}
    assert counts == {"reference": 4, "strong": 6}
    assert human_battles == 18


def test_minimum_anchor_battles_is_per_language(monkeypatch, tmp_path):
    config = _write_config(tmp_path, calibrate=True)
    battles = pd.concat([_battles(8), _battles(4).assign(lang="fr")])
    monkeypatch.setattr(freeze, "load_battles", lambda _task: battles)

    def fail_calibration(*_args, **_kwargs):
        pytest.fail("Every language must pass the minimum before calibration")

    monkeypatch.setattr(freeze, "calibrate_frozen_temperature", fail_calibration)
    with pytest.raises(ValueError, match="Language 'fr': anchors need at least 5"):
        freeze.freeze_leaderboard(
            config,
            tmp_path / "frozen",
            ["strong"],
            ["en", "fr"],
            4,
            min_anchor_battles=5,
        )
