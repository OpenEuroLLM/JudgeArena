"""Focused integration tests for the meta-evaluation task runner."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

import judgearena.benchmarks.meta_eval.runner as runner_module
from judgearena.config import RunConfig
from judgearena.datasets.data import PreferenceData
from judgearena.tasks.registry import get_packaged_task


def _arena() -> pd.DataFrame:
    pairs = [
        ("a", "b", "model_a", "en"),
        ("b", "c", "model_b", "fr"),
        ("a", "c", "tie", "en"),
        ("a", "b", "model_b", "fr"),
        ("b", "c", "tie", "en"),
        ("a", "c", "model_a", "fr"),
    ] * 20
    frame = pd.DataFrame(
        [(f"q{i}", *pair) for i, pair in enumerate(pairs)],
        columns=["question_id", "model_a", "model_b", "winner", "lang"],
    )
    return frame.assign(
        instruction_index=frame.index,
        instruction=frame["question_id"],
        completion_a="answer a",
        completion_b="answer b",
        source="human",
        pref=frame["winner"].map({"model_a": 0.0, "model_b": 1.0, "tie": 0.5}),
    )


def _config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        task="meta-eval-comparia",
        judge={"model": "Dummy/judge", "swap_mode": "fixed"},
        meta_eval={"top_models": 3, "battles_per_model": 12},
        run={"result_folder": str(tmp_path), "no_log_file": True, "seed": 7},
    )


def test_prepare_arena_battles_drops_self_comparisons():
    arena = _arena().iloc[:2].copy()
    arena.loc[0, "model_b"] = arena.loc[0, "model_a"]
    prepared = runner_module._prepare_arena_battles(
        PreferenceData.from_battles(arena),
        task="meta-eval-test",
        arena="TestArena",
    )
    assert prepared["battle_id"].tolist() == ["TestArena:q1"]


def test_meta_eval_scores_renders_and_saves(tmp_path, monkeypatch, capsys):
    task = deepcopy(get_packaged_task("meta-eval-comparia"))
    for request in task.spec.protocol.scoring.metrics:
        if "n_bootstraps" in request.parameters:
            request.parameters["n_bootstraps"] = 2
        else:
            request.parameters.update(battle_counts=[4], n_seeds=2)
    arena = _arena()
    for side in ("a", "b"):
        arena[f"conversation_{side}"] = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"answer {side}"},
            ]
            if lang == "en"
            else []
            for question, lang in zip(arena["question_id"], arena["lang"], strict=True)
        ]
    data = PreferenceData.from_battles(arena.drop(columns=["winner", "question_id"]))
    monkeypatch.setattr(runner_module, "load_task_data", lambda _task: data)
    monkeypatch.setattr(
        runner_module,
        "build_judge",
        lambda _cfg: FakeListLLM(
            responses=["unparseable"] + ["score_A: 9\nscore_B: 1"] * 120
        ),
    )
    cfg = _config(tmp_path)
    cfg.meta_eval.languages = ["en"]
    result = runner_module.run_meta_eval(cfg, task)

    result_path = Path(result["result_path"])
    sample = pd.read_parquet(result_path.parent / "sample.parquet")
    annotations = pd.read_parquet(result_path.parent / "annotations.parquet")
    battles = pd.read_parquet(result_path.parent / "battles.parquet")
    assert set(sample["battle_id"]) == set(annotations["battle_id"])
    assert sample["lang"].eq("en").all()
    assert "winner" not in sample
    assert sample["reference_pref"].equals(sample["pref"])
    assert sample["completion_a"].eq("answer a").all()
    sampled = battles.loc[battles["sampled"]].set_index("battle_id")
    pd.testing.assert_series_equal(
        sampled["pref"].sort_index(),
        annotations.set_index("battle_id")["pref"].sort_index(),
    )
    assert 0 < len(sampled) == len(sample) < len(battles)
    assert battles.loc[~battles["sampled"], "pref"].isna().all()

    saved = json.loads(result_path.read_text())
    agreement = saved["metrics"]["meta_eval_agreement"]["all"]
    correct = sample.iloc[1:]["pref"].eq(0.0).sum()
    assert agreement["accuracy_attempted"] == pytest.approx(correct / len(sample))
    assert saved["metrics"]["meta_eval_elo_gap"]["soft"][0]["n_seeds_valid"] == 2
    assert f"complete {len(sample) - 1}/{len(sample)}" in capsys.readouterr().out
    metadata = json.loads((result_path.parent / "run-metadata.v1.json").read_text())
    assert metadata["dataset_statistics"]["battle_id_count"] == len(sample)


@pytest.mark.parametrize(
    "column, value, message",
    [
        ("source", "llm-judge", "human labels"),
        ("pref", 0.25, "human preferences"),
        ("pref", None, "human preferences"),
    ],
)
def test_meta_eval_rejects_invalid_human_labels_before_judge_build(
    tmp_path, monkeypatch, column, value, message
):
    task = get_packaged_task("meta-eval-comparia")
    arena = _arena()
    arena.loc[0, column] = value
    data = PreferenceData.from_battles(arena)
    monkeypatch.setattr(runner_module, "load_task_data", lambda _task: data)
    monkeypatch.setattr(
        runner_module,
        "build_judge",
        lambda _cfg: pytest.fail("judge must not be built before validation"),
    )

    with pytest.raises(ValueError, match=message):
        runner_module.run_meta_eval(_config(tmp_path), task)
