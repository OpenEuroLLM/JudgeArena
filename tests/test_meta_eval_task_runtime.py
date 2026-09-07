"""Focused integration tests for the meta-evaluation task runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.runner as runner_module
from judgearena.config import RunConfig
from judgearena.tasks.registry import get_packaged_task


def _arena() -> pd.DataFrame:
    pairs = [
        ("a", "b", "model_a", "en"),
        ("b", "c", "model_b", "fr"),
        ("a", "c", "tie", "en"),
        ("a", "b", "model_b", "fr"),
        ("b", "c", "tie (bothbad)", "en"),
        ("a", "c", "model_a", "fr"),
    ] * 20
    rows = []
    for index, (model_a, model_b, winner, language) in enumerate(pairs):
        prompt = f"prompt {index}"
        rows.append(
            {
                "question_id": f"q{index}",
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner,
                "lang": language,
                "conversation_a": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"{model_a} answer"},
                ],
                "conversation_b": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"{model_b} answer"},
                ],
            }
        )
    return pd.DataFrame(rows)


def _config(tmp_path: Path, *, battles_per_model: int = 50) -> RunConfig:
    return RunConfig(
        task="meta-eval-comparia",
        judge={"model": "Dummy/judge", "swap_mode": "fixed"},
        meta_eval={"top_models": 3, "battles_per_model": battles_per_model},
        run={"result_folder": str(tmp_path), "no_log_file": True, "seed": 7},
    )


def test_prepare_arena_battles_drops_self_comparisons():
    arena = _arena()
    self_comparison = arena.iloc[[0]].copy()
    self_comparison["question_id"] = "self-comparison"
    self_comparison["model_b"] = self_comparison["model_a"]
    source = pd.concat([arena, self_comparison], ignore_index=True)

    prepared = runner_module._prepare_arena_battles(
        source, task="meta-eval-test", arena="TestArena", languages=[]
    )

    assert len(prepared) == len(arena)
    assert not (prepared["model_a"] == prepared["model_b"]).any()


def test_meta_eval_runner_builds_full_metric_table_and_artifacts(tmp_path, monkeypatch):
    task = get_packaged_task("meta-eval-comparia")
    captured = {}
    judge = object()

    monkeypatch.setattr(runner_module, "load_battles", lambda _task: _arena())
    monkeypatch.setattr(runner_module, "build_judge", lambda _cfg: judge)

    def fake_annotate(sample, _cfg, *, judge_chat_model, resolved_prompt):
        assert judge_chat_model is judge
        assert resolved_prompt.delegated is False
        captured["sample_ids"] = set(sample["battle_id"])
        return sample[["battle_id"]].assign(orientation="single", pref=0.25)

    monkeypatch.setattr(runner_module, "annotate_sample", fake_annotate)

    def fake_calculate(metric_battles, metrics, *, runtime_by_metric):
        captured["battles"] = metric_battles.copy()
        captured["runtime"] = runtime_by_metric
        return {request.metric: {"ok": True} for request, _metric in metrics}

    monkeypatch.setattr(runner_module, "calculate_metrics", fake_calculate)
    monkeypatch.setattr(runner_module.MetaEvalReport, "render", lambda self: None)

    result = runner_module.run_meta_eval(_config(tmp_path), task)

    metric_battles = captured["battles"]
    assert len(metric_battles) == len(_arena())
    assert metric_battles["battle_id"].is_unique
    assert set(metric_battles["reference_pref"]) == {0.0, 0.5, 1.0}
    sampled = metric_battles.loc[metric_battles["sampled"]]
    assert set(sampled["battle_id"]) == captured["sample_ids"]
    assert set(sampled["pref"]) == {0.25}
    assert (~metric_battles["sampled"]).any()
    assert metric_battles.loc[~metric_battles["sampled"], "pref"].isna().all()
    assert set(captured["runtime"]) == {
        "meta_eval_agreement",
        "meta_eval_ranking",
        "meta_eval_elo_gap",
    }
    assert all(
        isinstance(runtime["rng"], np.random.Generator)
        for runtime in captured["runtime"].values()
    )

    result_path = Path(result["result_path"])
    assert {"sample.parquet", "annotations.parquet", "battles.parquet"}.issubset(
        path.name for path in result_path.parent.iterdir()
    )
    saved_battles = pd.read_parquet(result_path.parent / "battles.parquet")
    pd.testing.assert_frame_equal(saved_battles, metric_battles, check_like=True)
    saved = json.loads(result_path.read_text())
    assert saved["metrics"] == result["metrics"]
    assert "agreement" not in saved


def test_meta_eval_elo_gap_budget_is_checked_before_data_loading(tmp_path, monkeypatch):
    task = get_packaged_task("meta-eval-comparia")
    monkeypatch.setattr(
        runner_module,
        "load_battles",
        lambda _task: pytest.fail("data must not load before static preflight"),
    )

    with pytest.raises(ValueError, match="exceeds meta_eval.battles_per_model"):
        runner_module.run_meta_eval(_config(tmp_path, battles_per_model=49), task)


def test_meta_eval_rejects_unknown_human_winner_before_judge_build(
    tmp_path, monkeypatch
):
    task = get_packaged_task("meta-eval-comparia")
    arena = _arena()
    arena.loc[0, "winner"] = "unknown"
    monkeypatch.setattr(runner_module, "load_battles", lambda _task: arena)
    monkeypatch.setattr(
        runner_module,
        "build_judge",
        lambda _cfg: pytest.fail("judge must not be built before validation"),
    )

    with pytest.raises(ValueError, match="invalid human winners.*unknown"):
        runner_module.run_meta_eval(_config(tmp_path), task)
