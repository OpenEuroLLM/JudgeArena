"""Focused integration tests for the meta-evaluation task runner."""

from __future__ import annotations

import json
from copy import deepcopy
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
    return pd.DataFrame(
        [(f"q{i}", *pair) for i, pair in enumerate(pairs)],
        columns=["question_id", "model_a", "model_b", "winner", "lang"],
    ).assign(conversation_a=None, conversation_b=None)


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


@pytest.mark.parametrize("battles_per_model", [2, 12], ids=["unavailable", "available"])
def test_meta_eval_scores_renders_and_saves(
    tmp_path, monkeypatch, capsys, battles_per_model
):
    task = deepcopy(get_packaged_task("meta-eval-comparia"))
    for request in task.spec.protocol.scoring.metrics:
        if "n_bootstraps" in request.parameters:
            request.parameters["n_bootstraps"] = 2
        else:
            request.parameters.update(battle_counts=[4, 8], n_seeds=2)
    monkeypatch.setattr(runner_module, "load_battles", lambda _task: _arena())
    monkeypatch.setattr(runner_module, "build_judge", lambda _cfg: object())

    def fake_annotate(sample, *_args, **_kwargs):
        rows = sample[["battle_id", "reference_pref"]].rename(
            columns={"reference_pref": "pref"}
        )
        rows.loc[rows.index[0], "pref"] = np.nan
        return rows.assign(orientation="single")

    monkeypatch.setattr(runner_module, "annotate_sample", fake_annotate)
    cfg = _config(tmp_path, battles_per_model=battles_per_model)
    result = runner_module.run_meta_eval(cfg, task)

    result_path = Path(result["result_path"])
    sample = pd.read_parquet(result_path.parent / "sample.parquet")
    annotations = pd.read_parquet(result_path.parent / "annotations.parquet")
    battles = pd.read_parquet(result_path.parent / "battles.parquet")
    assert battles["battle_id"].is_unique
    assert set(battles["battle_id"]) == set("ComparIA:" + _arena()["question_id"])
    assert set(sample["battle_id"]) == set(annotations["battle_id"])
    sampled = battles.loc[battles["sampled"]].set_index("battle_id")
    pd.testing.assert_series_equal(
        sampled["pref"].sort_index(),
        annotations.set_index("battle_id")["pref"].sort_index(),
    )
    assert 0 < len(sampled) == len(sample) < len(battles)
    assert sampled["pref"].isna().sum() == 1
    assert battles.loc[~battles["sampled"], "pref"].isna().all()
    assert set(battles["reference_pref"]) == {0.0, 0.5, 1.0}

    saved = json.loads(result_path.read_text())
    agreement = saved["metrics"]["meta_eval_agreement"]["all"]
    assert agreement["n_attempted"] == len(sample)
    assert agreement["n_complete"] == len(sample) - 1
    assert agreement["accuracy_complete"] == 1
    assert agreement["accuracy_attempted"] == pytest.approx(1 - 1 / len(sample))
    ranking = saved["metrics"]["meta_eval_ranking"]
    complete_non_ties = sampled["pref"].notna() & sampled["reference_pref"].ne(0.5)
    assert ranking["n_battles"] == complete_non_ties.sum()
    output = capsys.readouterr().out
    assert f"complete {len(sample) - 1}/{len(sample)}" in output
    elo_gap = saved["metrics"]["meta_eval_elo_gap"]
    if battles_per_model == 2:
        assert elo_gap == {}
        assert "meta_eval_elo_gap: unavailable" in output
    else:
        for method in ("hard", "soft"):
            assert ranking[method]["elo_mae"] == pytest.approx(0)
            for row, budget in zip(elo_gap[method], [4, 8], strict=True):
                assert row["attempted_battles_per_model"] == budget
                assert np.isfinite(row["mean_gap"])
                assert row["n_seeds_valid"] == 2
        assert "8 attempted battles/focal" in output
    metadata = json.loads((result_path.parent / "run-metadata.v1.json").read_text())
    assert metadata["results"] == saved
    assert metadata["dataset_statistics"]["battle_id_count"] == len(sample)
    assert metadata["run"]["meta_eval"]["battles_per_model"] == battles_per_model


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
