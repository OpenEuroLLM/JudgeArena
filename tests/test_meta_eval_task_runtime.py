from __future__ import annotations

import json

import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.runner as runner_module
from judgearena.benchmarks.meta_eval.runner import SAMPLE_FILENAME, run_meta_eval
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.benchmarks.registry import resolve_benchmark
from judgearena.config import RunConfig
from judgearena.tasks.registry import get_packaged_task


def _battles() -> pd.DataFrame:
    models = ["m1", "m2", "m3"]
    rows = [
        {
            "question_id": f"q{i}",
            "model_a": models[i % 3],
            "model_b": models[(i + 1) % 3],
            "winner": "tie (bothbad)" if i % 5 == 0 else "model_a",
            "lang": "fr" if i % 2 else "en",
        }
        for i in range(30)
    ]
    rows.append(
        {
            "question_id": "q-rare",
            "model_a": "rare",
            "model_b": "m1",
            "winner": "model_b",
            "lang": "en",
        }
    )
    return pd.DataFrame(rows)


def _cfg(tmp_path, **overrides) -> RunConfig:
    return RunConfig(
        task="meta-eval-comparia",
        judge={"model": "Dummy/j"},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        **overrides,
    )


def test_meta_eval_task_is_registered_with_the_arena_battle_stack():
    task = get_packaged_task("meta-eval-lmarena-140k")
    assert task is not None
    assert task.spec.protocol.runner == "meta_eval"
    assert task.spec.protocol.arena == "LMArena-140k"
    assert task.spec.dataset.adapter == "arena_battles"
    assert resolve_benchmark("meta-eval-lmarena-140k").adapter.name == "meta_eval"


def test_meta_eval_language_variant_selects_one_language():
    variant = get_packaged_task("meta-eval-comparia-fr")
    assert variant is not None
    assert variant.selection is not None
    assert variant.selection.values == ("fr",)


def test_sampling_is_deterministic_and_stays_inside_the_top_models():
    battles = _battles()
    top, df_top = select_top_models(battles, top_models=3)
    assert set(top) == {"m1", "m2", "m3"}
    assert "q-rare" not in set(df_top["question_id"])

    sample = sample_battles_per_model(df_top, top, battles_per_model=5, seed=0)
    again = sample_battles_per_model(df_top, top, battles_per_model=5, seed=0)

    assert len(sample) == 15
    pd.testing.assert_frame_equal(sample, again)


def test_sampling_rejects_a_disconnected_top_model_set():
    battles = pd.DataFrame(
        [{"question_id": "q", "model_a": "a", "model_b": "b", "winner": "tie"}]
    )
    with pytest.raises(MetaEvalSamplingError, match="No battles remain"):
        select_top_models(battles, top_models=1)


def test_runner_writes_a_reproducible_sample_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "load_battles", lambda _task: _battles())
    cfg = _cfg(tmp_path, meta_eval={"top_models": 3, "battles_per_model": 5})

    results = run_meta_eval(cfg, get_packaged_task("meta-eval-comparia"))

    res_dir = tmp_path / "meta-eval-comparia-dummy-j"
    sample = pd.read_parquet(res_dir / SAMPLE_FILENAME)
    assert results["n_battles"] == len(sample) == 15
    assert set(sample["winner"]) <= {"model_a", "model_b", "tie"}
    assert sorted(sample.columns) == [
        "lang",
        "model_a",
        "model_b",
        "question_id",
        "winner",
    ]
    assert json.loads((res_dir / "results.json").read_text())["arena"] == "ComparIA"
