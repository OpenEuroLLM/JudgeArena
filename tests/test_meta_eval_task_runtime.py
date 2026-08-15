from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.annotate as annotate_module
import judgearena.benchmarks.meta_eval.runner as runner_module
from judgearena.benchmarks.meta_eval.agreement import compute_agreement_metrics
from judgearena.benchmarks.meta_eval.annotate import (
    invert_winner,
    serialize_judge_input,
)
from judgearena.benchmarks.meta_eval.parsers import (
    parse_pairscore_pref,
    parse_pairscore_winner,
    parse_pref,
    parse_winner,
)
from judgearena.benchmarks.meta_eval.runner import (
    ANNOTATIONS_FILENAME,
    SAMPLE_FILENAME,
    SUMMARY_FILENAME,
    run_meta_eval,
)
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.benchmarks.meta_eval.scoring import summarize_language_splits
from judgearena.benchmarks.registry import resolve_benchmark
from judgearena.config import RunConfig
from judgearena.evaluate import JudgeAnnotation
from judgearena.prompts.registry import resolve_judge_prompt
from judgearena.tasks.registry import get_packaged_task


def _turns(answer: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": answer},
    ]


def _battles() -> pd.DataFrame:
    models = ["m1", "m2", "m3"]
    rows = [
        {
            "question_id": f"q{i}",
            "model_a": models[i % 3],
            "model_b": models[(i + 1) % 3],
            "winner": "tie (bothbad)" if i % 5 == 0 else "model_a",
            "lang": "fr" if i % 2 else "en",
            "conversation_a": _turns("a"),
            "conversation_b": _turns("b"),
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
            "conversation_a": _turns("a"),
            "conversation_b": _turns("b"),
        }
    )
    return pd.DataFrame(rows)


def _cfg(tmp_path, **overrides) -> RunConfig:
    values = {
        "task": "meta-eval-comparia",
        "judge": {"model": "Dummy/j"},
        "run": {"result_folder": str(tmp_path), "no_log_file": True},
    }
    values.update(overrides)
    return RunConfig(**values)


def _fake_annotations(**kwargs):
    return [
        JudgeAnnotation(
            instruction=instruction,
            completion_A=completion_a,
            completion_B=completion_b,
            judge_completion="score_A: 9\nscore_B: 1",
            judge_input=SimpleNamespace(to_string=lambda: "prompt"),
        )
        for instruction, completion_a, completion_b in zip(
            kwargs["instructions"],
            kwargs["completions_A"],
            kwargs["completions_B"],
            strict=True,
        )
    ]


def test_meta_eval_task_is_registered_with_the_arena_battle_stack():
    task = get_packaged_task("meta-eval-lmarena-140k")
    assert task is not None
    assert task.spec.protocol.runner == "meta_eval"
    assert task.spec.protocol.arena == "LMArena-140k"
    assert task.spec.dataset.adapter == "arena_battles"
    assert task.spec.protocol.scoring.adapter == "ranking"
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


def test_pairscore_parses_winners_at_meta_eval_temperature():
    assert parse_pairscore_winner("score_A: 10\nscore_B: 0") == "model_a"
    assert parse_pairscore_winner("score_A: 0\nscore_B: 10") == "model_b"
    assert parse_pairscore_winner("score_A: 5\nscore_B: 5") == "tie"
    completion = "score_A: 10\nscore_B: 0"
    preference = parse_pairscore_pref(completion)
    assert preference < 0.5
    assert parse_pairscore_winner(completion, eps=abs(preference - 0.5)) == "tie"
    assert serialize_judge_input(SimpleNamespace(to_string=lambda: "p")) == "p"
    assert invert_winner("model_a") == "model_b"
    assert invert_winner("tie") == "tie"


def test_prompt_presets_select_their_parsers():
    assert parse_winner("score_A: 9\nscore_B: 1", "default") == "model_a"
    assert parse_winner("[[A=B]]", "arena-hard") == "tie"
    assert parse_winner("[[B>>A]]", "arena-hard") == "model_b"
    assert (
        parse_winner(
            '{"ordered_models": [{"model": "m", "rank": 1}, {"model": "M", "rank": 2}]}',
            "alpaca-eval",
        )
        == "model_a"
    )
    assert parse_pref("[[A=B]]", "arena-hard") == 0.5
    for preset in ("arena-hard", "alpaca-eval", "alpaca-eval-pair-score"):
        resolved = resolve_judge_prompt(preset=preset)
        assert resolved.system_prompt
        assert resolved.user_prompt_template


def test_agreement_metrics_on_fixture():
    metrics = compute_agreement_metrics(
        ["model_a", "model_b", "tie", "model_a"],
        ["model_a", "model_a", "tie", "model_b"],
        n_bootstraps=10,
        seed=0,
    )
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["n_nt"] == 3


def test_degenerate_agreement_reports_nan_kappa():
    metrics = compute_agreement_metrics(
        ["model_a", "model_a"],
        ["model_a", "model_a"],
        n_bootstraps=4,
        seed=0,
    )
    assert metrics["accuracy"] == 1.0
    assert pd.isna(metrics["kappa"])


def test_runner_writes_annotations_and_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "load_battles", lambda _task: _battles())
    monkeypatch.setattr(runner_module, "build_judge", lambda _cfg: object())
    monkeypatch.setattr(annotate_module, "annotate_battles", _fake_annotations)
    cfg = _cfg(tmp_path, meta_eval={"top_models": 3, "battles_per_model": 5})

    results = run_meta_eval(cfg, get_packaged_task("meta-eval-comparia"))

    res_dir = tmp_path / "meta-eval-comparia-dummy-j"
    sample = pd.read_parquet(res_dir / SAMPLE_FILENAME)
    annotations = pd.read_parquet(res_dir / ANNOTATIONS_FILENAME)
    assert results["n_battles"] == results["n_annotations"] == len(sample) == 15
    assert len(annotations) == 15
    assert set(annotations["orientation"]) == {"forward"}
    assert set(annotations["winner_llm"]) == {"model_a"}
    assert annotations["judge_input"].tolist() == ["prompt"] * 15
    assert results["agreement"]["all"]["n"] == 15
    assert results["agreement"]["no_human_ties"]["n"] < 15
    assert set(results["language_summary"]) == {"English", "Multilingual"}
    summary = pd.read_csv(res_dir / SUMMARY_FILENAME)
    assert set(summary["split"]) == {"English", "Multilingual"}
    assert json.loads((res_dir / "results.json").read_text())["arena"] == "ComparIA"


def test_swap_mode_both_inverts_the_reversed_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "load_battles", lambda _task: _battles())
    monkeypatch.setattr(runner_module, "build_judge", lambda _cfg: object())
    monkeypatch.setattr(annotate_module, "annotate_battles", _fake_annotations)
    cfg = _cfg(
        tmp_path,
        judge={"model": "Dummy/j", "swap_mode": "both"},
        meta_eval={"top_models": 3, "battles_per_model": 5},
    )

    results = run_meta_eval(cfg, get_packaged_task("meta-eval-comparia"))

    annotations = pd.read_parquet(
        tmp_path / "meta-eval-comparia-dummy-j" / ANNOTATIONS_FILENAME
    )
    assert results["n_battles"] == 15
    assert results["n_annotations"] == results["agreement"]["all"]["n"] == 30
    assert set(annotations["orientation"]) == {"forward", "swapped"}
    forward = annotations[annotations["orientation"] == "forward"]
    swapped = annotations[annotations["orientation"] == "swapped"]
    assert set(forward["winner_llm"]) == {"model_a"}
    assert set(swapped["winner_llm"]) == {"model_b"}
    assert forward["model_a"].tolist() == swapped["model_a"].tolist()
    assert (
        forward["presented_model_a"].tolist() == swapped["presented_model_b"].tolist()
    )
    ranking_n = sum(split["n"] for split in results["language_summary"].values())
    assert ranking_n == int((forward["winner"] != "tie").sum())
    assert ranking_n < results["n_annotations"]


def test_empty_language_split_reports_na():
    df = pd.DataFrame(
        {
            "lang": ["fr", "fr"],
            "model_a": ["a", "a"],
            "model_b": ["b", "b"],
            "winner": ["model_a", "model_b"],
            "winner_llm": ["model_a", "model_a"],
            "pref_llm": [0.1, 0.2],
            "orientation": ["forward", "forward"],
        }
    )
    summary = summarize_language_splits(
        df, exclude_human_ties=True, n_bootstraps=4, seed=0
    )
    assert summary["English"] == {
        "n": 0,
        "kappa": "n/a",
        "spearman": "n/a",
        "spearman_soft": "n/a",
        "mae_elo": "n/a",
        "mae_soft_elo": "n/a",
    }
    assert summary["Multilingual"]["n"] == 2
