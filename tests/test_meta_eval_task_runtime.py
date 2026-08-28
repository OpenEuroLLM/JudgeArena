from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.annotate as annotate_module
import judgearena.benchmarks.meta_eval.runner as runner_module
import judgearena.evaluate as evaluate_module
from judgearena.benchmarks.meta_eval.agreement import compute_agreement_metrics
from judgearena.benchmarks.meta_eval.annotate import (
    invert_winner,
    preference_to_winner,
    serialize_judge_input,
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
from judgearena.benchmarks.meta_eval.scoring import (
    compute_elo_gap_summary,
    summarize_language_splits,
)
from judgearena.benchmarks.registry import resolve_benchmark
from judgearena.config import RunConfig
from judgearena.evaluate import JudgeAnnotation
from judgearena.prompts.parsing import JUDGE_PARSERS, parser_name
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


@pytest.mark.parametrize(
    ("task_name", "arena"),
    [
        ("meta-eval-lmarena-100k", "LMArena-100k"),
        ("meta-eval-lmarena-140k", "LMArena-140k"),
    ],
)
def test_meta_eval_task_is_registered_with_the_arena_battle_stack(task_name, arena):
    task = get_packaged_task(task_name)
    assert task is not None
    assert task.spec.protocol.runner == "meta_eval"
    assert task.spec.protocol.arena == arena
    assert task.spec.dataset.adapter == "arena_battles"
    assert task.spec.protocol.scoring.adapter == "ranking"
    assert resolve_benchmark(task_name).adapter.name == "meta_eval"


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
    parser = JUDGE_PARSERS["meta-eval-score"]
    assert preference_to_winner(parser("score_A: 10\nscore_B: 0")) == "model_a"
    assert preference_to_winner(parser("score_A: 0\nscore_B: 10")) == "model_b"
    assert preference_to_winner(parser("score_A: 5\nscore_B: 5")) == "tie"
    completion = "score_A: 10\nscore_B: 0"
    preference = parser(completion)
    assert preference is not None
    assert preference < 0.5
    assert preference_to_winner(0.495) == "tie"
    assert serialize_judge_input(SimpleNamespace(to_string=lambda: "p")) == "p"
    assert invert_winner("model_a") == "model_b"
    assert invert_winner("tie") == "tie"


def test_prompt_presets_select_their_parsers():
    assert preference_to_winner(JUDGE_PARSERS["arena-hard-verdict"]("[[A=B]]")) == "tie"
    assert (
        preference_to_winner(JUDGE_PARSERS["arena-hard-verdict"]("[[B>>A]]"))
        == "model_b"
    )
    assert (
        preference_to_winner(JUDGE_PARSERS["arena-hard-verdict"]("[[B<<A]]"))
        == "model_a"
    )
    alpaca_json = (
        '{"ordered_models": [{"model": "m", "rank": 1}, {"model": "M", "rank": 2}]}'
    )
    assert preference_to_winner(JUDGE_PARSERS["alpaca-eval-json"](alpaca_json)) == (
        "model_a"
    )
    expected_parsers = {
        "meta-eval-pair-score": "meta-eval-score",
        "arena-hard": "arena-hard-verdict",
        "meta-eval-alpaca-eval-json": "alpaca-eval-json",
        "meta-eval-alpaca-eval-pair-score": "meta-eval-score",
    }
    for preset, expected_parser in expected_parsers.items():
        resolved = resolve_judge_prompt(preset=preset)
        assert resolved.system_prompt
        assert resolved.user_prompt_template
        assert parser_name(resolved.parser) == expected_parser


def test_meta_eval_preserves_parser_soft_preference(monkeypatch):
    def fake_alpaca_annotations(**kwargs):
        return [
            JudgeAnnotation(
                instruction=instruction,
                completion_A=completion_a,
                completion_B=completion_b,
                judge_completion="M",
                judge_input="prompt",
                judge_top_logprobs={"m": math.log(0.25), "M": math.log(0.75)},
            )
            for instruction, completion_a, completion_b in zip(
                kwargs["instructions"],
                kwargs["completions_A"],
                kwargs["completions_B"],
                strict=True,
            )
        ]

    monkeypatch.setattr(evaluate_module, "annotate_battles", fake_alpaca_annotations)
    cfg = _cfg("unused", judge={"model": "Dummy/j", "prompt_preset": "alpaca-eval"})
    annotations = annotate_module.annotate_sample(
        _battles().head(1),
        cfg,
        judge_chat_model=object(),
        resolved_prompt=resolve_judge_prompt(preset="alpaca-eval"),
    )

    assert annotations.loc[0, "winner_llm"] == "model_b"
    assert annotations.loc[0, "pref_llm"] == pytest.approx(0.75)


def test_meta_eval_delegates_judging_parsing_and_swapping_to_shared_path(monkeypatch):
    captured = {}

    def fake_judge_and_parse_prefs(**kwargs):
        captured.update(kwargs)
        annotation = JudgeAnnotation(
            instruction=kwargs["instructions"][0],
            completion_A=kwargs["completions_A"][0],
            completion_B=kwargs["completions_B"][0],
            judge_completion="unparseable",
            judge_input="prompt",
        )
        return [annotation], None, pd.Series([float("nan")])

    monkeypatch.setattr(
        annotate_module, "judge_and_parse_prefs", fake_judge_and_parse_prefs
    )
    cfg = _cfg(
        "unused",
        judge={
            "model": "Dummy/j",
            "prompt_preset": "arena-hard",
            "strip_thinking_before_judging": True,
        },
        generation={"truncate_judge_input_chars": 123},
    )
    resolved = resolve_judge_prompt(preset="arena-hard")

    annotations = annotate_module.annotate_sample(
        _battles().head(1),
        cfg,
        judge_chat_model=object(),
        resolved_prompt=resolved,
    )

    assert captured["swap_mode"] == "fixed"
    assert captured["parse"] is resolved.parser
    assert captured["strip_thinking_before_judging"] is True
    assert captured["truncate_input_chars"] == 123
    assert annotations.loc[0, "winner_llm"] == "tie"
    assert annotations.loc[0, "pref_llm"] == 0.5


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
    monkeypatch.setattr(evaluate_module, "annotate_battles", _fake_annotations)
    cfg = _cfg(
        tmp_path,
        meta_eval={
            "top_models": 3,
            "battles_per_model": 5,
            "elo_gap_battles": [2],
            "elo_gap_seeds": 1,
        },
    )

    results = run_meta_eval(cfg, get_packaged_task("meta-eval-comparia"))

    res_dir = tmp_path / "meta-eval-comparia-meta-eval-pair-score-dummy-j-fixed"
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
    assert results["elo_gap_all"][0]["num_battles"] == 2
    assert results["elo_gap_exclude_ties"][0]["exclude_ties"] is True
    assert results["elo_gap_soft"][0]["num_battles"] == 2
    summary = pd.read_csv(res_dir / SUMMARY_FILENAME)
    assert set(summary["split"]) == {"English", "Multilingual"}
    assert json.loads((res_dir / "results.json").read_text())["arena"] == "ComparIA"


def test_swap_mode_both_inverts_the_reversed_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "load_battles", lambda _task: _battles())
    monkeypatch.setattr(runner_module, "build_judge", lambda _cfg: object())
    monkeypatch.setattr(evaluate_module, "annotate_battles", _fake_annotations)
    cfg = _cfg(
        tmp_path,
        judge={"model": "Dummy/j", "swap_mode": "both"},
        meta_eval={"top_models": 3, "battles_per_model": 5},
    )

    results = run_meta_eval(cfg, get_packaged_task("meta-eval-comparia"))

    annotations = pd.read_parquet(
        tmp_path
        / "meta-eval-comparia-meta-eval-pair-score-dummy-j-both"
        / ANNOTATIONS_FILENAME
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


def test_elo_gap_summary_runs():
    battles = _battles()
    top, df_top = select_top_models(battles, top_models=3)
    sample = sample_battles_per_model(df_top, top, battles_per_model=5, seed=0)
    sample["winner_llm"] = sample["winner"]
    sample["pref_llm"] = sample["winner"].map(
        {"model_a": 0.1, "model_b": 0.9, "tie": 0.5, "tie (bothbad)": 0.5}
    )
    summary = compute_elo_gap_summary(
        df_top,
        sample,
        top,
        n_battles_list=[2],
        n_seeds=2,
        seed=0,
        exclude_ties=False,
    )
    assert not summary.empty
    soft = compute_elo_gap_summary(
        df_top,
        sample,
        top,
        n_battles_list=[2],
        n_seeds=2,
        seed=0,
        exclude_ties=False,
        soft=True,
    )
    assert not soft.empty
    neutral = sample.copy()
    neutral["pref_llm"] = 0.5
    neutral_soft = compute_elo_gap_summary(
        df_top,
        neutral,
        top,
        n_battles_list=[2],
        n_seeds=2,
        seed=0,
        exclude_ties=False,
        soft=True,
    )
    assert soft.loc[0, "mean"] != pytest.approx(neutral_soft.loc[0, "mean"])
