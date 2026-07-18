"""Tests for judge meta-evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import judgearena.meta_eval.annotate as meta_annotate
import judgearena.meta_eval.cost as meta_cost
import judgearena.meta_eval.runner as meta_eval_runner
import judgearena.meta_eval.sampling as meta_sampling
from judgearena import cli as cli_module
from judgearena.arenas_utils import extract_turn_text
from judgearena.evaluate import JudgeAnnotation, PairScore
from judgearena.meta_eval.cache import AnnotationCache, AnnotationEntry, AnnotationKey
from judgearena.meta_eval.cli_args import CliMetaEvalArgs
from judgearena.meta_eval.metrics import (
    compute_agreement_metrics,
    compute_elo_gap_summary,
)
from judgearena.meta_eval.parsers import (
    META_EVAL_PAIRSCORE_TEMPERATURE,
    invert_winner,
    parse_alpaca_eval_winner,
    parse_arena_hard_winner,
    parse_pairscore_winner,
    parse_pref,
    parse_winner,
)
from judgearena.meta_eval.prompts import PromptModeSpec, resolve_prompt_mode
from judgearena.meta_eval.sampling import (
    MetaEvalSamplingError,
    load_reference_arena_battles,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.repro import METADATA_FILENAME


def _conversation(
    instruction: str,
    answer_a: str,
    answer_b: str,
    *,
    structured: bool = False,
) -> tuple[list[dict], list[dict]]:
    if structured:
        user_content = [{"type": "text", "text": instruction, "image": None}]
        assistant_content = [{"type": "text", "text": answer_a}]
        assistant_b_content = [{"type": "text", "text": answer_b}]
    else:
        user_content = instruction
        assistant_content = answer_a
        assistant_b_content = answer_b
    conv_a = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    conv_b = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_b_content},
    ]
    return conv_a, conv_b


@pytest.fixture
def synthetic_arena_df() -> pd.DataFrame:
    rows = []
    models = [f"model-{idx}" for idx in range(5)]
    for idx in range(120):
        model_a = models[idx % len(models)]
        model_b = models[(idx + 1) % len(models)]
        winner = ["model_a", "model_b", "tie"][idx % 3]
        lang = "en" if idx % 2 == 0 else "es"
        conv_a, conv_b = _conversation(
            f"Question {idx}",
            f"Answer A {idx}",
            f"Answer B {idx}",
            structured=idx % 4 == 0,
        )
        rows.append(
            {
                "question_id": f"q-{idx}",
                "tstamp": 1_700_000_000 + idx,
                "model_a": model_a,
                "model_b": model_b,
                "winner": winner,
                "conversation_a": conv_a,
                "conversation_b": conv_b,
                "benchmark": "LMArena-140k",
                "lang": lang,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def meta_args(tmp_path: Path) -> CliMetaEvalArgs:
    return CliMetaEvalArgs(
        reference_arena="LMArena-140k",
        prompt_mode="standard",
        top_models=3,
        battles_per_model=4,
        batch_size=8,
        languages=["en", "es"],
        n_bootstraps=20,
        seed=7,
        judge_model="Dummy/judge",
        result_folder=str(tmp_path / "results"),
        no_log_file=True,
    )


def _judge_annotations(*, instructions, completions_A, completions_B, **_kwargs):
    return [
        JudgeAnnotation(
            instruction=instruction,
            completion_A=completion_a,
            completion_B=completion_b,
            judge_completion="score_A: 9\nscore_B: 1",
            judge_input="judge prompt",
        )
        for instruction, completion_a, completion_b in zip(
            instructions,
            completions_A,
            completions_B,
            strict=True,
        )
    ]


@pytest.fixture
def stub_meta_eval_runner(monkeypatch, synthetic_arena_df):
    monkeypatch.setattr(
        meta_eval_runner,
        "load_reference_arena_battles",
        lambda reference_arena, languages=None: synthetic_arena_df,
    )
    monkeypatch.setattr(meta_eval_runner, "make_model", lambda **_kwargs: object())


def test_cli_meta_eval_dispatch(monkeypatch):
    captured: dict[str, object] = {}

    def fake_main(args: CliMetaEvalArgs) -> None:
        captured["args"] = args

    monkeypatch.setattr(cli_module, "main_meta_eval", fake_main)
    cli_module.cli(
        [
            "--task",
            "meta-eval",
            "--judge",
            "Dummy/J",
            "--reference_arena",
            "LMArena-140k",
            "--prompt_mode",
            "arena-hard",
            "--languages",
            "en",
            "es",
            "--top_models",
            "5",
            "--battles_per_model",
            "10",
        ]
    )
    args: CliMetaEvalArgs = captured["args"]
    assert args.reference_arena == "LMArena-140k"
    assert args.prompt_mode == "arena-hard"
    assert args.languages == ["en", "es"]
    assert args.top_models == 5


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--model_A", "Dummy/A", "--model_A/--model_B are not used"),
        ("--n_instructions", "10", "--n_instructions is not used"),
        (
            "--max_out_tokens_models",
            "1024",
            "--max_out_tokens_models is not used",
        ),
    ],
)
def test_cli_meta_eval_rejects_irrelevant_flags(
    monkeypatch,
    flag,
    value,
    message,
):
    monkeypatch.setattr(cli_module, "main_meta_eval", lambda _args: None)
    with pytest.raises(SystemExit, match=message):
        cli_module.cli(
            [
                "--task",
                "meta-eval",
                "--judge",
                "Dummy/J",
                flag,
                value,
            ]
        )


def test_extract_text_structured_content():
    conv_a, _ = _conversation("hello", "A", "B", structured=True)
    assert extract_turn_text(conv_a[0]) == "hello"
    assert (
        extract_turn_text(
            {"content": [{"type": "text", "text": "part one"}]},
        )
        == "part one"
    )


def test_language_filter_empty_raises(monkeypatch, synthetic_arena_df):
    monkeypatch.setattr(
        meta_sampling,
        "load_arena_dataframe",
        lambda arena: synthetic_arena_df,
    )
    with pytest.raises(MetaEvalSamplingError, match="languages"):
        load_reference_arena_battles("LMArena-140k", languages=["zz"])


def test_sampling_is_deterministic(synthetic_arena_df):
    df = synthetic_arena_df.copy()
    top_models, df_top = select_top_models(df, top_models=3)
    first = sample_battles_per_model(
        df_top,
        top_models,
        battles_per_model=3,
        seed=11,
    )
    second = sample_battles_per_model(
        df_top,
        top_models,
        battles_per_model=3,
        seed=11,
    )
    assert first["question_id"].tolist() == second["question_id"].tolist()
    assert len(first) == 9


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("My verdict is [[A>>B]]", "model_a"),
        ("Final: [[B>A]]", "model_b"),
        ("No signal", "tie"),
    ],
)
def test_parse_arena_hard_winner(completion, expected):
    assert parse_arena_hard_winner(completion) == expected


def test_parse_alpaca_eval_winner():
    completion = (
        '```json\n{"ordered_models": [{"model": "m", "rank": 2}, '
        '{"model": "M", "rank": 1}]}\n```'
    )
    assert parse_alpaca_eval_winner(completion) == "model_b"


def test_pairscore_temperature_and_tie():
    completion = "score_A: 5\nscore_B: 5"
    benchmark = PairScore()
    benchmark.temperature = 0.3
    meta = PairScore()
    meta.temperature = META_EVAL_PAIRSCORE_TEMPERATURE
    assert benchmark.parse_model_raw(completion) == 0.5
    assert meta.parse_model_raw(completion) == 0.5
    assert parse_pairscore_winner(completion, temperature=0.5) == "tie"
    assert (
        parse_pairscore_winner(
            "score_A: 10\nscore_B: 0",
            temperature=0.5,
        )
        == "model_a"
    )


@pytest.mark.parametrize(
    ("mode", "completion", "expected"),
    [
        ("standard", "score_A: 9\nscore_B: 1", "model_a"),
        ("arena-hard", "[[A=B]]", "tie"),
        (
            "alpaca-eval",
            '{"ordered_models": [{"model": "m", "rank": 1}]}',
            "model_a",
        ),
        ("alpaca-eval-pair-score", "score_A: 9\nscore_B: 1", "model_a"),
    ],
)
def test_parse_winner_modes(mode, completion, expected):
    assert parse_winner(completion, mode) == expected


@pytest.mark.parametrize(
    "mode",
    ["arena-hard", "alpaca-eval", "alpaca-eval-pair-score"],
)
def test_file_backed_prompt_modes_load_packaged_resources(mode):
    prompt = resolve_prompt_mode(mode)
    assert prompt.system_prompt
    assert prompt.user_prompt_template


def test_parse_pref_continuous_semantics():
    assert parse_pref("score_A: 10\nscore_B: 0", "standard") < 0.5
    assert parse_pref("score_A: 0\nscore_B: 10", "standard") > 0.5
    assert parse_pref("[[A=B]]", "arena-hard") == 0.5


def test_swap_inversion():
    assert invert_winner("model_a") == "model_b"
    assert invert_winner("tie") == "tie"


def test_agreement_metrics_on_fixture():
    human = ["model_a", "model_b", "tie", "model_a"]
    llm = ["model_a", "model_a", "tie", "model_b"]
    metrics = compute_agreement_metrics(
        human,
        llm,
        n_bootstraps=10,
        seed=0,
    )
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["n_nt"] == 3


def _cache_key(**overrides) -> AnnotationKey:
    values = {
        "benchmark": "LMArena-140k",
        "instruction_id": "q-1",
        "model_a": "model-a",
        "model_b": "model-b",
        "judge": "Dummy/judge",
    }
    values.update(overrides)
    return AnnotationKey(**values)


def _cache_entry(completion: str = "score_A: 9\nscore_B: 1", **overrides):
    key = _cache_key(**overrides)
    return AnnotationEntry(
        **key.__dict__,
        judge_input="judge prompt",
        judge_completion=completion,
    )


def test_annotation_cache_persists_and_preserves_batch_order(tmp_path):
    db_dir = tmp_path / "db"
    first = AnnotationCache(db_dir)
    first.batch_put(
        [
            _cache_entry(instruction_id="q-2", completion="second"),
            _cache_entry(instruction_id="q-1", completion="first"),
        ]
    )
    first.close()

    second = AnnotationCache(db_dir)
    entries = second.batch_get_annotations(
        [
            _cache_key(instruction_id="q-1"),
            _cache_key(instruction_id="q-2"),
            _cache_key(instruction_id="missing"),
        ]
    )
    assert [entry.judge_completion if entry else None for entry in entries] == [
        "first",
        "second",
        None,
    ]
    second.close()


def test_annotation_cache_distinguishes_prompt_mode_and_model_order(tmp_path):
    cache = AnnotationCache(tmp_path / "db")
    cache.batch_put(
        [
            _cache_entry(judge="Dummy/judge::arena-hard"),
            _cache_entry(model_a="model-b", model_b="model-a"),
        ]
    )
    entries = cache.batch_get_annotations(
        [
            _cache_key(judge="Dummy/judge"),
            _cache_key(judge="Dummy/judge::arena-hard"),
            _cache_key(model_a="model-b", model_b="model-a"),
        ]
    )
    assert entries[0] is None
    assert all(entry is not None for entry in entries[1:])
    cache.close()


def test_annotate_sample_uses_cache_and_inverts_swapped_pass(
    monkeypatch,
    synthetic_arena_df,
    meta_args,
    tmp_path,
):
    calls = {"count": 0}

    def fake_annotate_battles(**kwargs):
        calls["count"] += 1
        return _judge_annotations(**kwargs)

    monkeypatch.setattr(meta_annotate, "annotate_battles", fake_annotate_battles)
    meta_args.swap_mode = "both"
    sample = synthetic_arena_df.iloc[:1]
    cache = AnnotationCache(tmp_path / "db")
    prompt_spec = PromptModeSpec(
        name="standard",
        system_prompt="system",
        user_prompt_template="user",
    )

    annotations = meta_annotate.annotate_sample(
        sample,
        meta_args,
        judge_chat_model=object(),
        prompt_spec=prompt_spec,
        annotation_cache=cache,
    )
    assert len(annotations) == 2
    assert annotations["orientation"].tolist() == ["forward", "swapped"]
    assert annotations["winner"].tolist() == [sample.iloc[0]["winner"]] * 2
    assert annotations["winner_llm"].tolist() == ["model_a", "model_b"]
    assert annotations["model_a"].tolist() == [sample.iloc[0]["model_a"]] * 2
    assert annotations["presented_model_a"].tolist() == [
        sample.iloc[0]["model_a"],
        sample.iloc[0]["model_b"],
    ]
    assert annotations["completion_a"].nunique() == 1
    assert (
        annotations.iloc[1]["presented_completion_a"]
        == annotations.iloc[0]["completion_b"]
    )
    assert calls["count"] == 2

    meta_annotate.annotate_sample(
        sample,
        meta_args,
        judge_chat_model=object(),
        prompt_spec=prompt_spec,
        annotation_cache=cache,
    )
    assert calls["count"] == 2

    meta_args.ignore_cache = True
    meta_annotate.annotate_sample(
        sample,
        meta_args,
        judge_chat_model=object(),
        prompt_spec=prompt_spec,
        annotation_cache=cache,
    )
    assert calls["count"] == 4
    cache.close()


def test_cost_uses_offline_reference_pricing(monkeypatch, tmp_path):
    pricing_file = tmp_path / "openrouter_pricing.json"
    pricing_file.write_text(
        json.dumps({"provider/model": [1.0, 2.0]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(meta_cost, "_PRICING_CACHE_FILE", pricing_file)
    meta_cost._openrouter_pricing_cache.clear()

    cost, source = meta_cost.estimate_annotation_cost_usd(
        judge_input="a" * 40,
        judge_completion="b" * 20,
        judge_model="OpenRouter/provider/model",
    )
    assert cost == pytest.approx((10 * 1.0 + 5 * 2.0) / 1e6)
    assert source == "estimated"
    meta_cost._openrouter_pricing_cache.clear()


def test_swapped_pass_telemetry_counts_both_judgements():
    annotations = pd.DataFrame(
        {
            "cost_usd": [0.1, 0.2],
            "cost_source": ["estimated", "estimated"],
            "estimated_input_tokens": [10, 11],
            "estimated_output_tokens": [5, 6],
        }
    )
    telemetry = meta_eval_runner._annotation_telemetry(
        annotations,
        swap_mode="both",
    )
    assert telemetry["judgement_count"] == 2
    assert telemetry["total_cost_usd"] == pytest.approx(0.3)
    assert telemetry["estimated_input_tokens"] == 21
    assert telemetry["estimated_output_tokens"] == 11


def test_degenerate_agreement_metrics_do_not_warn(recwarn):
    metrics = compute_agreement_metrics(
        ["model_a"] * 4,
        ["model_a"] * 4,
        n_bootstraps=10,
        seed=0,
    )
    assert metrics["accuracy"] == 1.0
    assert pd.isna(metrics["kappa"])
    assert not recwarn.list


def test_integration_meta_eval_artifacts(
    monkeypatch,
    meta_args: CliMetaEvalArgs,
    stub_meta_eval_runner,
):
    def fake_annotate(df_sample, args, **kwargs):
        frame = df_sample[
            ["question_id", "model_a", "model_b", "winner", "lang", "benchmark"]
        ].copy()
        winners = frame["winner"].where(frame["winner"] == "model_a", "model_b")
        return frame.assign(
            orientation="forward",
            instruction="instr",
            completion_a="A",
            completion_b="B",
            judge_input="prompt",
            judge_completion="score_A: 9\nscore_B: 1",
            estimated_input_tokens=2,
            estimated_output_tokens=5,
            cost_usd=0.001,
            cost_source="estimated",
            winner_llm=winners,
            pref_llm=winners.map({"model_a": 0.0, "model_b": 1.0}),
        )

    monkeypatch.setattr(meta_eval_runner, "annotate_sample", fake_annotate)

    results = meta_eval_runner.main(meta_args)
    output_dirs = list(Path(meta_args.result_folder).glob("meta-eval-*"))
    assert len(output_dirs) == 1
    out = output_dirs[0]
    assert (out / "args.json").exists()
    assert (out / "annotations.parquet").exists()
    assert (out / "results.json").exists()
    assert (out / "summary.csv").exists()
    assert (out / METADATA_FILENAME).exists()
    metadata = json.loads((out / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["entrypoint"] == "judgearena.meta_eval.runner"
    assert results["agreement"]["primary_view"] == "no_human_ties"
    assert results["agreement"]["all"]["n"] > results["agreement"]["no_human_ties"]["n"]
    assert results["judgement_count"] == results["sample_size"]
    assert results["total_cost_usd"] == pytest.approx(results["sample_size"] * 0.001)
    assert "English" in results["language_summary"]


def test_swap_mode_both_artifact_reproduces_overall_agreement(
    monkeypatch,
    meta_args,
    stub_meta_eval_runner,
    tmp_path,
):
    cache_class = AnnotationCache
    monkeypatch.setattr(
        meta_annotate,
        "AnnotationCache",
        lambda: cache_class(tmp_path / "cache"),
    )
    monkeypatch.setattr(meta_annotate, "annotate_battles", _judge_annotations)
    meta_args.swap_mode = "both"
    meta_args.battles_per_model = 2
    meta_args.elo_gap_battles = [1]
    meta_args.elo_gap_seeds = 1

    results = meta_eval_runner.main(meta_args)
    output_dir = next(Path(meta_args.result_folder).glob("meta-eval-*"))
    annotations = pd.read_parquet(output_dir / "annotations.parquet")
    forward = annotations[annotations["orientation"] == "forward"]
    swapped = annotations[annotations["orientation"] == "swapped"]

    assert len(annotations) == 2 * results["sample_size"]
    assert len(forward) == len(swapped) == results["sample_size"]
    assert results["judgement_count"] == len(annotations)
    assert results["ranking_annotation_count"] == len(forward)
    assert (swapped["model_a"].to_numpy() == swapped["presented_model_b"]).all()
    assert (swapped["model_b"].to_numpy() == swapped["presented_model_a"]).all()
    assert (swapped["completion_a"].to_numpy() == forward["completion_a"]).all()
    assert (swapped["completion_b"].to_numpy() == forward["completion_b"]).all()

    recomputed = compute_agreement_metrics(
        annotations["winner"].tolist(),
        annotations["winner_llm"].tolist(),
        n_bootstraps=meta_args.n_bootstraps,
        seed=meta_args.seed,
    )
    assert results["agreement"]["all"]["accuracy"] == recomputed["accuracy"]
    assert results["agreement"]["all"]["kappa"] == recomputed["kappa"]


def test_elo_gap_summary_runs(synthetic_arena_df):
    top_models, df_top = select_top_models(synthetic_arena_df, top_models=3)
    df_sample = sample_battles_per_model(
        df_top,
        top_models,
        battles_per_model=3,
        seed=1,
    )
    df_ann = df_sample.copy()
    df_ann["winner_llm"] = df_ann["winner"]
    df_ann["pref_llm"] = 0.5
    summary = compute_elo_gap_summary(
        df_top,
        df_ann,
        top_models,
        n_battles_list=[2],
        n_seeds=2,
        seed=0,
        exclude_ties=False,
    )
    assert not summary.empty


def test_empty_language_subset_reports_na(
    monkeypatch,
    synthetic_arena_df,
    meta_args: CliMetaEvalArgs,
):
    meta_args.languages = ["en"]

    def _raise_empty(_reference_arena, languages=None):
        raise MetaEvalSamplingError(
            "No battles remain after filtering to languages: en."
        )

    monkeypatch.setattr(meta_eval_runner, "load_reference_arena_battles", _raise_empty)
    with pytest.raises(SystemExit):
        meta_eval_runner.run_or_exit(meta_args)
