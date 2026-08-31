from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.annotate as annotate_module
import judgearena.benchmarks.meta_eval.runner as runner_module
import judgearena.evaluate as evaluate_module
from judgearena.artifacts import (
    atomic_write_path,
    prepare_unique_run_directory,
    scoped_run_file_logging,
)
from judgearena.benchmarks.meta_eval.agreement import compute_agreement_metrics
from judgearena.benchmarks.meta_eval.annotate import (
    aggregate_battle_preferences,
    preference_to_winner,
)
from judgearena.benchmarks.meta_eval.runner import (
    ANNOTATIONS_FILENAME,
    BATTLE_RESULTS_FILENAME,
    SAMPLE_FILENAME,
    SUMMARY_FILENAME,
    run_meta_eval,
)
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    comparison_components,
    count_battles_per_model,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.benchmarks.meta_eval.scoring import (
    _fit_connected_preferences,
    compute_elo_gap_summary,
    summarize_language_splits,
)
from judgearena.benchmarks.registry import resolve_benchmark
from judgearena.config import RunConfig
from judgearena.evaluate import JudgeAnnotation
from judgearena.log import get_logger
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
            "battle_id": f"ComparIA:q{i}",
            "question_id": f"q{i}",
            "model_a": models[i % 3],
            "model_b": models[(i + 1) % 3],
            "winner": "tie" if i % 5 == 0 else "model_a",
            "lang": "fr" if i % 2 else "en",
            "conversation_a": _turns("a"),
            "conversation_b": _turns("b"),
        }
        for i in range(30)
    ]
    rows.append(
        {
            "battle_id": "ComparIA:q-rare",
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


def _runtime_cfg(tmp_path, *, swap_mode="fixed", **meta_overrides) -> RunConfig:
    meta_eval = {
        "top_models": 3,
        "battles_per_model": 5,
        "elo_gap_battles": [2],
        "elo_gap_seeds": 2,
    }
    meta_eval.update(meta_overrides)
    return _cfg(
        tmp_path,
        judge={"model": "Dummy/j", "swap_mode": swap_mode},
        meta_eval=meta_eval,
    )


def _stub_runtime(monkeypatch, *, battles=None, annotate=_fake_annotations, build=None):
    monkeypatch.setattr(
        runner_module,
        "load_battles",
        lambda _task: _battles() if battles is None else battles,
    )
    monkeypatch.setattr(runner_module, "build_judge", build or (lambda _cfg: object()))
    if annotate is not None:
        monkeypatch.setattr(evaluate_module, "annotate_battles", annotate)


def _assert_rejected_before_judge(
    tmp_path, monkeypatch, battles, exception, match, *, cfg=None
):
    _stub_runtime(
        monkeypatch,
        battles=battles,
        annotate=None,
        build=lambda _cfg: pytest.fail("judge must not be built"),
    )
    with pytest.raises(exception, match=match):
        run_meta_eval(
            cfg or _runtime_cfg(tmp_path), get_packaged_task("meta-eval-comparia")
        )
    assert not list(tmp_path.iterdir())


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

    assert sample["battle_id"].is_unique
    assert all(count >= 5 for count in count_battles_per_model(sample).values())
    assert len(comparison_components(sample, top)) == 1
    pd.testing.assert_frame_equal(sample, again)
    shuffled = sample_battles_per_model(
        df_top.sample(frac=1, random_state=42).reset_index(drop=True),
        top,
        battles_per_model=5,
        seed=0,
    )
    assert shuffled["battle_id"].tolist() == sample["battle_id"].tolist()


def test_sampling_rejects_a_disconnected_top_model_set():
    battles = pd.DataFrame(
        [
            {"question_id": "q1", "model_a": "a", "model_b": "b", "winner": "tie"},
            {"question_id": "q2", "model_a": "c", "model_b": "d", "winner": "tie"},
        ]
    )
    with pytest.raises(MetaEvalSamplingError, match="disconnected"):
        select_top_models(battles, top_models=4)


def test_sampling_requires_stable_battle_ids():
    battles = _battles().drop(columns="battle_id")
    top, df_top = select_top_models(battles, top_models=3)
    with pytest.raises(MetaEvalSamplingError, match="Stable battle_id"):
        sample_battles_per_model(df_top, top, battles_per_model=5, seed=0)


def test_sampling_rejects_self_comparisons():
    battles = pd.DataFrame(
        [
            {
                "battle_id": "arena:self",
                "question_id": "self",
                "model_a": "a",
                "model_b": "a",
                "winner": "model_a",
            },
            {
                "battle_id": "arena:ab",
                "question_id": "ab",
                "model_a": "a",
                "model_b": "b",
                "winner": "model_a",
            },
        ]
    )
    with pytest.raises(MetaEvalSamplingError, match="self-comparisons"):
        select_top_models(battles, top_models=2)
    with pytest.raises(MetaEvalSamplingError, match="self-comparisons"):
        sample_battles_per_model(
            battles,
            ["a", "b"],
            battles_per_model=2,
            seed=0,
        )
    assert count_battles_per_model(battles) == {"a": 2, "b": 1}


def test_sampling_rejects_insufficient_unique_quota():
    battles = pd.DataFrame(
        [
            {
                "battle_id": f"arena:q{i}",
                "question_id": f"q{i}",
                "model_a": "a",
                "model_b": "b",
                "winner": "model_a",
            }
            for i in range(2)
        ]
    )
    top, df_top = select_top_models(battles, top_models=2)
    with pytest.raises(MetaEvalSamplingError, match="Insufficient unique battles"):
        sample_battles_per_model(df_top, top, battles_per_model=3, seed=0)


def test_meta_eval_bt_fit_rejects_disconnected_parsed_graph():
    disconnected = pd.DataFrame(
        {
            "model_a": ["a", "c"],
            "model_b": ["b", "d"],
            "pref": [0.2, 0.8],
        }
    )
    with pytest.raises(MetaEvalSamplingError, match="disconnected"):
        _fit_connected_preferences(disconnected, pref_col="pref")


@pytest.mark.parametrize(
    ("completion", "winner"),
    [
        ("score_A: 10\nscore_B: 0", "model_a"),
        ("score_A: 0\nscore_B: 10", "model_b"),
        ("score_A: 5\nscore_B: 5", "tie"),
        ('"score_A": 9,\n"score_B": 1', "model_a"),
    ],
)
def test_pairscore_parses_winners_at_meta_eval_temperature(completion, winner):
    preference = JUDGE_PARSERS["meta-eval-score"](completion)

    assert preference is not None
    assert preference_to_winner(preference) == winner


@pytest.mark.parametrize(
    "completion",
    [
        "score_A: 8.9\nscore_B: 8.1",
        "score_A: 9,5\nscore_B: 1",
        "score_A: 9/10\nscore_B: 1",
        "scores: banana 7\nscore_B: 3",
        "score_A: 1e1\nscore_B: 8",
        "score_A: -1\nscore_B: 8",
        "score_A: 11\nscore_B: 8",
        "score_A: 0009\nscore_B: 1",
        "score_A: " + "0" * 4_300 + "9\nscore_B: 1",
        "score_A: 8\nscore_B: 3.0",
    ],
)
def test_meta_eval_pairscore_rejects_non_integer_or_out_of_range_scores(completion):
    assert JUDGE_PARSERS["meta-eval-score"](completion) is None


@pytest.mark.parametrize(
    ("preset", "expected_parser"),
    [
        ("meta-eval-pair-score", "meta-eval-score"),
        ("arena-hard", "arena-hard-verdict"),
        ("meta-eval-alpaca-eval-json", "alpaca-eval-json"),
        ("meta-eval-alpaca-eval-pair-score", "meta-eval-score"),
    ],
)
def test_prompt_presets_select_their_parsers(preset, expected_parser):
    assert parser_name(resolve_judge_prompt(preset=preset).parser) == expected_parser


@pytest.mark.parametrize(
    "completion",
    [
        "[]",
        '{"ordered_models": ["m", "M"]}',
        '{"ordered_models": [{"model": "m", "rank": 1}]}',
        '{"ordered_models": [{"model": "m", "rank": 1}, {"model": "M", "rank": 1}]}',
        '{"ordered_models": [{"model": "m", "rank": 1}, {"model": "X", "rank": 2}]}',
        '{"ordered_models": [{"model": [], "rank": 1}, {"model": "M", "rank": 2}]}',
        '{"ordered_models": [{"model": {}, "rank": 1}, {"model": "M", "rank": 2}]}',
        '{"ordered_models": [{"model": "m", "rank": true}, {"model": "M", "rank": 2}]}',
        '{"ordered_models": [{"model": "m", "rank": 1}, '
        '{"model": "M", "rank": 2}, {"model": "X", "rank": 3}]}',
    ],
)
def test_alpaca_eval_json_parser_rejects_malformed_rankings(completion):
    assert JUDGE_PARSERS["alpaca-eval-json"](completion) is None


def test_alpaca_eval_json_parser_requires_complementary_complete_ranking():
    parser = JUDGE_PARSERS["alpaca-eval-json"]

    assert (
        parser(
            '{"ordered_models": [{"model": "m", "rank": 1}, {"model": "M", "rank": 2}]}'
        )
        == 0.0
    )
    assert (
        parser(
            '```json\n{"ordered_models": [{"model": "M", "rank": 1}, '
            '{"model": "m", "rank": 2}]}\n```'
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "preset",
    ["meta-eval-alpaca-eval-json", "meta-eval-alpaca-eval-pair-score"],
)
def test_meta_eval_alpaca_prompts_embed_valid_json(preset, monkeypatch):
    captured_inputs = []

    def fake_do_inference(**kwargs):
        captured_inputs.extend(kwargs["inputs"])
        return [
            '{"ordered_models": [{"model": "m", "rank": 1}, {"model": "M", "rank": 2}]}'
        ]

    monkeypatch.setattr(evaluate_module, "do_inference", fake_do_inference)
    instruction = 'Say "hi"\r\nnext \\ path {curly} café'
    completion_a = 'A says "yes"\nC:\\tmp {a}'
    completion_b = 'B says "no"\r\nD:\\tmp {b}'

    evaluate_module.annotate_battles(
        judge_chat_model=object(),
        instructions=[instruction],
        completions_A=[completion_a],
        completions_B=[completion_b],
        prompt_preset=preset,
        truncate_input_chars=None,
    )

    rendered = captured_inputs[0].messages[-1].content
    prompt_block = rendered.split("## Prompt\n\n", 1)[1].split(
        "\n\n## Model Outputs", 1
    )[0]
    outputs_block = rendered.split("## Model Outputs\n\n", 1)[1]
    outputs_block = outputs_block.split("\n\n## Task", 1)[0].split("\n\n", 1)[1]

    assert json.loads(prompt_block) == {"instruction": instruction}
    assert json.loads(outputs_block) == [
        {
            "model": "m" if preset.endswith("json") else "model A",
            "output": completion_a,
        },
        {
            "model": "M" if preset.endswith("json") else "model B",
            "output": completion_b,
        },
    ]


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
    assert annotations.loc[0, "parse_ok"]
    top_logprobs = json.loads(annotations.loc[0, "judge_top_logprobs_json"])
    assert top_logprobs["M"] == pytest.approx(math.log(0.75))
    assert top_logprobs["m"] == pytest.approx(math.log(0.25))


def test_physical_battle_aggregation_combines_both_orders_and_parse_statuses():
    rows = []
    for battle_id, preferences in {
        "complete": [0.2, 0.4],
        "partial": [0.7, float("nan")],
        "missing": [float("nan"), float("nan")],
    }.items():
        for orientation, preference in zip(
            ["forward", "swapped"], preferences, strict=True
        ):
            rows.append(
                {
                    "battle_id": battle_id,
                    "question_id": battle_id,
                    "model_a": "a",
                    "model_b": "b",
                    "winner": "model_a",
                    "lang": "en",
                    "orientation": orientation,
                    "parse_ok": not pd.isna(preference),
                    "pref_llm": preference,
                }
            )

    aggregated = aggregate_battle_preferences(
        pd.DataFrame(rows).sample(frac=1, random_state=7), swap_mode="both"
    ).set_index("battle_id")

    assert aggregated.loc["complete", "pref_llm"] == pytest.approx(0.3)
    assert aggregated.loc["complete", "winner_llm"] == "model_a"
    assert aggregated.loc["complete", "parse_status"] == "complete"
    assert aggregated.loc["partial", "pref_llm"] == pytest.approx(0.7)
    assert aggregated.loc["partial", "winner_llm"] == "model_b"
    assert aggregated.loc["partial", "parse_status"] == "partial"
    assert not bool(aggregated.loc["missing", "parse_ok"])
    assert pd.isna(aggregated.loc["missing", "pref_llm"])
    assert pd.isna(aggregated.loc["missing", "winner_llm"])
    assert aggregated.loc["missing", "parse_status"] == "missing"


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
    def fake_build(_cfg):
        run_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / SAMPLE_FILENAME).is_file()
        assert not (run_dir / ANNOTATIONS_FILENAME).exists()
        return object()

    _stub_runtime(monkeypatch, build=fake_build)
    results = run_meta_eval(
        _runtime_cfg(tmp_path), get_packaged_task("meta-eval-comparia")
    )

    res_dir = Path(results["result_path"]).parent
    sample = pd.read_parquet(res_dir / SAMPLE_FILENAME)
    annotations = pd.read_parquet(res_dir / ANNOTATIONS_FILENAME)
    battle_results = pd.read_parquet(res_dir / BATTLE_RESULTS_FILENAME)
    assert res_dir.parent == tmp_path
    assert sample["battle_id"].is_unique
    assert len(annotations) == len(battle_results) == results["n_battles"]
    assert results["battle_parse_status"] == {"complete": len(sample)}
    assert set(pd.read_csv(res_dir / SUMMARY_FILENAME)["split"]) == {
        "English",
        "Multilingual",
    }
    assert json.loads((res_dir / "results.json").read_text())["arena"] == "ComparIA"


def test_swap_mode_both_inverts_the_reversed_pass(tmp_path, monkeypatch):
    _stub_runtime(monkeypatch)
    results = run_meta_eval(
        _runtime_cfg(tmp_path, swap_mode="both"),
        get_packaged_task("meta-eval-comparia"),
    )

    res_dir = Path(results["result_path"]).parent
    annotations = pd.read_parquet(res_dir / ANNOTATIONS_FILENAME)
    battle_results = pd.read_parquet(res_dir / BATTLE_RESULTS_FILENAME)
    assert results["n_annotations"] == 2 * results["n_battles"]
    assert set(annotations["orientation"]) == {"forward", "swapped"}
    forward = annotations[annotations["orientation"] == "forward"]
    swapped = annotations[annotations["orientation"] == "swapped"]
    assert set(forward["winner_llm"]) == {"model_a"}
    assert set(swapped["winner_llm"]) == {"model_b"}
    assert forward["model_a"].tolist() == swapped["model_a"].tolist()
    assert (
        forward["presented_model_a"].tolist() == swapped["presented_model_b"].tolist()
    )
    assert set(battle_results["winner_llm"]) == {"tie"}
    assert set(battle_results["parse_status"]) == {"complete"}


def test_runner_excludes_missing_parses_from_metrics(tmp_path, monkeypatch):
    _stub_runtime(monkeypatch, annotate=None)

    def annotations_with_one_failure(**kwargs):
        annotations = _fake_annotations(**kwargs)
        annotations[0].judge_completion = "unparseable"
        return annotations

    monkeypatch.setattr(
        evaluate_module, "annotate_battles", annotations_with_one_failure
    )
    cfg = _runtime_cfg(tmp_path)

    results = run_meta_eval(cfg, get_packaged_task("meta-eval-comparia"))
    res_dir = Path(results["result_path"]).parent
    annotations = pd.read_parquet(res_dir / ANNOTATIONS_FILENAME)
    battle_results = pd.read_parquet(res_dir / BATTLE_RESULTS_FILENAME)

    assert results["n_parsed_annotations"] == results["n_annotations"] - 1
    assert results["n_scored_battles"] == results["n_battles"] - 1
    assert results["agreement"]["all"]["n"] == results["n_scored_battles"]
    assert results["battle_parse_status"]["missing"] == 1
    assert (~annotations["parse_ok"]).sum() == 1
    assert annotations.loc[~annotations["parse_ok"], "pref_llm"].isna().all()
    assert (
        battle_results.loc[battle_results["parse_status"] == "missing", "pref_llm"]
        .isna()
        .all()
    )


@pytest.mark.parametrize(
    "meta_eval",
    [
        {"top_models": 1},
        {"elo_gap_battles": []},
        {"elo_gap_battles": [0]},
        {"elo_gap_battles": [10, 10]},
        {"elo_gap_battles": [20, 10]},
        {"elo_gap_seeds": 1},
        {"n_bootstraps": 1},
        {"battles_per_model": 5, "elo_gap_battles": [6]},
    ],
)
def test_meta_eval_settings_reject_invalid_values_before_runtime(meta_eval):
    with pytest.raises(ValueError):
        _cfg("unused", meta_eval=meta_eval)


def test_runner_rejects_disconnected_pool_before_building_judge(tmp_path, monkeypatch):
    disconnected = pd.DataFrame(
        [
            {
                "question_id": f"q-{model_a}-{model_b}",
                "model_a": model_a,
                "model_b": model_b,
                "winner": "model_a",
                "lang": "en",
                "conversation_a": _turns("a"),
                "conversation_b": _turns("b"),
            }
            for model_a, model_b in [("a", "b"), ("c", "d")]
        ]
    )
    cfg = _runtime_cfg(tmp_path, top_models=4, battles_per_model=1, elo_gap_battles=[1])
    _assert_rejected_before_judge(
        tmp_path,
        monkeypatch,
        disconnected,
        MetaEvalSamplingError,
        "disconnected",
        cfg=cfg,
    )


def test_runner_rejects_duplicate_source_battle_ids_before_building_judge(
    tmp_path, monkeypatch
):
    duplicate_ids = _battles()
    duplicate_ids.loc[1, "question_id"] = duplicate_ids.loc[0, "question_id"]
    _assert_rejected_before_judge(
        tmp_path,
        monkeypatch,
        duplicate_ids,
        MetaEvalSamplingError,
        "duplicate physical battle IDs",
    )


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("winner", "bogus", "invalid human winners"),
        ("conversation_a", [], "empty or invalid conversation_a"),
    ],
)
def test_runner_rejects_invalid_battles_before_building_judge(
    tmp_path, monkeypatch, column, value, error
):
    invalid = _battles()
    invalid[column] = [value for _ in range(len(invalid))]
    _assert_rejected_before_judge(tmp_path, monkeypatch, invalid, ValueError, error)


def test_runner_checkpoints_annotations_before_scoring(tmp_path, monkeypatch):
    _stub_runtime(monkeypatch)

    def failing_scorer(*args, **kwargs):
        run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
        assert len(run_dirs) == 1
        assert (run_dirs[0] / ANNOTATIONS_FILENAME).is_file()
        assert (run_dirs[0] / BATTLE_RESULTS_FILENAME).is_file()
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(
        runner_module, "resolve_meta_eval_scorer", lambda _name: failing_scorer
    )
    cfg = _runtime_cfg(tmp_path)

    with pytest.raises(RuntimeError, match="scoring failed"):
        run_meta_eval(cfg, get_packaged_task("meta-eval-comparia"))
    res_dir = next(path for path in tmp_path.iterdir() if path.is_dir())
    assert not (res_dir / "results.json").exists()
    assert not (res_dir / SUMMARY_FILENAME).exists()


def test_unique_run_directories_never_reuse_an_invocation(tmp_path):
    cfg = _cfg(tmp_path)
    first = prepare_unique_run_directory(cfg, tmp_path, task=cfg.task)
    first_config = (first / "config.yaml").read_bytes()
    second = prepare_unique_run_directory(cfg, tmp_path, task=cfg.task)

    assert first != second
    assert (first / "config.yaml").read_bytes() == first_config
    assert (second / "config.yaml").is_file()


def test_scoped_run_log_does_not_leak_into_later_runs(tmp_path):
    cfg = _cfg(
        tmp_path,
        run={"result_folder": str(tmp_path), "no_log_file": False},
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    root_logger = get_logger()
    handlers_before = list(root_logger.handlers)

    with scoped_run_file_logging(cfg, first_dir):
        get_logger(__name__).warning("FIRST-RUN-ONLY")
        with scoped_run_file_logging(cfg, second_dir):
            get_logger(__name__).warning("SECOND-RUN-ONLY")
        get_logger(__name__).warning("FIRST-RUN-AGAIN")

    first_log = next(first_dir.glob("run-*.log")).read_text()
    second_log = next(second_dir.glob("run-*.log")).read_text()
    assert "FIRST-RUN-ONLY" in first_log
    assert "FIRST-RUN-AGAIN" in first_log
    assert "SECOND-RUN-ONLY" not in first_log
    assert "SECOND-RUN-ONLY" in second_log
    assert "FIRST-RUN-ONLY" not in second_log
    assert root_logger.handlers == handlers_before

    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    with pytest.raises(RuntimeError, match="run failed"):
        with scoped_run_file_logging(cfg, failed_dir):
            get_logger(__name__).warning("FAILED-RUN-ONLY")
            raise RuntimeError("run failed")
    assert "FAILED-RUN-ONLY" in next(failed_dir.glob("run-*.log")).read_text()
    assert root_logger.handlers == handlers_before


def test_atomic_writer_does_not_publish_or_leave_temp_file_on_failure(tmp_path):
    final_path = tmp_path / "annotations.parquet"

    def failing_writer(path):
        path.write_text("partial")
        raise RuntimeError("write failed")

    with pytest.raises(RuntimeError, match="write failed"):
        atomic_write_path(final_path, failing_writer)
    assert not final_path.exists()
    assert list(tmp_path.iterdir()) == []


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
    assert summary["English"].pop("n") == 0
    assert set(summary["English"].values()) == {"n/a"}
    assert summary["Multilingual"]["n"] == 2


def test_soft_elo_gap_uses_preference_magnitude():
    battles = _battles()
    top, df_top = select_top_models(battles, top_models=3)
    sample = sample_battles_per_model(df_top, top, battles_per_model=5, seed=0)
    sample["winner_llm"] = sample["winner"]
    sample["pref_llm"] = sample["winner"].map(
        {"model_a": 0.1, "model_b": 0.9, "tie": 0.5}
    )
    kwargs = {
        "n_battles_list": [2],
        "n_seeds": 2,
        "seed": 0,
        "exclude_ties": False,
        "soft": True,
    }
    soft = compute_elo_gap_summary(df_top, sample, top, **kwargs)
    neutral = sample.assign(pref_llm=0.5)
    neutral_soft = compute_elo_gap_summary(df_top, neutral, top, **kwargs)

    assert not soft.empty
    assert soft.loc[0, "mean"] != pytest.approx(neutral_soft.loc[0, "mean"])
