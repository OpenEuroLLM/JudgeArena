import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

import judgearena.benchmarks.mt_bench.runner as mt_bench_runner
import judgearena.datasets.mt_bench as mt_bench
from judgearena.config import RunConfig
from judgearena.prompts.registry import FASTCHAT_PAIRWISE_PROMPT_PRESET
from judgearena.tasks.registry import get_packaged_task


def test_mt_bench_adapter_normalizes_questions_and_references(monkeypatch, tmp_path):
    task = get_packaged_task("mt-bench")
    assert task is not None
    question_path = tmp_path / "question.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    question_path.write_text(
        '{"question_id": 1, "category": "math", "turns": ["Q1", "Q2"]}\n'
    )
    reference_path.write_text(
        '{"question_id": 1, "choices": [{"turns": ["R1", "R2"]}]}\n'
    )
    monkeypatch.setattr(
        mt_bench,
        "_download_mt_bench",
        lambda _task, _local_dir: (question_path, reference_path),
    )

    loaded = mt_bench.load_task_instructions(task, tmp_path)

    row = loaded.set_index("instruction_index").loc[1]
    assert row["instruction"] == row["turn_1"] == "Q1"
    assert row["turn_2"] == "Q2"
    assert row["reference_turn_2"] == "R2"
    assert row["category"] == "math"


def test_download_mt_bench_skips_question_download_if_cached(tmp_path, monkeypatch):
    question_path = tmp_path / "data" / "mt_bench" / "question.jsonl"
    question_path.parent.mkdir(parents=True, exist_ok=True)
    question_path.write_text('{"question_id": 1, "turns": ["Q1"]}\n')

    reference_path = tmp_path / "reference_answer" / "gpt-4.jsonl"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text('{"question_id": 1, "choices": [{"turns": ["A1"]}]}\n')

    calls = {"snapshot_download": 0}

    def _snapshot_download_stub(**_kwargs):
        calls["snapshot_download"] += 1

    monkeypatch.setattr(mt_bench, "snapshot_download", _snapshot_download_stub)
    monkeypatch.setattr(
        mt_bench, "_download_references", lambda _task, _local_dir: reference_path
    )

    downloaded_question_path, downloaded_reference_path = mt_bench.download_mt_bench(
        local_dir=tmp_path
    )

    assert downloaded_question_path == question_path
    assert downloaded_reference_path == reference_path
    assert calls["snapshot_download"] == 0


def test_load_mt_bench_model_answers_reads_cached_baseline_file(tmp_path):
    answer_path = tmp_path / "data" / "mt_bench" / "model_answer" / "gpt-4.jsonl"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(
        '{"question_id": 2, "choices": [{"turns": ["A2", "B2"]}]}\n'
        '{"question_id": 1, "choices": [{"turns": ["A1"]}]}\n'
    )

    df_answers = mt_bench.load_mt_bench_model_answers("gpt-4", local_dir=tmp_path)

    assert df_answers["instruction_index"].tolist() == [1, 2]
    assert df_answers["completion_turn_1"].tolist() == ["A1", "A2"]
    assert df_answers["completion_turn_2"].tolist() == ["", "B2"]


def test_generate_mt_bench_completions_uses_pregenerated_baseline(monkeypatch):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1", "Q2"], "turn_2": ["Q1b", "Q2b"]},
        index=pd.Index([1, 2], name="instruction_index"),
    )
    generated_calls = []

    monkeypatch.setattr(
        mt_bench_runner, "cache_function_dataframe", lambda fun, **_kwargs: fun()
    )

    def fake_generate_multiturn(**kwargs):
        generated_calls.append(kwargs)
        return pd.DataFrame(
            {
                "instruction_index": [1, 2],
                "completion_turn_1": ["Gen A1", "Gen A2"],
                "completion_turn_2": ["Gen B1", "Gen B2"],
            }
        )

    monkeypatch.setattr(mt_bench_runner, "generate_multiturn", fake_generate_multiturn)
    monkeypatch.setattr(
        mt_bench_runner,
        "load_mt_bench_model_answers",
        lambda model, n_instructions=None: (
            pd.DataFrame(
                {
                    "instruction_index": [2, 1],
                    "completion_turn_1": ["Base A2", "Base A1"],
                    "completion_turn_2": ["Base B2", "Base B1"],
                }
            )
            if model == "gpt-4"
            else None
        ),
    )

    cfg = RunConfig(
        task="mt-bench",
        model={
            "name": "VLLM/Qwen/Qwen3.5-9B",
            "baseline": "gpt-4",
            "max_out_tokens": 8192,
        },
        judge={
            "model": "Dummy/J",
            "battle_thinking_token_budget": 16384,
            "strip_thinking_before_judging": True,
        },
        generation={"n_instructions": 2},
    )

    completions_a, completions_b = mt_bench_runner._generate_mt_bench_completions(
        cfg=cfg,
        protocol=get_packaged_task("mt-bench").spec.protocol,
        questions_df=questions_df,
    )

    assert len(generated_calls) == 1
    call = generated_calls[0]
    assert call["model"] == "VLLM/Qwen/Qwen3.5-9B"
    assert call["thinking_token_budget"] == 8192
    assert call["strip_thinking_before_turn_2_prompt"] is True
    assert call["temperature_config"]["writing"] == 0.7
    assert call["temperature_config"]["math"] == 0.0
    assert completions_a.loc[1, "completion_turn_1"] == "Gen A1"
    assert completions_b.loc[1, "completion_turn_1"] == "Base A1"
    assert completions_b.loc[2, "completion_turn_2"] == "Base B2"


def test_generate_mt_bench_completions_reports_missing_baseline_rows(monkeypatch):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1", "Q2"], "turn_2": ["Q1b", "Q2b"]},
        index=pd.Index([1, 2], name="instruction_index"),
    )

    monkeypatch.setattr(
        mt_bench_runner,
        "load_mt_bench_model_answers",
        lambda model, n_instructions=None: pd.DataFrame(
            {
                "instruction_index": [1],
                "completion_turn_1": ["Base A1"],
                "completion_turn_2": ["Base B1"],
            }
        ),
    )

    cfg = RunConfig(
        task="mt-bench",
        model={"name": "gpt-4", "baseline": "gpt-4"},
        judge={"model": "Dummy/J"},
        generation={"n_instructions": 2},
    )

    with pytest.raises(ValueError, match="missing 1 question"):
        mt_bench_runner._generate_mt_bench_completions(
            cfg=cfg,
            protocol=get_packaged_task("mt-bench").spec.protocol,
            questions_df=questions_df,
        )


def test_save_mt_bench_results_writes_run_metadata(monkeypatch, tmp_path):
    captured = {}

    def fake_write_run_metadata(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run-metadata.v1.json"

    monkeypatch.setattr(
        mt_bench_runner, "write_run_metadata_safely", fake_write_run_metadata
    )
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "model-a", "baseline": "model-b"},
        judge={"model": "judge"},
    )
    started_at = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)

    mt_bench_runner._save_mt_bench_results(
        cfg=cfg,
        res_folder=tmp_path,
        result_name="mt-bench-test",
        results={"win_rate": 0.5, "preferences": [1.0]},
        annotations_df=pd.DataFrame([{"preference": 1.0}]),
        started_at_utc=started_at,
        input_payloads={"instruction_index": [1]},
        judge_system_prompt="system",
        judge_user_prompt_template="user",
    )

    assert (tmp_path / "mt-bench-test-annotations.csv").exists()
    assert (
        captured["entrypoint"]
        == "judgearena.benchmarks.mt_bench.runner.run_mt_bench_benchmark"
    )
    assert captured["input_payloads"] == {"instruction_index": [1]}
    assert captured["judge_system_prompt"] == "system"
    assert captured["judge_user_prompt_template"] == "user"
    assert captured["started_at_utc"] == started_at


def test_run_mt_bench_rejects_random_before_preparation(monkeypatch):
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "Dummy/model"},
        judge={
            "model": "Dummy/judge",
            "prompt_preset": FASTCHAT_PAIRWISE_PROMPT_PRESET,
            "swap_mode": "random",
        },
    )

    def unexpected_preparation(*_args, **_kwargs):
        pytest.fail("Unsupported swap mode must fail before preparing the run")

    monkeypatch.setattr(
        mt_bench_runner, "prepare_run_directory", unexpected_preparation
    )

    with pytest.raises(ValueError, match="MT-Bench supports only.*got 'random'"):
        mt_bench_runner.run_mt_bench_benchmark(cfg, get_packaged_task("mt-bench"))


def _stub_mt_bench_generation(monkeypatch, captured):
    questions = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )

    def answers(prefix):
        return pd.DataFrame(
            {"completion_turn_1": [f"{prefix}1"], "completion_turn_2": [f"{prefix}2"]},
            index=questions.index,
        )

    monkeypatch.setattr(
        mt_bench_runner, "load_instructions", lambda *_args, **_kwargs: questions
    )
    monkeypatch.setattr(
        mt_bench_runner,
        "_generate_mt_bench_completions",
        lambda *_args, **_kwargs: (answers("A"), answers("B")),
    )

    def make_model(**kwargs):
        captured["make_model"] = kwargs
        return object()

    monkeypatch.setattr(mt_bench_runner, "make_model", make_model)


def test_run_mt_bench_dispatches_prompt_override(monkeypatch, tmp_path):
    captured = {}
    _stub_mt_bench_generation(monkeypatch, captured)

    def run_preset(**kwargs):
        captured["preset"] = kwargs
        return pd.Series([0.0])

    monkeypatch.setattr(mt_bench_runner, "_run_mt_bench_preset", run_preset)
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "VLLM/example/model-a"},
        judge={"model": "VLLM/Judge", "prompt_preset": "default_with_explanation"},
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_runner.run_mt_bench_benchmark(cfg, get_packaged_task("mt-bench"))

    assert "temperature" not in captured["make_model"]
    assert (
        captured["preset"]["resolved_prompt"].preset_name == "default_with_explanation"
    )


def test_run_mt_bench_resolves_baseline_and_forwards_judge_controls(
    monkeypatch, tmp_path
):
    captured = {}
    _stub_mt_bench_generation(monkeypatch, captured)
    monkeypatch.setattr(
        mt_bench_runner, "_finalize_mt_bench_run", lambda **kwargs: kwargs["prefs"]
    )

    def fake_judge(**kwargs):
        captured["judge"] = kwargs
        return pd.Series([0.0], dtype=float), [], [], 0

    monkeypatch.setattr(mt_bench_runner, "judge_mt_bench_pairwise_fastchat", fake_judge)

    cfg = RunConfig(
        task="mt-bench",
        model={
            "name": "VLLM/example/model-a",
            "engine_kwargs": {"tensor_parallel_size": 1},
        },
        judge={
            "model": "VLLM/Judge",
            "strip_thinking_before_judging": True,
            "max_model_len": 65536,
            "engine_kwargs": {"tensor_parallel_size": 4},
        },
        generation={"n_instructions": 1, "truncate_judge_input_chars": 80000},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_runner.run_mt_bench_benchmark(cfg, get_packaged_task("mt-bench"))

    assert captured["judge"]["strip_thinking_before_judging"] is True
    assert cfg.model.baseline == "gpt-3.5-turbo"
    assert captured["make_model"]["temperature"] == 0.0
    assert captured["make_model"]["max_model_len"] == 65536
    assert captured["make_model"]["tensor_parallel_size"] == 4
    assert captured["judge"]["truncate_input_chars"] == 80000
    assert "math" in captured["judge"]["reference_categories"]
    assert captured["judge"]["prompt_preset"] == "default"


def test_mt_bench_finalization_uses_shared_grouped_metric(monkeypatch, tmp_path):
    task = get_packaged_task("mt-bench")
    assert task is not None
    monkeypatch.setattr(
        mt_bench_runner, "write_run_metadata_safely", lambda **_kwargs: None
    )
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "candidate", "baseline": "reference"},
        judge={"model": "judge"},
    )
    prompt = SimpleNamespace(
        metadata=lambda: {}, system_prompt=None, user_prompt_template="{instruction}"
    )
    index = pd.Index([1, 2], name="question_id")
    questions = pd.DataFrame(
        {"turn_1": ["q1", "q2"], "turn_2": ["q1b", "q2b"]}, index=index
    )
    completions_a = pd.DataFrame(
        {"completion_turn_1": ["a1", "a2"], "completion_turn_2": ["a1b", "a2b"]},
        index=index,
    )
    completions_b = pd.DataFrame(
        {"completion_turn_1": ["b1", "b2"], "completion_turn_2": ["b1b", "b2b"]},
        index=index,
    )
    preferences = pd.Series([0.0, 1.0, 0.0, 0.5])
    metadata = [
        {"question_id": 1, "category": "math", "turn": 1},
        {"question_id": 1, "category": "math", "turn": 2},
        {"question_id": 2, "category": "writing", "turn": 1},
        {"question_id": 2, "category": "writing", "turn": 2},
    ]

    returned = mt_bench_runner._finalize_mt_bench_run(
        cfg=cfg,
        protocol=task.spec.protocol,
        res_folder=tmp_path,
        result_name="result",
        prefs=preferences,
        annotations=[],
        combined_metadata=metadata,
        resolved_prompt=prompt,
        questions_df=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        started_at_utc=datetime.now(UTC),
    )

    assert returned.equals(preferences)
    saved = json.loads((tmp_path / "results-result.json").read_text())
    metric = saved["metrics"]["pairwise_win_rate"]
    assert metric["winrate"] == pytest.approx(0.625)
    assert [
        (item["group"], item["values"]["winrate"])
        for item in metric["groups"]["category"]
    ] == [("math", 0.5), ("writing", 0.75)]
    assert [
        (item["group"], item["values"]["winrate"]) for item in metric["groups"]["turn"]
    ] == [(1, 1.0), (2, 0.25)]


def test_run_mt_bench_rejects_logprob_parser_before_preparation(monkeypatch):
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "Dummy/model"},
        judge={
            "model": "Dummy/judge",
            "prompt_preset": "alpaca-eval",
            "top_logprobs": 5,
        },
    )

    def unexpected_preparation(*_args, **_kwargs):
        pytest.fail("Unsupported parser must fail before preparing the run")

    monkeypatch.setattr(
        mt_bench_runner, "prepare_run_directory", unexpected_preparation
    )

    with pytest.raises(ValueError, match="MT-Bench does not support.*logprobs"):
        mt_bench_runner.run_mt_bench_benchmark(cfg, get_packaged_task("mt-bench"))
