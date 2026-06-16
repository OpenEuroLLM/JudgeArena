from datetime import UTC, datetime

import pandas as pd
import pytest

import judgearena.instruction_dataset.mt_bench as mt_bench
import judgearena.mt_bench.mt_bench_utils as mt_bench_utils
import judgearena.utils.io as utils_io
from judgearena.config import RunConfig
from judgearena.prompts.registry import FASTCHAT_PAIRWISE_PROMPT_PRESET


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
        mt_bench,
        "_download_gpt4_references",
        lambda _local_dir: reference_path,
    )

    downloaded_question_path, downloaded_reference_path = mt_bench.download_mt_bench(
        local_dir=tmp_path
    )

    assert downloaded_question_path == question_path
    assert downloaded_reference_path == reference_path
    assert calls["snapshot_download"] == 0


def test_download_all_includes_mt_bench(tmp_path, monkeypatch):
    hf_datasets = []
    arena_hard_datasets = []
    calls = {"contexts": 0, "mt_bench": 0}

    monkeypatch.setattr(utils_io, "data_root", tmp_path)
    monkeypatch.setattr(
        utils_io,
        "download_hf",
        lambda name, local_path: hf_datasets.append((name, local_path)),
    )
    monkeypatch.setattr(
        utils_io,
        "download_arena_hard",
        lambda dataset, local_tables_path: arena_hard_datasets.append(
            (dataset, local_tables_path)
        ),
    )

    def _contexts_snapshot_stub(**_kwargs):
        calls["contexts"] += 1

    monkeypatch.setattr(utils_io, "snapshot_download", _contexts_snapshot_stub)
    monkeypatch.setattr(
        mt_bench,
        "download_mt_bench",
        lambda: calls.__setitem__("mt_bench", calls["mt_bench"] + 1),
    )

    utils_io.download_all()

    tables_dir = tmp_path / "tables"
    assert [name for name, _ in hf_datasets] == [
        "alpaca-eval",
        "m-arena-hard-v0.1",
        "m-arena-hard-v2.0",
    ]
    assert arena_hard_datasets == [
        ("arena-hard-v0.1", tables_dir),
        ("arena-hard-v2.0", tables_dir),
    ]
    assert calls["contexts"] == 1
    assert calls["mt_bench"] == 1


def test_load_mt_bench_model_answers_reads_cached_baseline_file(tmp_path):
    answer_path = tmp_path / "data" / "mt_bench" / "model_answer" / "gpt-4.jsonl"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(
        '{"question_id": 2, "choices": [{"turns": ["A2", "B2"]}]}\n'
        '{"question_id": 1, "choices": [{"turns": ["A1"]}]}\n'
    )

    df_answers = mt_bench.load_mt_bench_model_answers("gpt-4", local_dir=tmp_path)

    assert df_answers.to_dict(orient="records") == [
        {
            "instruction_index": 1,
            "completion_turn_1": "A1",
            "completion_turn_2": "",
        },
        {
            "instruction_index": 2,
            "completion_turn_1": "A2",
            "completion_turn_2": "B2",
        },
    ]


def test_generate_mt_bench_completions_uses_pregenerated_baseline(monkeypatch):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1", "Q2"], "turn_2": ["Q1b", "Q2b"]},
        index=pd.Index([1, 2], name="instruction_index"),
    )
    generated_models = []

    monkeypatch.setattr(
        mt_bench_utils, "cache_function_dataframe", lambda fun, **_kwargs: fun()
    )

    def fake_generate_multiturn(**kwargs):
        generated_models.append(kwargs["model"])
        return pd.DataFrame(
            {
                "instruction_index": [1, 2],
                "completion_turn_1": ["Gen A1", "Gen A2"],
                "completion_turn_2": ["Gen B1", "Gen B2"],
            }
        )

    monkeypatch.setattr(mt_bench_utils, "generate_multiturn", fake_generate_multiturn)
    monkeypatch.setattr(
        mt_bench_utils,
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
            "name": "VLLM/example/model-a",
            "baseline": "gpt-4",
            "engine_kwargs": {"gpu_memory_utilization": 0.7},
        },
        judge={"model": "Dummy/J"},
        generation={"n_instructions": 2},
    )

    completions_a, completions_b = mt_bench_utils._generate_mt_bench_completions(
        cfg=cfg,
        questions_df=questions_df,
        ignore_cache=False,
    )

    assert generated_models == ["VLLM/example/model-a"]
    assert completions_a.loc[1, "completion_turn_1"] == "Gen A1"
    assert completions_b.loc[1, "completion_turn_1"] == "Base A1"
    assert completions_b.loc[2, "completion_turn_2"] == "Base B2"


def test_generate_mt_bench_completions_reports_missing_baseline_rows(monkeypatch):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1", "Q2"], "turn_2": ["Q1b", "Q2b"]},
        index=pd.Index([1, 2], name="instruction_index"),
    )

    monkeypatch.setattr(
        mt_bench_utils,
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
        mt_bench_utils._generate_mt_bench_completions(
            cfg=cfg,
            questions_df=questions_df,
            ignore_cache=False,
        )


def test_save_mt_bench_results_writes_run_metadata(monkeypatch, tmp_path):
    captured = {}

    def fake_write_run_metadata(**kwargs):
        captured.update(kwargs)
        return tmp_path / "run-metadata.v1.json"

    monkeypatch.setattr(
        mt_bench_utils,
        "write_run_metadata",
        fake_write_run_metadata,
    )
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "model-a", "baseline": "model-b"},
        judge={"model": "judge"},
    )
    started_at = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)

    mt_bench_utils._save_mt_bench_results(
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

    assert (tmp_path / "config.yaml").exists()
    assert (tmp_path / "mt-bench-test-annotations.csv").exists()
    assert (tmp_path / "results-mt-bench-test.json").exists()
    assert captured["entrypoint"] == "judgearena.mt_bench.mt_bench_utils.run_mt_bench"
    assert captured["input_payloads"] == {"instruction_index": [1]}
    assert captured["judge_system_prompt"] == "system"
    assert captured["judge_user_prompt_template"] == "user"
    assert captured["started_at_utc"] == started_at


def test_run_mt_bench_resolves_native_baseline_and_judge_controls(
    monkeypatch, tmp_path
):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured = {}

    monkeypatch.setattr(
        mt_bench_utils,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_generate_mt_bench_completions",
        lambda cfg, questions_df, ignore_cache: (
            pd.DataFrame(
                {"completion_turn_1": ["A1"], "completion_turn_2": ["A2"]},
                index=questions_df.index,
            ),
            pd.DataFrame(
                {"completion_turn_1": ["B1"], "completion_turn_2": ["B2"]},
                index=questions_df.index,
            ),
        ),
    )

    def fake_make_model(**kwargs):
        captured["make_model"] = kwargs
        return object()

    monkeypatch.setattr(mt_bench_utils, "make_model", fake_make_model)

    def fake_run_mt_bench_fastchat(**kwargs):
        captured["fastchat"] = kwargs
        return pd.Series([0.0], dtype=float)

    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_fastchat",
        fake_run_mt_bench_fastchat,
    )

    cfg = RunConfig(
        task="mt-bench",
        model={
            "name": "VLLM/example/model-a",
            "baseline": None,
            "engine_kwargs": {"tensor_parallel_size": 1},
        },
        judge={
            "model": "VLLM/Judge",
            "max_model_len": 65536,
            "engine_kwargs": {"tensor_parallel_size": 4},
        },
        generation={"n_instructions": 1, "truncate_judge_input_chars": 80000},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_utils.run_mt_bench(
        cfg,
        ignore_cache=False,
        res_folder=tmp_path,
        result_name="mt-bench-test",
    )

    assert cfg.model.baseline == "gpt-4"
    assert captured["make_model"]["max_model_len"] == 65536
    assert captured["make_model"]["tensor_parallel_size"] == 4
    assert captured["fastchat"]["cfg"].generation.truncate_judge_input_chars == 80000
    assert captured["fastchat"]["fastchat_prompt_preset"] == "default"
    assert captured["fastchat"]["resolved_prompt"].preset_name == (
        FASTCHAT_PAIRWISE_PROMPT_PRESET
    )


def test_run_mt_bench_defaults_to_delegated_fastchat(monkeypatch, tmp_path):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured = {}

    monkeypatch.setattr(
        mt_bench_utils,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_generate_mt_bench_completions",
        lambda cfg, questions_df, ignore_cache: (
            pd.DataFrame(
                {"completion_turn_1": ["A1"], "completion_turn_2": ["A2"]},
                index=questions_df.index,
            ),
            pd.DataFrame(
                {"completion_turn_1": ["B1"], "completion_turn_2": ["B2"]},
                index=questions_df.index,
            ),
        ),
    )

    def fake_make_model(**kwargs):
        captured["make_model"] = kwargs
        return object()

    monkeypatch.setattr(mt_bench_utils, "make_model", fake_make_model)

    def fake_run_mt_bench_fastchat(**kwargs):
        captured["fastchat"] = kwargs
        return pd.Series([0.0], dtype=float)

    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_fastchat",
        fake_run_mt_bench_fastchat,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_preset",
        lambda **_kwargs: pytest.fail("preset path should not run"),
    )

    cfg = RunConfig(
        task="mt-bench",
        model={"name": "VLLM/example/model-a"},
        judge={"model": "VLLM/Judge"},
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_utils.run_mt_bench(
        cfg,
        ignore_cache=False,
        res_folder=tmp_path,
        result_name="mt-bench-test",
    )

    assert cfg.model.baseline == "gpt-4"
    assert captured["make_model"]["temperature"] == 0.0
    assert captured["fastchat"]["fastchat_prompt_preset"] == "default"
    assert captured["fastchat"]["resolved_prompt"].preset_name == (
        FASTCHAT_PAIRWISE_PROMPT_PRESET
    )


def test_run_mt_bench_concrete_prompt_preset_uses_preset_judging(monkeypatch, tmp_path):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )

    monkeypatch.setattr(
        mt_bench_utils,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_generate_mt_bench_completions",
        lambda cfg, questions_df, ignore_cache: (
            pd.DataFrame(
                {"completion_turn_1": ["A1"], "completion_turn_2": ["A2"]},
                index=questions_df.index,
            ),
            pd.DataFrame(
                {"completion_turn_1": ["B1"], "completion_turn_2": ["B2"]},
                index=questions_df.index,
            ),
        ),
    )

    def fake_make_model(**kwargs):
        captured["make_model"] = kwargs
        return object()

    def fake_run_mt_bench_preset(**kwargs):
        captured["preset"] = kwargs
        return pd.Series([0.0], dtype=float)

    captured = {}
    monkeypatch.setattr(mt_bench_utils, "make_model", fake_make_model)
    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_preset",
        fake_run_mt_bench_preset,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_fastchat",
        lambda **_kwargs: pytest.fail("fastchat path should not run"),
    )

    cfg = RunConfig(
        task="mt-bench",
        model={"name": "VLLM/example/model-a"},
        judge={"model": "VLLM/Judge", "prompt_preset": "default_with_explanation"},
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_utils.run_mt_bench(
        cfg,
        ignore_cache=False,
        res_folder=tmp_path,
        result_name="mt-bench-test",
    )

    assert captured["preset"]["resolved_prompt"].preset_name == (
        "default_with_explanation"
    )
    assert "temperature" not in captured["make_model"]


def test_generate_mt_bench_completions_forwards_thinking_controls(monkeypatch):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        mt_bench_utils, "cache_function_dataframe", lambda fun, **_kwargs: fun()
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "load_mt_bench_model_answers",
        lambda model, n_instructions=None: None,
    )

    def fake_generate_multiturn(**kwargs):
        captured[kwargs["model"]] = kwargs
        return pd.DataFrame(
            {
                "instruction_index": [1],
                "completion_turn_1": ["A1"],
                "completion_turn_2": ["B1"],
            }
        )

    monkeypatch.setattr(mt_bench_utils, "generate_multiturn", fake_generate_multiturn)

    cfg = RunConfig(
        task="mt-bench",
        model={
            "name": "VLLM/Qwen/Qwen3.5-9B",
            "baseline": "VLLM/meta-llama/Llama-3.1-8B",
            "max_out_tokens": 8192,
        },
        judge={
            "model": "Dummy/J",
            "battle_thinking_token_budget": 16384,
            "strip_thinking_before_judging": True,
        },
        generation={"n_instructions": 1},
    )

    mt_bench_utils._generate_mt_bench_completions(
        cfg=cfg,
        questions_df=questions_df,
        ignore_cache=False,
    )

    thinking_call = captured["VLLM/Qwen/Qwen3.5-9B"]
    plain_call = captured["VLLM/meta-llama/Llama-3.1-8B"]

    assert thinking_call["strip_thinking_before_turn_2_prompt"] is True
    assert thinking_call["thinking_token_budget"] == 8192
    assert plain_call["strip_thinking_before_turn_2_prompt"] is True
    assert "thinking_token_budget" not in plain_call


def test_run_mt_bench_forwards_strip_thinking_to_fastchat_judge(monkeypatch, tmp_path):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        mt_bench_utils,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_generate_mt_bench_completions",
        lambda cfg, questions_df, ignore_cache: (
            pd.DataFrame(
                {"completion_turn_1": ["A1"], "completion_turn_2": ["A2"]},
                index=questions_df.index,
            ),
            pd.DataFrame(
                {"completion_turn_1": ["B1"], "completion_turn_2": ["B2"]},
                index=questions_df.index,
            ),
        ),
    )
    monkeypatch.setattr(mt_bench_utils, "make_model", lambda **kwargs: object())
    monkeypatch.setattr(
        mt_bench_utils, "_finalize_mt_bench_run", lambda **kwargs: kwargs["prefs"]
    )

    def fake_judge(**kwargs):
        captured["judge"] = kwargs
        return pd.Series([0.0], dtype=float), [], [], 0

    monkeypatch.setattr(mt_bench_utils, "judge_mt_bench_pairwise_fastchat", fake_judge)

    cfg = RunConfig(
        task="mt-bench",
        model={"name": "VLLM/example/model-a"},
        judge={"model": "VLLM/Judge", "strip_thinking_before_judging": True},
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_utils.run_mt_bench(
        cfg,
        ignore_cache=False,
        res_folder=tmp_path,
        result_name="mt-bench-test",
    )

    assert captured["judge"]["strip_thinking_before_judging"] is True
