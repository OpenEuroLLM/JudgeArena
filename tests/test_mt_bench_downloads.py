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

    assert loaded.to_dict(orient="records") == [
        {
            "instruction_index": 1,
            "category": "math",
            "turn_1": "Q1",
            "turn_2": "Q2",
            "reference_turn_1": "R1",
            "reference_turn_2": "R2",
            "instruction": "Q1",
        }
    ]


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
        "_download_references",
        lambda _task, _local_dir: reference_path,
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
        mt_bench_runner, "cache_function_dataframe", lambda fun, **_kwargs: fun()
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
            "name": "VLLM/example/model-a",
            "baseline": "gpt-4",
            "engine_kwargs": {"gpu_memory_utilization": 0.7},
        },
        judge={"model": "Dummy/J"},
        generation={"n_instructions": 2},
    )

    completions_a, completions_b = mt_bench_runner._generate_mt_bench_completions(
        cfg=cfg,
        protocol=get_packaged_task("mt-bench").spec.protocol,
        questions_df=questions_df,
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
        mt_bench_runner,
        "write_run_metadata_safely",
        fake_write_run_metadata,
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


def test_run_mt_bench_resolves_native_baseline_and_judge_controls(
    monkeypatch, tmp_path
):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured = {}

    monkeypatch.setattr(
        mt_bench_runner,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_runner,
        "_generate_mt_bench_completions",
        lambda cfg, protocol, questions_df: (
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

    monkeypatch.setattr(mt_bench_runner, "make_model", fake_make_model)

    def fake_run_mt_bench_fastchat(**kwargs):
        captured["fastchat"] = kwargs
        return pd.Series([0.0], dtype=float)

    monkeypatch.setattr(
        mt_bench_runner,
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

    mt_bench_runner.run_mt_bench_benchmark(cfg, get_packaged_task("mt-bench"))

    assert cfg.model.baseline == "gpt-4"
    assert captured["make_model"]["max_model_len"] == 65536
    assert captured["make_model"]["tensor_parallel_size"] == 4
    assert captured["fastchat"]["cfg"].generation.truncate_judge_input_chars == 80000
    assert captured["fastchat"]["protocol"].judge.fastchat_prompt_preset == "default"
    assert captured["fastchat"]["resolved_prompt"].preset_name == (
        FASTCHAT_PAIRWISE_PROMPT_PRESET
    )


def _stub_mt_bench_dispatch(monkeypatch, captured):
    questions = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )

    def answers(prefix):
        return pd.DataFrame(
            {
                "completion_turn_1": [f"{prefix}1"],
                "completion_turn_2": [f"{prefix}2"],
            },
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

    def dispatch(path, kwargs):
        captured.setdefault("dispatch", []).append(path)
        captured[path] = kwargs
        return pd.Series([0.0], dtype=float)

    monkeypatch.setattr(
        mt_bench_runner,
        "_run_mt_bench_fastchat",
        lambda **kwargs: dispatch("fastchat", kwargs),
    )
    monkeypatch.setattr(
        mt_bench_runner,
        "_run_mt_bench_preset",
        lambda **kwargs: dispatch("preset", kwargs),
    )


@pytest.mark.parametrize(
    ("prompt_preset", "expected_path"),
    [(None, "fastchat"), ("default_with_explanation", "preset")],
)
def test_run_mt_bench_dispatches_defaults_and_prompt_overrides(
    monkeypatch, tmp_path, prompt_preset, expected_path
):
    captured = {}
    _stub_mt_bench_dispatch(monkeypatch, captured)
    judge = {"model": "VLLM/Judge"}
    if prompt_preset is not None:
        judge["prompt_preset"] = prompt_preset
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "VLLM/example/model-a"},
        judge=judge,
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_runner.run_mt_bench_benchmark(cfg, get_packaged_task("mt-bench"))

    assert cfg.model.baseline == "gpt-4"
    assert captured["dispatch"] == [expected_path]
    if expected_path == "fastchat":
        assert captured["make_model"]["temperature"] == 0.0
        assert captured["fastchat"]["resolved_prompt"].preset_name == (
            FASTCHAT_PAIRWISE_PROMPT_PRESET
        )
    else:
        assert "temperature" not in captured["make_model"]
        assert captured["preset"]["resolved_prompt"].preset_name == prompt_preset


def test_generate_mt_bench_completions_forwards_thinking_controls(monkeypatch):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        mt_bench_runner, "cache_function_dataframe", lambda fun, **_kwargs: fun()
    )
    monkeypatch.setattr(
        mt_bench_runner,
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

    monkeypatch.setattr(mt_bench_runner, "generate_multiturn", fake_generate_multiturn)

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

    mt_bench_runner._generate_mt_bench_completions(
        cfg=cfg,
        protocol=get_packaged_task("mt-bench").spec.protocol,
        questions_df=questions_df,
    )

    thinking_call = captured["VLLM/Qwen/Qwen3.5-9B"]
    plain_call = captured["VLLM/meta-llama/Llama-3.1-8B"]

    assert thinking_call["strip_thinking_before_turn_2_prompt"] is True
    assert thinking_call["thinking_token_budget"] == 8192
    assert thinking_call["temperature_config"]["writing"] == 0.7
    assert thinking_call["temperature_config"]["math"] == 0.0
    assert plain_call["strip_thinking_before_turn_2_prompt"] is True
    assert "thinking_token_budget" not in plain_call


def test_run_mt_bench_forwards_strip_thinking_to_fastchat_judge(monkeypatch, tmp_path):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        mt_bench_runner,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_runner,
        "_generate_mt_bench_completions",
        lambda cfg, protocol, questions_df: (
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
    monkeypatch.setattr(mt_bench_runner, "make_model", lambda **kwargs: object())
    monkeypatch.setattr(
        mt_bench_runner, "_finalize_mt_bench_run", lambda **kwargs: kwargs["prefs"]
    )

    def fake_judge(**kwargs):
        captured["judge"] = kwargs
        return pd.Series([0.0], dtype=float), [], [], 0

    monkeypatch.setattr(mt_bench_runner, "judge_mt_bench_pairwise_fastchat", fake_judge)

    cfg = RunConfig(
        task="mt-bench",
        model={"name": "VLLM/example/model-a"},
        judge={"model": "VLLM/Judge", "strip_thinking_before_judging": True},
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path)},
    )

    mt_bench_runner.run_mt_bench_benchmark(cfg, get_packaged_task("mt-bench"))

    assert captured["judge"]["strip_thinking_before_judging"] is True
    assert captured["judge"]["reference_categories"] == (
        "math",
        "reasoning",
        "coding",
        "arena-hard-200",
    )


def test_mt_bench_finalization_uses_shared_grouped_metric(monkeypatch, tmp_path):
    task = get_packaged_task("mt-bench")
    assert task is not None
    captured = {}

    class CapturingReport:
        def __init__(self, **values):
            captured.update(values)

        def to_dict(self):
            return dict(captured)

        def render(self):
            return None

        def save(self, _path):
            return None

    monkeypatch.setattr(mt_bench_runner, "BattleReport", CapturingReport)
    monkeypatch.setattr(
        mt_bench_runner, "_save_mt_bench_results", lambda **_kwargs: None
    )
    calculate_metrics = mt_bench_runner.calculate_metrics

    def capture_battles(battles, metrics):
        captured["battles"] = battles.copy()
        return calculate_metrics(battles, metrics)

    monkeypatch.setattr(mt_bench_runner, "calculate_metrics", capture_battles)
    cfg = SimpleNamespace(
        task="mt-bench",
        model=SimpleNamespace(name="candidate", baseline="reference"),
        judge=SimpleNamespace(
            model="judge",
            battle_thinking_token_budget=None,
            strip_thinking_before_judging=False,
        ),
    )
    prompt = SimpleNamespace(
        metadata=lambda: {},
        system_prompt=None,
        user_prompt_template="{instruction}",
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
    metric = captured["metrics"]["pairwise_win_rate"]
    assert metric["winrate"] == pytest.approx(0.625)
    assert [item["group"] for item in metric["groups"]["category"]] == [
        "math",
        "writing",
    ]
    assert [item["group"] for item in metric["groups"]["turn"]] == [1, 2]
    assert {
        "instruction_index",
        "model",
        "baseline",
        "completion_model",
        "completion_baseline",
        "orientation",
        "pref",
    } <= set(captured["battles"])
    assert captured["battles"]["instruction_index"].tolist() == [
        "1:turn-1",
        "1:turn-2",
        "2:turn-1",
        "2:turn-2",
    ]
