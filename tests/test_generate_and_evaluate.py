import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

import judgearena.benchmarks.evaluation as generate_and_evaluate
import judgearena.benchmarks.execution as benchmark_execution
from judgearena.benchmarks.elo.rating import fit_bradley_terry
from judgearena.benchmarks.evaluation import run_evaluation
from judgearena.benchmarks.pairwise.baselines import (
    BaselinePlan,
    resolve_baseline_plan,
)
from judgearena.benchmarks.pairwise.judging import prepare_pairwise_judging
from judgearena.cache.sqlite import JUDGEMENT_DB_NAME, JudgementCache
from judgearena.config import RunConfig
from judgearena.datasets.data import CompletionData
from judgearena.models import InferenceResult, prepare_model
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import MetricSpec, ScoringSpec


def _cfg(
    *,
    task: str,
    model_A: str,
    model_B: str | None = None,
    judge_model: str,
    n_instructions: int | None = None,
    swap_mode: str = "fixed",
    result_folder: str = "results",
    truncate_judge_input_chars: int | None = None,
    max_judge_model_len: int | None = None,
    engine_kwargs: dict | None = None,
    judge_engine_kwargs: dict | None = None,
) -> RunConfig:
    return RunConfig(
        task=task,
        model={
            "name": model_A,
            "baseline": model_B,
            "engine_kwargs": engine_kwargs or {},
        },
        judge={
            "model": judge_model,
            "swap_mode": swap_mode,
            "max_model_len": max_judge_model_len,
            "engine_kwargs": judge_engine_kwargs or {},
        },
        generation={
            "n_instructions": n_instructions,
            "truncate_judge_input_chars": truncate_judge_input_chars,
        },
        run={"result_folder": result_folder, "store_root": f"{result_folder}/cache"},
    )


@pytest.fixture(autouse=True)
def mock_external_data_and_cache(monkeypatch):
    instructions = pd.DataFrame(
        {
            "instruction": [f"Synthetic instruction {i}" for i in range(20)],
        },
        index=pd.Index(range(20), name="instruction_index"),
    )

    def load_data(task):
        frame = instructions
        if task.selection is not None:
            frame = frame.assign(**{task.selection.selector: task.selection.values[0]})
        return CompletionData(frame)

    monkeypatch.setattr(generate_and_evaluate, "load_task_data", load_data)


def _mock_alpaca_judge(monkeypatch, messages) -> dict[str, object]:
    from judgearena.benchmarks.pairwise.scoring import alpaca_eval

    captured: dict[str, object] = {}

    class FakeJudge:
        def batch(self, inputs, **_kwargs):
            return messages

    def make_fake_judge(**kwargs):
        captured.update(kwargs)
        return FakeJudge()

    monkeypatch.setattr(benchmark_execution, "prepare_model", make_fake_judge)
    monkeypatch.setattr(
        alpaca_eval,
        "_length_controlled_metrics",
        lambda *_args, **_kwargs: {
            "length_controlled_winrate": 50.0,
            "lc_standard_error": 1.0,
            "win_rate": 50.0,
        },
    )
    return captured


def _instructions(ids: list[str], categories: list[str] | None = None) -> pd.DataFrame:
    data = {"instruction": list(ids)}
    if categories is not None:
        data["category"] = list(categories)
    return pd.DataFrame(data, index=pd.Index(ids, name="instruction_index"))


def test_resolve_plan_v01_flat_default():
    plan = resolve_baseline_plan(
        task_id="arena-hard-v0.1",
        task=get_packaged_task("arena-hard-v0.1"),
        runtime_baseline=None,
        instructions=_instructions(["q1", "q2"]),
    )
    assert plan.is_single_model
    assert plan.single_model == "gpt-4-0314"


def test_resolve_plan_v20_routes_per_category():
    plan = resolve_baseline_plan(
        task_id="arena-hard-v2.0",
        task=get_packaged_task("arena-hard-v2.0"),
        runtime_baseline=None,
        instructions=_instructions(
            ["qh", "qc"],
            categories=["hard_prompt", "creative_writing"],
        ),
    )
    assert not plan.is_single_model
    assert plan.baseline_by_index.loc["qh"] == "o3-mini-2025-01-31"
    assert plan.baseline_by_index.loc["qc"] == "gemini-2.0-flash-001"


def test_resolve_plan_alpaca_eval_uses_native_baseline():
    plan = resolve_baseline_plan(
        task_id="alpaca-eval",
        task=get_packaged_task("alpaca-eval"),
        runtime_baseline=None,
        instructions=_instructions(["q1", "q2"]),
    )
    assert plan.is_single_model
    assert plan.single_model == "gpt4_1106_preview"


def test_resolve_plan_explicit_model_b_overrides_native():
    plan = resolve_baseline_plan(
        task_id="arena-hard-v2.0",
        task=get_packaged_task("arena-hard-v2.0"),
        runtime_baseline="override",
        instructions=_instructions(
            ["q1", "q2"],
            categories=["hard_prompt", "creative_writing"],
        ),
    )
    assert plan.is_single_model
    assert plan.single_model == "override"


def test_resolve_plan_task_without_native_baseline_requires_model_b():
    with pytest.raises(ValueError, match="baseline"):
        resolve_baseline_plan(
            task_id="fluency-french",
            task=None,
            runtime_baseline=None,
            instructions=_instructions(["q1"]),
        )


def test_resolve_plan_v20_missing_category_raises():
    with pytest.raises(ValueError, match="category"):
        resolve_baseline_plan(
            task_id="arena-hard-v2.0",
            task=get_packaged_task("arena-hard-v2.0"),
            runtime_baseline=None,
            instructions=_instructions(["q1"]),
        )


def test_resolve_plan_v20_unknown_category_raises():
    with pytest.raises(ValueError, match="brand_new"):
        resolve_baseline_plan(
            task_id="arena-hard-v2.0",
            task=get_packaged_task("arena-hard-v2.0"),
            runtime_baseline=None,
            instructions=_instructions(["q1"], categories=["brand_new"]),
        )


def test_baseline_plan_flat_repeats_model():
    plan = BaselinePlan.flat("b", index=pd.Index(["a", "b"]))
    assert plan.is_single_model
    assert plan.baseline_by_index.tolist() == ["b", "b"]


def test_baseline_plan_per_row_preserves_order():
    series = pd.Series(["m1", "m2"], index=["a", "b"], name="model_B")
    plan = BaselinePlan.per_row(series)
    assert not plan.is_single_model
    assert plan.unique_models == ["m1", "m2"]


@pytest.mark.parametrize(
    "task",
    [
        "alpaca-eval-ja",
        "arena-hard-v2.0-ja",
        "arena-hard-v0.1-ja",
        "fluency-french",
        "m-arena-hard-v0.1-EU",
        "m-arena-hard-v2.0-EU",
    ],
)
def test_generate_and_evaluate_context_completion(task: str, tmp_path):
    prefs = run_evaluation(
        _cfg(
            task=task,
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            result_folder=str(tmp_path),
            # default for swap_mode is "fixed"
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref >= 0.9


@pytest.mark.parametrize(
    ("generation", "expected_limit"),
    [({}, None), ({"truncate_all_input_chars": 128}, 128)],
)
def test_official_generation_receives_full_prompt_unless_explicitly_capped(
    monkeypatch, tmp_path, generation, expected_limit
):
    import judgearena.generate as generation_module

    instruction = "x" * 9000 + "END"
    instructions = pd.DataFrame(
        {"instruction": [instruction]}, index=pd.Index([0], name="instruction_index")
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_task_data",
        lambda *_args, **_kwargs: CompletionData(instructions=instructions),
    )
    seen_inputs = []

    def capture_inference(*, inputs, **_kwargs):
        seen_inputs.extend(prompt.to_messages()[-1].content for prompt in inputs)
        return ["Generated answer"] * len(inputs)

    monkeypatch.setattr(generation_module, "do_inference", capture_inference)
    cfg = RunConfig(
        task="arena-hard-v0.1",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={"model": "Dummy/[[A=B]]"},
        generation=generation,
        run={"result_folder": str(tmp_path), "use_tqdm": False},
    )

    run_evaluation(cfg)

    expected = instruction if expected_limit is None else instruction[:expected_limit]
    assert seen_inputs == [expected, expected]


def test_generate_and_evaluate_correct_order_bias(tmp_path):
    """Test the correction for model order bias.

    In this test, a judge that is totally biased towards model B should be corrected to be neutral.
    Since the judge favors model B regardless of the order and the completions, the average
    preference should be 0.5.
    """
    prefs = run_evaluation(
        _cfg(
            task="alpaca-eval-ja",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            swap_mode="both",
            result_folder=str(tmp_path),
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref == 0.5


def _task_with_metric(metric):
    task = get_packaged_task("alpaca-eval-ja")
    protocol = task.spec.protocol.model_copy(
        update={"scoring": ScoringSpec(metrics=(metric,))}
    )
    return replace(task, spec=task.spec.model_copy(update={"protocol": protocol}))


def test_pairwise_bootstraps_use_run_seed(monkeypatch, tmp_path):
    cfg = _cfg(
        task="alpaca-eval-ja",
        model_A="Dummy/no answer",
        model_B="Dummy/x",
        judge_model="Dummy/score A: 0 score B: 10",
        n_instructions=2,
        result_folder=str(tmp_path),
    )
    monkeypatch.setattr(
        benchmark_execution,
        "prepare_model",
        lambda **_kwargs: FakeListLLM(
            responses=["score A: 10 score B: 0", "score A: 0 score B: 10"]
        ),
    )
    cfg.run.seed = 17
    task = _task_with_metric(
        MetricSpec(metric="bradley_terry", parameters={"n_bootstraps": 3})
    )
    prefs = run_evaluation(cfg, task)

    battles = pd.DataFrame(
        {"model_a": cfg.model.name, "model_b": cfg.model.baseline, "pref": prefs}
    )
    rng = np.random.default_rng(17)
    expected = [
        fit_bradley_terry(
            battles.sample(
                n=len(battles), replace=True, random_state=int(rng.integers(0, 2**31))
            )
        )
        for _ in range(3)
    ]
    saved = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    assert saved["metrics"]["bradley_terry"]["bootstrap_ratings"] == expected


@pytest.mark.parametrize("failure", [None, "calculate", "render"])
def test_run_writes_roundtrippable_config(tmp_path, monkeypatch, caplog, failure):
    from judgearena.benchmarks.pairwise.scoring import PairwiseWinRateMetric
    from judgearena.config import load_config

    if failure:

        def fail(*args, **kwargs):
            raise RuntimeError("scorer unavailable")

        monkeypatch.setattr(PairwiseWinRateMetric, failure, fail)

    prefs = run_evaluation(
        _cfg(
            task="alpaca-eval-ja",
            model_A="Dummy/no answer",
            model_B="Dummy/x",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=2,
            result_folder=str(tmp_path),
        ),
        _task_with_metric(
            MetricSpec(metric="pairwise_win_rate", breakdown_by=("model_a",))
        ),
    )
    assert len(prefs) == 2 and (prefs > 0.5).all()
    assert len(pd.read_csv(next(tmp_path.glob("*/*-annotations.csv")))) == 2
    written = list(tmp_path.glob("*/config.yaml"))
    assert written, "config.yaml not written"
    result = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    if failure == "calculate":
        assert result["metrics"] == {
            "pairwise_win_rate": {"error": "scorer unavailable"}
        }
    else:
        metric = result["metrics"]["pairwise_win_rate"]
        assert metric["num_battles"] == 2
        assert metric["groups"]["model_a"][0]["group"] == "Dummy/no answer"
    if failure:
        assert "scorer unavailable" in caplog.text
    assert "winrate" not in result
    reloaded = load_config(written[0])
    assert reloaded.task == "alpaca-eval-ja"
    assert reloaded.model.name == "Dummy/no answer"


@pytest.mark.parametrize(
    ("swap_mode", "expected_prefs"),
    [
        ("fixed", [0.25, np.nan, 0.25]),
        ("random", [0.25, np.nan, 0.75]),
        ("both", [0.25, np.nan, 0.25, 0.25, 0.75, 0.25]),
    ],
)
def test_run_evaluation_routes_arena_v2_baselines_and_prompts(
    monkeypatch, tmp_path, swap_mode, expected_prefs
):
    raw = ["[[A>B]]", "no verdict", "[[A>B]]"]
    if swap_mode == "both":
        raw = ["[[A>B]]", "no verdict", "[[B>A]]", "[[A>B]]", "[[A>B]]", "[[B>A]]"]

    def build_judge(cfg, **kwargs):
        model = prepare_model(
            "Dummy/judge", cache=benchmark_execution.build_judgement_cache(cfg)
        )
        model.factory = lambda: FakeListLLM(responses=raw)
        return model

    monkeypatch.setattr(generate_and_evaluate, "build_judge", build_judge)

    instructions = pd.DataFrame(
        {
            "instruction": ["hard one", "creative", "hard two", "not evaluated"],
            "instruction_id": ["source-h1", "source-c", "source-h2", "source-unused"],
            "source_instruction_index": ["qc", "qh1", "qh2", "unused"],
            "category": [
                "hard_prompt",
                "creative_writing",
                "hard_prompt",
                "hard_prompt",
            ],
        },
        index=pd.Index(["qh1", "qc", "qh2", "unused"], name="instruction_index"),
    )
    outputs = pd.DataFrame(
        {
            "instruction_index": ["qh1", "qc", "qh2", "qh1", "qc", "qh2"],
            "model": ["candidate"] * 3
            + [
                "o3-mini-2025-01-31",
                "gemini-2.0-flash-001",
                "o3-mini-2025-01-31",
            ],
            "completion": [
                "candidate hard one",
                "candidate creative",
                "candidate hard two",
                "baseline hard one",
                "baseline creative",
                "baseline hard two",
            ],
        }
    ).iloc[::-1]
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_task_data",
        lambda task, n_instructions=None: CompletionData(instructions, outputs),
    )
    prefs = run_evaluation(
        RunConfig(
            task="arena-hard-v2.0",
            model={"name": "candidate"},
            judge={"model": "Dummy/judge", "swap_mode": swap_mode},
            generation={"n_instructions": 3},
            run={"result_folder": str(tmp_path), "store_root": str(tmp_path / "cache")},
        )
    )

    assert prefs.tolist() == pytest.approx(expected_prefs, nan_ok=True)
    ids = ["qh1", "qh2", "qc"] * (2 if swap_mode == "both" else 1)
    annotations = pd.read_csv(next(tmp_path.glob("*/*-annotations.csv")))
    assert annotations["instruction_index"].tolist() == ids
    assert annotations["prompt_preset"].tolist() == [
        "arena-hard",
        "arena-hard",
        "arena-hard-creative",
    ] * (2 if swap_mode == "both" else 1)
    completions = outputs.set_index(["instruction_index", "model"])["completion"]
    for slot in ("a", "b"):
        assert annotations[f"completion_{slot}"].tolist() == [
            completions.loc[index, model]
            for index, model in zip(ids, annotations[f"model_{slot}"], strict=True)
        ]
    assert annotations["judge_completion"].tolist() == (
        ["[[A>B]]", "no verdict", "[[A>B]]", "[[B>A]]", "[[A>B]]", "[[B>A]]"]
        if swap_mode == "both"
        else raw
    )
    with JudgementCache(next((tmp_path / "cache").rglob(JUDGEMENT_DB_NAME))) as cache:
        cached = cache.query()
    assert set(
        cached[["instruction_id", "model_a", "model_b", "judge_completion"]].itertuples(
            index=False, name=None
        )
    ) == set(
        annotations.assign(
            judge_a=annotations.model_a.where(
                ~annotations.swapped, annotations.model_b
            ),
            judge_b=annotations.model_b.where(
                ~annotations.swapped, annotations.model_a
            ),
        )[["instruction_index", "judge_a", "judge_b", "judge_completion"]].itertuples(
            index=False, name=None
        )
    )
    assert cached["orientation"].tolist() == [
        "direct" if model == "candidate" else "reversed" for model in cached["model_a"]
    ]
    metadata = json.loads(next(tmp_path.glob("*/run-metadata.v1.json")).read_text())
    assert {item["judge_prompt_preset"] for item in metadata["judge_prompts"]} == {
        "arena-hard",
        "arena-hard-creative",
    }


def test_run_evaluation_weights_and_preserves_incomplete_alpaca_annotations(
    monkeypatch, tmp_path, caplog
):
    valid = InferenceResult(
        text="M", first_token_top_logprobs={"m": np.log(0.25), "M": np.log(0.75)}
    )
    instructions = pd.DataFrame(
        {"instruction": [f"Synthetic instruction {i}" for i in range(4)]},
        index=pd.Index(range(4), name="instruction_index"),
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_task_data",
        lambda *_args, **_kwargs: CompletionData(instructions),
    )
    missing = InferenceResult(text="M")
    judge_kwargs = _mock_alpaca_judge(monkeypatch, [valid, missing, valid, missing])

    prefs = run_evaluation(
        RunConfig(
            task="alpaca-eval",
            model={"name": "Dummy/a", "baseline": "Dummy/b"},
            judge={"model": "OpenRouter/fake-judge", "swap_mode": "random"},
            generation={"n_instructions": 4},
            run={"result_folder": str(tmp_path)},
        )
    )

    assert prefs.tolist() == pytest.approx([0.75, np.nan, 0.25, np.nan], nan_ok=True)
    annotations = pd.read_csv(next(tmp_path.glob("*/*-annotations.csv")))
    assert annotations["judge_completion"].tolist() == ["M"] * 4
    assert annotations["parsed"].isna().tolist() == [False, True, False, True]
    results = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    metric = results["metrics"]["alpaca_eval_length_controlled"]
    assert judge_kwargs["top_logprobs"] == 5
    assert metric["num_missing"] == 2
    assert metric["num_wins"] == metric["num_losses"] == 1
    assert "2/4 judge outputs could not be parsed" in caplog.text


def test_all_missing_alpaca_judgments_save_empty_results(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        benchmark_execution,
        "prepare_model",
        lambda **_kwargs: FakeListLLM(responses=["M"]),
    )

    prefs = run_evaluation(
        RunConfig(
            task="alpaca-eval",
            model={"name": "Dummy/a", "baseline": "Dummy/b"},
            judge={"model": "OpenRouter/fake-judge", "swap_mode": "random"},
            generation={"n_instructions": 2},
            run={"result_folder": str(tmp_path)},
        )
    )

    assert prefs.isna().all()
    annotations = pd.read_csv(next(tmp_path.glob("*/*-annotations.csv")))
    assert annotations["judge_completion"].tolist() == ["M", "M"]
    assert annotations["parsed"].isna().all()
    results = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    assert results["metrics"]["alpaca_eval_length_controlled"] == {}
    assert next(tmp_path.glob("*/run-metadata.v1.json")).is_file()
    assert "alpaca_eval_length_controlled: unavailable" in capsys.readouterr().out


def test_judge_comparisons_preserves_prepared_order_and_cache_identity(tmp_path):
    from judgearena.benchmarks.pairwise.judging import judge_comparisons
    from judgearena.prompts.registry import resolve_judge_prompt

    cfg = RunConfig(
        task="arena-hard-v2.0",
        model={"name": "candidate"},
        judge={"model": "Dummy/judge"},
        run={"store_root": str(tmp_path)},
    )
    model = prepare_model(
        "Dummy/judge", cache=benchmark_execution.build_judgement_cache(cfg)
    )
    model.factory = lambda: FakeListLLM(
        responses=["[[A>B]]", "[[B>A]]", "no verdict", "[[A>B]]"]
    )
    comparisons = pd.DataFrame(
        {
            "instruction": ["first", "second"],
            "completion_a": ["opponent first", "candidate second"],
            "completion_b": ["candidate first", "opponent second"],
            "model_a": ["opponent", "candidate"],
            "model_b": ["candidate", "opponent"],
            "instruction_id": ["arena:shared", "arena:shared"],
            "cache_orientation": ["reversed", "direct"],
        },
        index=[8, 3],
    )
    result = judge_comparisons(
        model,
        comparisons,
        [
            (resolve_judge_prompt(preset="arena-hard"), pd.Index([3])),
            (resolve_judge_prompt(preset="arena-hard-creative"), pd.Index([8])),
        ],
        swap_mode="both",
        strip_thinking_before_judging=False,
        truncate_input_chars=None,
        use_tqdm=False,
    )
    assert result["instruction_index"].tolist() == [3, 8, 3, 8]
    assert result["completion_a"].tolist() == ["candidate second", "opponent first"] * 2
    assert result["pref"].tolist() == pytest.approx(
        [0.25, np.nan, 0.25, 0.75], nan_ok=True
    )
    with JudgementCache(next(tmp_path.rglob(JUDGEMENT_DB_NAME))) as cache:
        cached = cache.query()
    assert cached["instruction_id"].tolist() == ["arena:shared"] * 4
    assert cached["orientation"].tolist() == [
        "direct" if name == "candidate" else "reversed" for name in cached["model_a"]
    ]


def test_complete_missing_preserves_repeated_cache_ids():
    rows = pd.Series(["q1", "q2", "q3"], index=["shared"] * 3)
    existing = pd.DataFrame(
        {"completion": ["", None, None], "checkpoint": ["saved", "stale", "stale"]},
        index=rows.index,
    )
    result = benchmark_execution.complete_missing(
        rows,
        existing,
        lambda pending: (
            pending.str.upper().to_frame("completion").assign(tokens=[2, 3])
        ),
    )
    assert result["completion"].tolist() == ["", "Q2", "Q3"]
    assert result.index.tolist() == ["shared"] * 3
    assert result.iloc[0]["checkpoint"] == "saved"
    assert result.iloc[1:]["checkpoint"].isna().all()
    assert result.iloc[1:]["tokens"].tolist() == [2, 3]
    assert existing["checkpoint"].tolist() == ["saved", "stale", "stale"]


def test_pairwise_preserves_selected_output_metadata_and_histories(
    tmp_path, monkeypatch
):
    from judgearena.datasets.data import PreferenceData

    data = PreferenceData.from_battles(
        pd.DataFrame(
            [
                dict(
                    instruction="question",
                    model_a="source-a",
                    model_b="source-b",
                    completion_a="old answer",
                    completion_b="baseline answer",
                    messages_a=[
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "stale source history"},
                    ],
                    messages_b=[
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "baseline answer"},
                    ],
                    checkpoint_a="source checkpoint",
                    checkpoint_b="baseline checkpoint",
                    reference_pref=0.5,
                    category="coding",
                )
            ]
        )
    )
    monkeypatch.setattr(generate_and_evaluate, "load_task_data", lambda _: data)
    run_evaluation(
        _cfg(
            task="arena-hard-v0.1",
            model_A="Dummy/current answer",
            model_B="source-b",
            judge_model="Dummy/[[A>B]]",
            result_folder=str(tmp_path),
        )
    )
    annotations = pd.read_csv(next(tmp_path.rglob("*-annotations.csv")))
    prompt = annotations.iloc[0].judge_input
    assert "reference_pref" not in annotations
    assert "current answer" in prompt and "baseline answer" in prompt
    assert "stale" not in prompt
    assert "coding" not in prompt
    assert annotations.iloc[0].category == "coding"
    assert annotations.iloc[0].checkpoint_b == "baseline checkpoint"
    assert "source checkpoint" not in annotations.to_string()
    assert "assistant: baseline answer" in prompt


@pytest.mark.parametrize("system_prompt", [None, "Use concise answers."])
def test_multiturn_generation_batches_active_turns_and_reuses_history_cache(
    tmp_path, monkeypatch, system_prompt
):
    import judgearena.generate as generation
    from judgearena.cache.inference import CompletionInferenceCache

    calls, temperatures = [], []

    def make_model(*args, **kwargs):
        model = prepare_model(*args, **kwargs)

        def batch(inputs, **_):
            calls.append([value.to_messages() for value in inputs])
            temperatures.append(kwargs["temperature"])
            return ["<think>hidden</think>answer"] * len(inputs)

        model.factory = lambda: SimpleNamespace(batch=batch)
        return model

    monkeypatch.setattr(generation, "prepare_model", make_model)
    questions = pd.DataFrame(
        {
            "user_turns": [["q1", "q2", "q3"], ["short"]],
            "category": ["coding", "other"],
        },
        index=[7, 9],
    )
    kwargs = dict(
        questions=questions,
        model="Dummy/answer",
        use_tqdm=False,
        truncate_input_chars=None,
        strip_thinking_before_history=True,
        inference_cache=CompletionInferenceCache(tmp_path, "conversations"),
        system_prompt=system_prompt,
        temperature_config={"coding": 0.0},
        fallback_temperature=0.4,
    )
    answers = generation.generate_multiturn(**kwargs)
    assert [len(batch) for batch in calls] == [1, 1, 1, 1]
    assert temperatures == [0.0, 0.4, 0.0, 0.0]
    assert [message.content for message in calls[3][0]] == (
        [system_prompt] if system_prompt is not None else []
    ) + [
        "q1",
        "answer",
        "q2",
        "answer",
        "q3",
    ]
    assert answers.instruction_index.tolist() == [7, 9]
    assert answers.messages.map(len).tolist() == [6, 2]
    assert answers.iloc[0].messages[1]["content"] == "<think>hidden</think>answer"
    pd.testing.assert_frame_equal(generation.generate_multiturn(**kwargs), answers)
    assert len(calls) == 4


def test_independent_score_presets_compare_expanded_candidates_in_order():
    from judgearena.prompts.criteria import resolve_criteria
    from judgearena.prompts.registry import resolve_run_judge_prompt

    task = get_packaged_task("arena-hard-v2.0")
    judge_spec = task.spec.protocol.judge.model_copy(
        update={
            "default_prompt_preset": "single-score",
            "category_prompts": {"creative": "criteria-score"},
        }
    )
    task = replace(
        task,
        spec=task.spec.model_copy(
            update={
                "protocol": task.spec.protocol.model_copy(update={"judge": judge_spec}),
            }
        ),
    )
    cfg = RunConfig(
        task=task.task,
        resolved_task=task,
        model={"name": "a"},
        judge={"model": "Dummy/j"},
    )
    instructions = pd.DataFrame(
        {
            "instruction": ["Q1", "Q2"],
            "category": ["creative", "other"],
        },
        index=["q1", "q2"],
    )
    comparisons = (
        instructions.loc[["q2", "q1", "q2"]]
        .assign(
            source_instruction_index=["q1", "q2", "q1"],
            model_a="a",
            model_b=["b", "b", "c"],
            completion_a=["A2", "A1", "A2"],
            completion_b=["B2", "B1", "C2"],
        )
        .set_axis(["pair2", "pair1", "pair3"])
    )
    criteria_response = json.dumps(
        {"criteria": {criterion.name: 6 for criterion in resolve_criteria()}}
    )
    model = FakeListLLM(
        responses=[
            criteria_response,
            '{"score": 8}',
            '{"score": 8}',
            criteria_response,
            '{"score": 4}',
            "no score",
        ]
    )

    judge, _ = prepare_pairwise_judging(
        cfg,
        resolve_run_judge_prompt(task, cfg.judge),
        instructions,
    )
    annotations, outcomes, metadata = judge(
        model,
        comparisons,
        source_ids=pd.Series({"pair1": "q1", "pair3": "q2", "pair2": "q2"}),
    )

    assert all(
        f"Answer: {row.completion}" in row.judge_input
        and f"Question: Q{row.completion[-1]}" in row.judge_input
        for row in annotations.itertuples()
    )
    assert annotations.instruction_index.tolist() == ["pair2", "pair1", "pair3"] * 2
    assert annotations.completion.tolist() == ["A2", "A1", "A2", "B2", "B1", "C2"]
    assert outcomes.pref.tolist() == pytest.approx(
        [0.2314752165, 0.5, np.nan], nan_ok=True
    )
    expected_prompts = ["single-score", "criteria-score", "single-score"]
    assert outcomes.prompt_preset.tolist() == expected_prompts
    assert {p["judge_prompt_preset"] for p in metadata["judge_prompts"]} == set(
        expected_prompts
    )
    assert annotations.iloc[1]["parsed"]["details"]["criteria"]["clarity"] == 6


def test_assemble_comparisons_aligns_outputs_without_inheriting_old_labels():
    from judgearena.benchmarks.pairwise.comparisons import assemble_comparisons

    instructions = pd.DataFrame(
        {"instruction": ["Q2", "Q1"], "category": "coding", "reference_pref": 1.0},
        index=[2, 1],
    )
    a = pd.DataFrame(
        {"model": "a", "completion": ["A1", "A2"], "checkpoint": ["v1", "v2"]},
        index=[1, 2],
    )
    b = pd.DataFrame(
        {
            "model": "b",
            "completion": ["B2", "B1"],
            "messages": [
                [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "B2"},
                ],
                [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "B1"},
                ],
            ],
        },
        index=[2, 1],
    )
    pairs = assemble_comparisons(instructions, a, b)
    assert pairs["completion_a"].tolist() == ["A2", "A1"]
    assert pairs["checkpoint_a"].tolist() == ["v2", "v1"]
    assert pairs["messages_b"].tolist() == b["messages"].tolist()
    assert pairs["category"].tolist() == ["coding", "coding"]
    assert "reference_pref" not in pairs


def test_minimum_score_grouping_requires_scoring_prompt():
    from judgearena.prompts.registry import resolve_judge_prompt

    cfg = _cfg(
        task="mt-bench-101",
        model_A="Dummy/a",
        model_B="Dummy/b",
        judge_model="Dummy/judge",
    )
    with pytest.raises(ValueError, match="minimum_score_by requires a single-answer"):
        prepare_pairwise_judging(
            cfg, resolve_judge_prompt(preset="default"), pd.DataFrame()
        )


def test_category_baseline_uses_configured_field():
    task = get_packaged_task("arena-hard-v2.0")
    spec = task.spec.model_dump()
    spec["protocol"]["baseline"]["category_field"] = "domain"
    task = replace(task, spec=type(task.spec).model_validate(spec))
    instructions = pd.DataFrame({"instruction": ["Q"], "domain": ["creative_writing"]})
    plan = resolve_baseline_plan(
        task_id=task.task, task=task, runtime_baseline=None, instructions=instructions
    )
    assert plan.baseline_by_index.tolist() == ["gemini-2.0-flash-001"]
