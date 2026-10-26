import json
from dataclasses import replace

import pandas as pd
import pytest

import judgearena.benchmarks.mt_bench.runner as mt_bench_runner
import judgearena.datasets.mt_bench as mt_bench
import judgearena.utils as judgearena_utils
from judgearena.config import RunConfig
from judgearena.datasets.data import CompletionData
from judgearena.datasets.registry import DatasetAdapter
from judgearena.prompts.registry import FASTCHAT_PAIRWISE_PROMPT_PRESET
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import TaskSelection


def test_mt_bench_answers_use_resolved_task_source_and_cache(monkeypatch, tmp_path):
    original = get_packaged_task("mt-bench")
    spec = original.spec.model_dump()
    spec["dataset"]["sources"]["benchmark"].update(
        repo_id="custom/mt-bench", revision="1234567890123456789012345678901234567890"
    )
    custom = replace(original, spec=type(original.spec).model_validate(spec))
    monkeypatch.setattr(judgearena_utils, "data_root", tmp_path)
    monkeypatch.setattr(mt_bench, "data_root", tmp_path)
    reference = tmp_path / "reference.jsonl"
    reference.write_text('{"question_id": 7, "choices": [{"turns": ["R1", "R2"]}]}\n')
    monkeypatch.setattr(
        mt_bench.MTBenchAdapter, "_download_references", lambda self: reference
    )
    downloads = []

    def snapshot(**kwargs):
        downloads.append(kwargs)
        path = kwargs["local_dir"] / kwargs["allow_patterns"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = f"{kwargs['repo_id']}@{kwargs['revision']}"
        if path.name == "question.jsonl":
            record = {"question_id": 7, "category": "math", "turns": [stamp, "Q2"]}
        else:
            record = {"question_id": 7, "choices": [{"turns": [stamp]}]}
        path.write_text(json.dumps(record) + "\n")

    monkeypatch.setattr(mt_bench, "snapshot_download", snapshot)
    for task in (original, custom, original):
        data = mt_bench_runner.load_task_data(task)
        source = task.spec.dataset.sources["benchmark"]
        expected = f"{source.repo_id}@{source.revision}"
        assert data.instructions.loc[
            7, ["category", "turn_1", "turn_2", "reference_turn_1", "reference_turn_2"]
        ].tolist() == ["math", expected, "Q2", "R1", "R2"]
        assert data.model_output_loader("candidate")["completion_turn_2"].tolist() == [
            ""
        ]
        assert data.model_completion("candidate").loc[7] == expected
    assert len(downloads) == 4  # One question and answer file per source.


def test_generate_mt_bench_completions_uses_pregenerated_baseline(tmp_path):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1", "Q2"], "turn_2": ["Q1b", "Q2b"]},
        index=pd.Index([1, 2], name="instruction_index"),
    )
    data = CompletionData(
        questions_df.assign(instruction=questions_df["turn_1"]),
        pd.DataFrame(
            {
                "instruction_index": [2, 1],
                "model": ["gpt-4", "gpt-4"],
                "output": ["Base A2", "Base A1"],
                "completion_turn_1": ["Base A2", "Base A1"],
                "completion_turn_2": ["Base B2", "Base B1"],
            }
        ),
    )

    cfg = RunConfig(
        task="mt-bench",
        model={
            "name": "Dummy/generated",
            "baseline": "gpt-4",
        },
        judge={"model": "Dummy/J"},
        generation={"n_instructions": 2},
        run={"store_root": str(tmp_path / "cache")},
    )

    completions_a, completions_b = mt_bench_runner._generate_mt_bench_completions(
        cfg=cfg,
        task=get_packaged_task("mt-bench"),
        task_data=data,
        questions_df=questions_df,
    )

    assert completions_a["completion_turn_1"].tolist() == ["generated", "generated"]
    assert completions_b.loc[1, "completion_turn_1"] == "Base A1"
    assert completions_b.loc[2, "completion_turn_2"] == "Base B2"
    data = replace(data, model_outputs=data.model_outputs.iloc[:1])
    with pytest.raises(ValueError, match="missing 1 question"):
        mt_bench_runner._generate_mt_bench_completions(
            cfg=cfg,
            task=get_packaged_task("mt-bench"),
            task_data=data,
            questions_df=questions_df,
        )


@pytest.mark.parametrize("prompt_preset", [FASTCHAT_PAIRWISE_PROMPT_PRESET, "default"])
def test_run_mt_bench_rejects_random_before_preparation(monkeypatch, prompt_preset):
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "Dummy/model"},
        judge={
            "model": "Dummy/judge",
            "prompt_preset": prompt_preset,
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


def test_multiturn_runner_accepts_registered_dataset_outputs(monkeypatch, tmp_path):
    import judgearena.datasets.registry as registry

    questions = pd.DataFrame(
        {
            "instruction_index": [9, 3],
            "turn_1": ["Keep first", "Skip first"],
            "turn_2": ["Keep second", "Skip second"],
            "domain": ["keep", "skip"],
        }
    )
    outputs = pd.DataFrame(
        {
            "instruction_index": [3, 9, 9, 3],
            "model": ["a", "b", "a", "b"],
            "output": ["skip a", "first b", "first a", "skip b"],
            "completion_turn_1": ["skip a", "first b", "first a", "skip b"],
            "completion_turn_2": ["skip a2", "second b", "second a", "skip b2"],
        }
    )
    monkeypatch.setattr(registry, "_DATASETS", registry._DATASETS.copy())

    @registry.register_dataset("custom_conversations")
    class CustomConversationsAdapter(DatasetAdapter):
        def load_data(self):
            return CompletionData.from_frames(
                questions.assign(instruction=questions["turn_1"]),
                model_output_loader=lambda model: outputs,
            )

    original = get_packaged_task("mt-bench")
    spec = original.spec.model_dump()
    spec["dataset"] = {"adapter": "custom_conversations"}
    task = replace(
        original,
        spec=type(original.spec).model_validate(spec),
        selection=TaskSelection(selector="domain", name="selected", values=("keep",)),
    )
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "a", "baseline": "b"},
        judge={
            "model": "Dummy/score_A: 10\nscore_B: 0",
            "prompt_preset": "default_with_explanation",
            "swap_mode": "fixed",
        },
        run={"result_folder": str(tmp_path), "store_root": str(tmp_path / "cache")},
    )
    prefs = mt_bench_runner.run_mt_bench_benchmark(cfg, task)
    assert len(prefs) == 2 and prefs.lt(0.5).all()
    annotations = pd.read_csv(next(tmp_path.glob("**/*-annotations.csv")))
    assert set(annotations["question_id"]) == {9}
    assert set(annotations["turn"]) == {1, 2}
    assert "second a" in annotations.iloc[1]["user_prompt"]
    assert "second b" in annotations.iloc[1]["user_prompt"]
    metadata = json.loads(next(tmp_path.glob("**/run-metadata.v1.json")).read_text())
    assert metadata["dataset_statistics"]["instruction_index_count"] == 1
    assert metadata["results"]["preferences_count"] == 2
    saved = json.loads(next(tmp_path.glob("**/results-*.json")).read_text())
    metric = saved["metrics"]["pairwise_win_rate"]
    assert metric["winrate"] == 1.0
    assert [(g["group"], g["values"]["winrate"]) for g in metric["groups"]["turn"]] == [
        (1, 1.0),
        (2, 1.0),
    ]
