import json
from dataclasses import replace

import pandas as pd
import pytest
from datasets import Dataset

from judgearena.benchmarks.scoring import build_metric
from judgearena.config import RunConfig
from judgearena.prompts.registry import resolve_judge_prompt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"choice": "A++", "analysis of A": "Accurate"}', 0.0),
        ('{"choice": "A+"}', 0.25),
        ('{"choice": "A=B"}', 0.5),
        ('{"choice": "B+"}', 0.75),
        ('{"choice": "B++"}', 1.0),
        ('Analysis\n```json\n{"choice": "B+",}\n```', 0.75),
        ('{"choice": "A"}', None),
        ('{"analysis": "A++"}', None),
    ],
)
def test_wildbench_choice_parser(text, expected):
    parsed = resolve_judge_prompt(preset="wildbench-reward").parser.parse_result(text)
    if expected is None:
        assert parsed is None
    else:
        assert parsed.preference == expected
        if "Accurate" in text:
            assert parsed.details["analysis of A"] == "Accurate"


def test_reward_penalty_preserves_strong_wins_and_strict_length_boundary():
    frame = pd.DataFrame(
        {
            "pref": [0, 0.5, 0.25, 0.25, 0.75, 0.75, None],
            "instruction_index": ["q", "q", "r", "s", "t", "u", "v"],
            "completion_model": ["x" * n for n in [20, 20, 15, 16, 4, 4, 2]],
            "completion_baseline": ["x" * n for n in [4, 4, 5, 5, 15, 14, 2]],
        }
    )
    metric = build_metric("wildbench_reward", {"length_penalty_threshold": 10})
    result = metric.calculate(frame)
    assert result["reward"] == pytest.approx(100 / 6)
    assert result["num_penalized"] == 2
    assert result["num_missing"] == 1
    assert result["num_judgments"] == 7
    assert frame.pref.iloc[3] == 0.25
    assert build_metric("wildbench_reward").calculate(frame)["num_penalized"] == 0
    assert metric.calculate(frame.iloc[-1:])["reward"] is None


@pytest.mark.parametrize("metric", ["wildbench_reward", "pairwise_win_rate"])
def test_wildbench_composes_saved_or_generated_answers_with_metrics(
    monkeypatch, tmp_path, metric
):
    from judgearena.benchmarks.runner import run_benchmark
    from judgearena.datasets import wildbench
    from judgearena.models import DummyModel
    from judgearena.tasks.registry import get_packaged_task
    from judgearena.tasks.schema import TaskSpec

    records = [
        {
            "session_id": "session-1",
            "id": "source-id",
            "conversation_input": [
                {"role": "user", "content": "Remember {42}"},
                {"role": "assistant", "content": "Understood"},
                {"role": "user", "content": "What number?"},
            ],
            "checklist": ["Does the answer give {42}?"],
            "primary_tag": "Reasoning",
            "secondary_tags": ["Memory"],
        }
    ]

    def load_dataset(repo, name, **kwargs):
        if repo == "allenai/WildBench":
            return Dataset.from_list(records)
        return Dataset.from_list(
            [
                {
                    "session_id": "session-1",
                    "generator": name,
                    "output": ["42"] if name == "candidate" else "43",
                },
                {
                    "session_id": "unused",
                    "generator": name,
                    "output": ["other"] if name == "candidate" else "other",
                },
            ]
        )

    monkeypatch.setattr(wildbench, "load_dataset", load_dataset)
    monkeypatch.setattr(
        wildbench,
        "get_dataset_config_names",
        lambda *a, **kw: ["candidate", "reference"],
    )
    task = get_packaged_task("wildbench")
    spec = task.model_dump()
    spec["protocol"]["scoring"]["metrics"] = [{"metric": metric}]
    task = replace(task, spec=TaskSpec.model_validate(spec))
    generate = metric == "pairwise_win_rate"
    batch = DummyModel.batch

    def check_generation_history(model, inputs, **kwargs):
        if model.name == "Dummy/42":
            assert [(m.type, m.content) for m in inputs[0].to_messages()] == [
                ("human", "Remember {42}"),
                ("ai", "Understood"),
                ("human", "What number?"),
            ]
        return batch(model, inputs, **kwargs)

    monkeypatch.setattr(DummyModel, "batch", check_generation_history)
    cfg = RunConfig(
        task="wildbench",
        resolved_task=task,
        model={
            "name": "Dummy/42" if generate else "candidate",
            "baseline": "reference",
        },
        judge={
            "model": 'Dummy/{"choice": "A+"}',
            "swap_mode": "fixed" if generate else "both",
        },
        run={
            "result_folder": str(tmp_path / "results"),
            "no_log_file": True,
            "store_root": str(tmp_path / "cache"),
            "use_tqdm": False,
        },
    )
    data = wildbench.WildBenchAdapter(cfg.resolved_task, tmp_path).load_data()
    assert (
        data.instructions.loc["session-1", "messages"]
        == records[0]["conversation_input"]
    )
    assert (
        data.instructions.loc["session-1", "history"]
        == "USER: Remember {42}\n\nASSISTANT: Understood\n\n"
    )
    assert data.model_completion("candidate").tolist() == ["42"]
    assert data.model_completion("new-model") is None
    prefs = run_benchmark(cfg)
    assert prefs.tolist() == ([0.25] if generate else [0.25, 0.75])
    annotations = pd.read_csv(next((tmp_path / "results").rglob("*-annotations.csv")))
    for prompt in annotations.judge_input:
        assert "Remember {42}" in prompt and "Understood" in prompt
        assert "- Does the answer give {42}?\n" in prompt
    assert annotations.category.tolist() == ["Reasoning"] * len(prefs)
    assert annotations.generator_b.eq("reference").all()
    if generate:
        assert json.loads(annotations.messages_a.iloc[0])[-1]["content"] == "42"
    report = json.loads(
        next((tmp_path / "results").rglob("results-*.json")).read_text()
    )
    assert (
        "<|begin_of_response_A|>\n\n42\n\n<|end_of_response_A|>"
        in annotations.judge_input.iloc[0]
    )
    assert (
        "<|begin_of_response_B|>\n\n43\n\n<|end_of_response_B|>"
        in annotations.judge_input.iloc[0]
    )
    if generate:
        result = report["metrics"]["pairwise_win_rate"]
        assert result["winrate"] == 1.0
    else:
        assert report["metrics"]["wildbench_reward"]["reward"] == 0


def test_reward_mix_weights_references_equally_and_keeps_missing_results():
    metric = build_metric("wildbench_reward_mix", {"length_penalty_threshold": 10})
    frame = pd.DataFrame(
        {
            "baseline": ["a", "a", "a", "b", "c"],
            "pref": [0.0, 0.0, None, 1.0, 0.25],
            "completion_model": ["long answer!", "x", "x", "x", "long answer!"],
            "completion_baseline": ["x", "x", "x", "x", "x"],
        }
    )
    result = metric.calculate(frame, references=["a", "b", "c"])
    assert result["reward"] == 0  # Mean of 100, -100, and penalized 0.
    assert result["references"]["a"]["num_missing"] == 1
    assert result["references"]["c"]["num_penalized"] == 1
    reversed_frame = frame.assign(
        evaluation_model="candidate",
        model_a=frame.baseline,
        model_b="candidate",
        pref=1 - frame.pref,
    )
    assert metric.calculate(reversed_frame, references=["a", "b", "c"]) == result
    result = metric.calculate(frame, references=["a", "b"])
    assert result["reward"] == 0  # Unequal valid counts must not change weights.
    frame.loc[frame.baseline == "a", "pref"] = None
    for missing in (frame, frame[frame.baseline != "a"]):
        result = metric.calculate(missing, references=["a", "b", "c"])
        assert result["reward"] is None
        assert result["references"]["a"]["reward"] is None
        assert result["references"]["b"]["reward"] == -100


@pytest.mark.parametrize("generation_mode", ["provided_messages", "multi_turn_chat"])
def test_reward_mix_generates_candidate_once_before_expanding_references(
    monkeypatch, tmp_path, generation_mode
):
    from judgearena.benchmarks.runner import run_benchmark
    from judgearena.datasets.data import CompletionData
    from judgearena.datasets.wildbench import WildBenchAdapter
    from judgearena.tasks.registry import get_packaged_task

    task = get_packaged_task("wildbench-reward-mix")
    assert task is not None, "Reward-Mix needs a task using the shared pairwise runner"
    from judgearena.tasks.schema import TaskSpec

    spec = task.model_dump()
    spec["protocol"]["generation"] = {"mode": generation_mode}
    task = replace(task, spec=TaskSpec.model_validate(spec))
    references = task.spec.protocol.baseline.references
    data = CompletionData.from_frames(
        pd.DataFrame(
            {
                "instruction_index": ["q1", "q2"],
                "instruction": ["First question", "Second question"],
                "user_turns": [["First question"], ["Second question"]],
                "source_instruction_index": ["foreign-1", "foreign-2"],
                "messages": [
                    [{"role": "user", "content": question}]
                    for question in ("First question", "Second question")
                ],
                "history": ["", ""],
                "checklist": ["- Relevant", "- Clear"],
                "category": ["Reasoning", "Coding"],
            }
        ),
        pd.DataFrame(
            [
                {
                    "instruction_index": q,
                    "model": model,
                    "completion": f"{model}: {q}",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": f"{model}: {q}"},
                    ],
                }
                for q in ("q1", "q2")
                for model in references
            ]
        ),
    )
    monkeypatch.setattr(WildBenchAdapter, "load_data", lambda self: data)
    first = None
    for run, cache in (
        ("uncached", None),
        ("live", str(tmp_path / "cache")),
        ("replay", str(tmp_path / "cache")),
    ):
        cfg = RunConfig(
            task=task.task,
            resolved_task=task,
            model={"name": "Dummy/generated"},
            judge={"model": 'Dummy/{"choice": "A+"}', "swap_mode": "both"},
            run={
                "store_root": cache,
                "result_folder": str(tmp_path / run),
                "use_tqdm": False,
                "no_log_file": True,
            },
        )
        run_benchmark(cfg)
        annotations = pd.read_csv(next((tmp_path / run).rglob("*-annotations.csv")))
        assert len(annotations) == 12
        assert annotations.instruction_index.nunique() == 6
        assert annotations.groupby("source_instruction_index").size().to_dict() == {
            "q1": 6,
            "q2": 6,
        }
        assert annotations.completion_a.eq("generated").all()
        assert set(annotations.model_b) == set(references)
        assert all(
            row.completion_b == f"{row.model_b}: {row.source_instruction_index}"
            for row in annotations.itertuples()
        )
        assert (
            annotations.loc[annotations.orientation == "direct", "pref"].eq(0.25).all()
        )
        assert (
            annotations.loc[annotations.orientation == "reversed", "pref"]
            .eq(0.75)
            .all()
        )
        metadata = json.loads(
            next((tmp_path / run).rglob("run-metadata.v1.json")).read_text()
        )
        if run == "replay":
            assert metadata["usage"]["total"]["requests"] == 0
            pd.testing.assert_frame_equal(first, annotations)
        else:
            assert metadata["usage"]["by_stage"]["generation"]["requests"] == 2
            assert metadata["usage"]["by_stage"]["judging"]["requests"] == 12
            first = annotations
        report = json.loads(next((tmp_path / run).rglob("results-*.json")).read_text())
        result = report["metrics"]["wildbench_reward_mix"]
        assert result["reward"] == 0
        assert len(result["references"]) == 3
        assert all(
            value["num_judgments"] == 4 for value in result["references"].values()
        )
        assert {group["group"] for group in result["groups"]["category"]} == {
            "Reasoning",
            "Coding",
        }


@pytest.mark.parametrize("references", [[], ["a", "a"], [""]])
def test_all_references_rejects_empty_or_duplicate_models(references):
    from judgearena.tasks.registry import get_packaged_task
    from judgearena.tasks.schema import TaskSpec

    task = get_packaged_task("wildbench-reward-mix")
    assert task is not None
    spec = task.model_dump()
    spec["protocol"]["baseline"]["references"] = references
    with pytest.raises(ValueError):
        TaskSpec.model_validate(spec)


@pytest.mark.parametrize(
    "model",
    [
        {"name": "Dummy/a", "baseline": "Dummy/b"},
        {"name": "gpt-4-turbo-2024-04-09"},
    ],
)
def test_all_references_rejects_reference_override_and_self_comparison(model):
    with pytest.raises(ValueError, match="all_references"):
        RunConfig(task="wildbench-reward-mix", model=model, judge={"model": "Dummy/A"})
