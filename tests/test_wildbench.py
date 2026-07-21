"""Behavioral tests for the packaged WildBench V2 implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from judgearena.benchmarks.wildbench import runner as wildbench_runner
from judgearena.benchmarks.wildbench.prompting import (
    render_wildbench_pairwise_prompt,
    render_wildbench_score_prompt,
    resolve_wildbench_prompt,
)
from judgearena.benchmarks.wildbench.scoring import (
    WildBenchRewardV2,
    WildBenchScoreV2,
    apply_wildbench_length_penalty,
    choice_to_candidate_reward,
    parse_wildbench_choice,
    parse_wildbench_score,
)
from judgearena.config import RunConfig
from judgearena.datasets.wildbench import normalize_wildbench


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction": ["Write a poem", "Solve this"],
            "history": ["", "USER: Earlier\n\nASSISTANT: Yes\n\n"],
            "checklist": [["Is it creative?"], ["Is it correct?"]],
            "task_categories": [["Creative Tasks"], ["Math & Data Analysis"]],
            "conversation_input": [
                [{"role": "user", "content": "Write a poem"}],
                [{"role": "user", "content": "Solve this"}],
            ],
        },
        index=pd.Index(["s1", "s2"], name="instruction_index"),
    )


def test_normalize_wildbench_multiturn_example():
    raw = pd.DataFrame(
        {
            "session_id": ["s1"],
            "conversation_input": [
                np.array(
                    [
                        {"role": "user", "content": "First question"},
                        {"role": "assistant", "content": "First answer"},
                        {"role": "user", "content": "Follow-up"},
                    ],
                    dtype=object,
                )
            ],
            "checklist": [np.array(["Correct?", "Clear?"], dtype=object)],
            "primary_tag": ["Math"],
            "secondary_tags": [np.array(["Reasoning"], dtype=object)],
        }
    )

    row = normalize_wildbench(raw).iloc[0]

    assert row["instruction_index"] == "s1"
    assert row["instruction"] == "Follow-up"
    assert row["history"] == "USER: First question\n\nASSISTANT: First answer\n\n"
    assert row["checklist"] == ["Correct?", "Clear?"]
    assert row["task_categories"] == [
        "Math & Data Analysis",
        "Planning & Reasoning",
    ]


@pytest.mark.parametrize(
    ("parser", "completion", "expected"),
    [
        (parse_wildbench_score, '{"score": "8"}', 8.0),
        (parse_wildbench_score, 'analysis\n{"score": 6}', 6.0),
        (parse_wildbench_score, "broken output: score: 4.5", 4.5),
        (parse_wildbench_score, '{"score": 11}', None),
        (parse_wildbench_choice, '{"choice": "A++"}', "A++"),
        (parse_wildbench_choice, "I choose A=B", "A=B"),
        (parse_wildbench_choice, "undecided", None),
    ],
)
def test_parse_judge_output(parser, completion, expected):
    assert parser(completion) == expected


def test_reward_helpers():
    assert choice_to_candidate_reward("A+", candidate_is_a=True) == 0.5
    assert choice_to_candidate_reward("A+", candidate_is_a=False) == -0.5
    assert apply_wildbench_length_penalty(0.5, "x" * 11, "", 10) == 0.0
    assert apply_wildbench_length_penalty(1.0, "x" * 100, "", 10) == 1.0


def test_official_prompt_resources_match_original_integration():
    score = resolve_wildbench_prompt("wildbench-score-v2", mode="score")
    reward = resolve_wildbench_prompt("wildbench-pairwise-v2", mode="reward")

    assert hashlib.sha256(score.template.encode()).hexdigest() == (
        "c365fc8c0052049ffd85c7a38f37ccffd99358cc5e7c3e681a385d259ffaa87b"
    )
    assert hashlib.sha256(reward.template.encode()).hexdigest() == (
        "aac23203487964a4b337ccfc4611b4bd7443147d22b31b31eb01f0a52b6408e4"
    )


def test_official_prompts_include_inputs_and_output_contracts():
    example = _examples().loc["s2"]
    score_prompt = render_wildbench_score_prompt(
        resolve_wildbench_prompt("wildbench-score-v2", mode="score"),
        example,
        "Model answer",
        max_words=1000,
        max_chars=None,
    )
    pairwise_prompt = render_wildbench_pairwise_prompt(
        resolve_wildbench_prompt("wildbench-pairwise-v2", mode="reward"),
        example,
        "Answer A",
        "Answer B",
        max_words=1000,
        max_chars=None,
    )

    assert all(
        value in score_prompt
        for value in ("USER: Earlier", "Solve this", "Model answer", "Is it correct?")
    )
    assert '"score": "[1~10]"' in score_prompt
    assert "Answer A" in pairwise_prompt and "Answer B" in pairwise_prompt
    assert '"choice": "[A++ or A+ or A=B or B+ or B++]"' in pairwise_prompt


def test_wildbench_metrics_use_published_scales():
    examples = _examples()
    score_annotations = pd.DataFrame({"session_id": ["s1", "s2"], "score": [8.0, 6.0]})
    score = WildBenchScoreV2().aggregate(
        examples, score_annotations, baseline_models=[]
    )
    assert (score.raw_mean, score.value) == (7.0, 40.0)
    assert score.per_category == {
        "Creative Tasks": 60.0,
        "Math & Data Analysis": 20.0,
    }

    reward_annotations = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s1", "s2"],
            "baseline_model": ["b1", "b1", "b2", "b2"],
            "reward": [1.0, 0.0, 0.5, -0.5],
        }
    )
    reward = WildBenchRewardV2().aggregate(
        examples, reward_annotations, baseline_models=["b1", "b2"]
    )
    assert reward.value == 25.0
    assert reward.per_baseline == {"b1": 50.0, "b2": 0.0}
    assert reward.per_category == {
        "Creative Tasks": 75.0,
        "Math & Data Analysis": -25.0,
    }


@pytest.mark.parametrize(
    ("task", "judge_completion", "metric", "expected"),
    [
        ("wildbench-score", '{"score": "8"}', "wb_score", 60.0),
        ("wildbench-reward", '{"choice": "A=B"}', "wb_reward", 0.0),
    ],
)
def test_wildbench_run_writes_results(
    tmp_path, monkeypatch, task, judge_completion, metric, expected
):
    examples = _examples().reset_index()
    adapter = SimpleNamespace(
        load_instructions=lambda *_args: examples,
        load_model_outputs=lambda *_args: pd.DataFrame(
            columns=["instruction_index", "model", "output"]
        ),
    )
    monkeypatch.setattr(
        wildbench_runner, "resolve_dataset_adapter", lambda _name: adapter
    )
    monkeypatch.setattr(
        wildbench_runner,
        "_load_or_generate_outputs",
        lambda cfg, frame, model_name, *, role, **kwargs: pd.Series(
            [f"{role}-{index}" for index in frame.index],
            index=frame.index,
            dtype=str,
        ),
    )
    monkeypatch.setattr(wildbench_runner, "build_judge", lambda cfg: object())
    monkeypatch.setattr(
        wildbench_runner,
        "_run_judge_prompts",
        lambda judge, prompts, **kwargs: [judge_completion] * len(prompts),
    )
    monkeypatch.setattr(
        wildbench_runner, "write_run_metadata_safely", lambda **kwargs: None
    )

    result = wildbench_runner.run_wildbench(
        RunConfig(
            task=task,
            model={"name": "Dummy/candidate"},
            judge={"model": "Dummy/judge"},
            run={"result_folder": str(tmp_path), "no_log_file": True},
        )
    )

    result_path = Path(result["result_path"])
    assert result[metric] == expected
    assert json.loads(result_path.read_text())[metric] == expected
    assert (result_path.parent / "config.yaml").exists()
    assert (result_path.parent / "annotations.parquet").exists()
    if task == "wildbench-reward":
        assert result["baseline_models"] == [
            "gpt-4-turbo-2024-04-09",
            "claude-3-haiku-20240307",
            "Llama-2-70b-chat-hf",
        ]
