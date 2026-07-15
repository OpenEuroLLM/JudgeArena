from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import judgearena.wildbench as wildbench_module
from judgearena.config import RunConfig
from judgearena.instruction_dataset.wildbench import (
    OFFICIAL_WILDBENCH_BASELINES,
    normalize_wildbench,
)
from judgearena.wildbench import (
    _reward_metrics,
    _score_metrics,
    apply_wildbench_length_penalty,
    choice_to_candidate_reward,
    parse_wildbench_choice,
    parse_wildbench_score,
    render_wildbench_pairwise_prompt,
    render_wildbench_score_prompt,
)


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction": ["Write a poem", "Solve this"],
            "history": ["", "USER: Earlier\n\nASSISTANT: Yes\n\n"],
            "checklist": [["Is it creative?"], ["Is it correct?"]],
            "task_categories": [["Creative Tasks"], ["Math & Data Analysis"]],
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
    assert row["task_categories"] == ["Math & Data Analysis", "Planning & Reasoning"]


@pytest.mark.parametrize(
    "parser, completion, expected",
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


def test_official_prompts_include_inputs_and_output_contracts():
    example = _examples().loc["s2"]
    score_prompt = render_wildbench_score_prompt(
        example, "Model answer", max_words=1000, max_chars=None
    )
    pairwise_prompt = render_wildbench_pairwise_prompt(
        example, "Answer A", "Answer B", max_words=1000, max_chars=None
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
    annotations = pd.DataFrame({"session_id": ["s1", "s2"], "score": [8.0, 6.0]})
    raw_mean, score, categories = _score_metrics(examples, annotations)
    assert (raw_mean, score) == (7.0, 40.0)
    assert categories == {"Creative Tasks": 60.0, "Math & Data Analysis": 20.0}

    rewards = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s1", "s2"],
            "baseline_model": ["b1", "b1", "b2", "b2"],
            "reward": [1.0, 0.0, 0.5, -0.5],
        }
    )
    reward, per_baseline, categories = _reward_metrics(examples, rewards, ["b1", "b2"])
    assert reward == 25.0
    assert per_baseline == {"b1": 50.0, "b2": 0.0}
    assert categories == {"Creative Tasks": 75.0, "Math & Data Analysis": -25.0}


@pytest.mark.parametrize(
    "task, judge_completion, metric, expected",
    [
        ("wildbench-score", '{"score": "8"}', "wb_score", 60.0),
        ("wildbench-reward", '{"choice": "A=B"}', "wb_reward", 0.0),
    ],
)
def test_wildbench_run_writes_results(
    tmp_path, monkeypatch, task, judge_completion, metric, expected
):
    examples = _examples()
    examples["conversation_input"] = [
        [{"role": "user", "content": instruction}]
        for instruction in examples["instruction"]
    ]
    monkeypatch.setattr(
        wildbench_module, "load_instructions", lambda *a, **kw: examples
    )
    monkeypatch.setattr(
        wildbench_module,
        "_load_or_generate_outputs",
        lambda cfg, frame, model_name, *, role: pd.Series(
            [f"{role}-{index}" for index in frame.index], index=frame.index, dtype=str
        ),
    )
    monkeypatch.setattr(wildbench_module, "_make_judge", lambda cfg: object())
    monkeypatch.setattr(
        wildbench_module,
        "_run_judge_prompts",
        lambda judge, prompts, **kw: [judge_completion] * len(prompts),
    )
    monkeypatch.setattr(wildbench_module, "write_run_metadata", lambda **kw: None)

    result = wildbench_module.main(
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
        assert result["baseline_models"] == list(OFFICIAL_WILDBENCH_BASELINES)
