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


def test_normalize_wildbench_preserves_multiturn_checklists_and_tags():
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

    normalized = normalize_wildbench(raw)

    row = normalized.iloc[0]
    assert row["instruction_index"] == "s1"
    assert row["instruction"] == "Follow-up"
    assert row["conversation_input"][-1] == {
        "role": "user",
        "content": "Follow-up",
    }
    assert row["history"] == ("USER: First question\n\nASSISTANT: First answer\n\n")
    assert row["checklist"] == ["Correct?", "Clear?"]
    assert row["task_categories"] == [
        "Math & Data Analysis",
        "Planning & Reasoning",
    ]


def test_normalize_wildbench_rejects_conversation_not_ending_in_user():
    raw = pd.DataFrame(
        {
            "session_id": ["s1"],
            "conversation_input": [[{"role": "assistant", "content": "No user query"}]],
            "checklist": [[]],
            "primary_tag": ["Others"],
            "secondary_tags": [[]],
        }
    )
    with pytest.raises(ValueError, match="must end with a user query"):
        normalize_wildbench(raw)


@pytest.mark.parametrize(
    "completion, expected",
    [
        ('{"strengths": "good", "weaknesses": "none", "score": "8"}', 8.0),
        ('analysis before JSON\n{"score": 6}', 6.0),
        ("broken output: score: 4.5", 4.5),
        ('{"score": 11}', None),
        ("no score", None),
    ],
)
def test_parse_wildbench_score(completion, expected):
    assert parse_wildbench_score(completion) == expected


@pytest.mark.parametrize(
    "completion, expected",
    [
        ('{"choice": "A++"}', "A++"),
        ('analysis\n{"choice": " b+ "}', "B+"),
        ("I choose A=B", "A=B"),
        ("undecided", None),
    ],
)
def test_parse_wildbench_choice(completion, expected):
    assert parse_wildbench_choice(completion) == expected


@pytest.mark.parametrize(
    "choice, candidate_is_a, expected",
    [
        ("A++", True, 1.0),
        ("A+", False, -0.5),
        ("A=B", False, 0.0),
        ("B+", False, 0.5),
        ("B++", True, -1.0),
    ],
)
def test_choice_to_candidate_reward(choice, candidate_is_a, expected):
    assert choice_to_candidate_reward(choice, candidate_is_a=candidate_is_a) == expected


def test_length_penalty_only_ties_slight_winner_with_length_advantage():
    assert apply_wildbench_length_penalty(0.5, "x" * 11, "", 10) == 0.0
    assert apply_wildbench_length_penalty(-0.5, "", "x" * 11, 10) == 0.0
    assert apply_wildbench_length_penalty(0.5, "x" * 10, "", 10) == 0.5
    assert apply_wildbench_length_penalty(1.0, "x" * 100, "", 10) == 1.0
    assert apply_wildbench_length_penalty(0.5, "x" * 100, "", None) == 0.5


def test_official_prompts_replace_fields_and_keep_required_output_contract():
    example = _examples().loc["s2"]
    score_prompt = render_wildbench_score_prompt(
        example, "Model answer", max_words=1000, max_chars=None
    )
    pairwise_prompt = render_wildbench_pairwise_prompt(
        example, "Answer A", "Answer B", max_words=1000, max_chars=None
    )

    assert "USER: Earlier" in score_prompt
    assert "Solve this" in score_prompt
    assert "Model answer" in score_prompt
    assert "Is it correct?" in score_prompt
    assert '"score": "[1~10]"' in score_prompt
    assert "Answer A" in pairwise_prompt
    assert "Answer B" in pairwise_prompt
    assert '"choice": "[A++ or A+ or A=B or B+ or B++]"' in pairwise_prompt
    assert "$" not in score_prompt
    assert "$" not in pairwise_prompt


def test_score_metrics_use_public_leaderboard_scale_and_secondary_categories():
    examples = _examples()
    examples.at["s1", "task_categories"] = [
        "Creative Tasks",
        "Planning & Reasoning",
    ]
    annotations = pd.DataFrame({"session_id": ["s1", "s2"], "score": [8.0, 6.0]})

    raw_mean, wb_score, per_category = _score_metrics(examples, annotations)

    assert raw_mean == 7.0
    assert wb_score == 40.0
    assert per_category == {
        "Creative Tasks": 60.0,
        "Planning & Reasoning": 60.0,
        "Math & Data Analysis": 20.0,
    }


def test_reward_metrics_average_each_reference_equally():
    examples = _examples()
    canonical = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s1", "s2"],
            "baseline_model": ["b1", "b1", "b2", "b2"],
            "reward": [1.0, 0.0, 0.5, -0.5],
        }
    )

    reward, per_baseline, per_category = _reward_metrics(
        examples, canonical, ["b1", "b2"]
    )

    assert reward == 25.0
    assert per_baseline == {"b1": 50.0, "b2": 0.0}
    assert per_category == {
        "Creative Tasks": 75.0,
        "Math & Data Analysis": -25.0,
    }


@pytest.mark.parametrize(
    "task, judge_completion, metric, expected",
    [
        ("wildbench-score", '{"score": "8"}', "wb_score", 60.0),
        ("wildbench-reward", '{"choice": "A=B"}', "wb_reward", 0.0),
    ],
)
def test_wildbench_main_writes_testable_artifacts(
    tmp_path, monkeypatch, task, judge_completion, metric, expected
):
    examples = _examples()
    examples["conversation_input"] = [
        [{"role": "user", "content": instruction}]
        for instruction in examples["instruction"]
    ]
    monkeypatch.setattr(
        wildbench_module,
        "load_instructions",
        lambda *args, **kwargs: examples,
    )

    def fake_outputs(cfg, frame, model_name, *, role):
        prefix = "candidate" if role == "A" else f"reference-{model_name}"
        return pd.Series([f"{prefix}-1", f"{prefix}-2"], index=frame.index, dtype=str)

    monkeypatch.setattr(wildbench_module, "_load_or_generate_outputs", fake_outputs)
    monkeypatch.setattr(wildbench_module, "_make_judge", lambda cfg: object())
    monkeypatch.setattr(
        wildbench_module,
        "_run_judge_prompts",
        lambda judge, prompts, **kwargs: [judge_completion] * len(prompts),
    )
    monkeypatch.setattr(wildbench_module, "write_run_metadata", lambda **kwargs: None)

    cfg = RunConfig(
        task=task,
        model={"name": "Dummy/candidate"},
        judge={"model": "Dummy/judge"},
        run={"result_folder": str(tmp_path), "no_log_file": True},
    )
    result = wildbench_module.main(cfg)

    assert result[metric] == expected
    if task == "wildbench-reward":
        assert result["baseline_models"] == list(OFFICIAL_WILDBENCH_BASELINES)
        assert result["num_judgments"] == len(examples) * 3
    result_path = Path(result["result_path"])
    run_dir = result_path.parent
    assert json.loads(result_path.read_text())[metric] == expected
    assert (run_dir / "config.yaml").exists()
    assert len(pd.read_parquet(run_dir / "annotations.parquet")) >= len(examples)
    assert len(pd.read_parquet(run_dir / "model_outputs.parquet")) >= len(examples)
