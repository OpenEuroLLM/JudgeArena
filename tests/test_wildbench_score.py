"""WildBench V2 score parser and aggregation tests."""

import math

import pandas as pd
import pytest

from judgearena.benchmarks.wildbench.parsing import (
    parse_wildbench_choice,
    parse_wildbench_score,
)
from judgearena.benchmarks.wildbench.prompting import EMPTY_RESPONSE
from judgearena.benchmarks.wildbench.scoring import (
    apply_length_penalty,
    candidate_reward,
    score_wildbench_reward_v2,
    score_wildbench_v2,
)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ('{"score": "8"}', 8.0),
        ('{"strengths": "good", "score": 6}', 6.0),
        ('analysis before {"score": "4"}', 4.0),
        ('{"score": "11"}', None),
        ('{"score": "invalid"}', None),
    ],
)
def test_parse_wildbench_score(completion, expected):
    assert parse_wildbench_score(completion) == expected


def test_wildbench_score_matches_official_transforms():
    examples = pd.DataFrame(
        {
            "instruction_index": ["s1", "s2"],
            "task_categories": [
                ["Creative Tasks"],
                ["Math & Data Analysis"],
            ],
        }
    )
    annotations = pd.DataFrame({"session_id": ["s1", "s2"], "score": [8.0, 6.0]})

    result = score_wildbench_v2(examples, annotations)

    assert result["raw_mean_score"] == 7.0
    assert result["wb_score"] == 4.0
    assert result["wb_score_leaderboard"] == 40.0
    assert result["per_category"] == {
        "Creative Tasks": 6.0,
        "Math & Data Analysis": 2.0,
    }
    assert result["task_macro_score"] == pytest.approx(5.0 / 4.75)


def test_wildbench_score_tracks_unparseable_annotations():
    examples = pd.DataFrame(
        {
            "instruction_index": ["s1"],
            "task_categories": [["Creative Tasks"]],
        }
    )
    annotations = pd.DataFrame({"session_id": ["s1"], "score": [None]})

    result = score_wildbench_v2(examples, annotations)

    assert result["num_missing"] == 1
    assert math.isnan(result["wb_score"])


def test_category_scores_exclude_empty_and_truncated_outputs():
    examples = pd.DataFrame(
        {
            "instruction_index": ["s1", "s2", "s3"],
            "task_categories": [["Creative Tasks"]] * 3,
        }
    )
    annotations = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s3"],
            "model_output": ["valid", "", "cut... (truncated)"],
            "score": [8.0, 1.0, 1.0],
        }
    )

    result = score_wildbench_v2(examples, annotations)

    assert result["raw_mean_score"] == pytest.approx(10 / 3)
    assert result["per_category"] == {"Creative Tasks": 6.0}


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ('{"choice": "A++"}', "A++"),
        ('analysis {"choice": "A=B"}', "A=B"),
        ('{"choice": "B+"}', "B+"),
        ('{"choice": "A"}', None),
    ],
)
def test_parse_wildbench_reward_choice(completion, expected):
    assert parse_wildbench_choice(completion) == expected


def test_reward_orientation_and_length_penalty():
    assert candidate_reward("A+", candidate_is_a=True) == 0.5
    assert candidate_reward("A+", candidate_is_a=False) == -0.5
    assert apply_length_penalty(0.5, "x" * 11, "", 10) == 0.0
    assert apply_length_penalty(1.0, "x" * 11, "", 10) == 1.0


def test_wildbench_reward_averages_official_reference_rewards():
    examples = pd.DataFrame(
        {
            "instruction_index": ["s1", "s2"],
            "task_categories": [
                ["Creative Tasks"],
                ["Math & Data Analysis"],
            ],
        }
    )
    annotations = pd.DataFrame(
        {
            "session_id": ["s1", "s2", "s1", "s2"],
            "baseline_model": ["b1", "b1", "b2", "b2"],
            "candidate_output": ["answer"] * 4,
            "baseline_output": ["reference"] * 4,
            "reward": [1.0, 0.0, 0.5, -0.5],
        }
    )

    result = score_wildbench_reward_v2(examples, annotations)

    assert result["wb_reward"] == 25.0
    assert result["per_baseline"] == {"b1": 50.0, "b2": 0.0}
    assert result["per_category"] == {
        "Creative Tasks": 75.0,
        "Math & Data Analysis": -25.0,
    }


def test_reward_skips_empty_marker_but_keeps_official_denominator():
    examples = pd.DataFrame(
        {
            "instruction_index": ["s1", "s2"],
            "task_categories": [["Creative Tasks"], ["Creative Tasks"]],
        }
    )
    annotations = pd.DataFrame(
        {
            "session_id": ["s1", "s2"],
            "baseline_model": ["b1", "b1"],
            "candidate_output": [EMPTY_RESPONSE, "answer"],
            "baseline_output": ["reference", "reference"],
            "reward": [1.0, 1.0],
        }
    )

    result = score_wildbench_reward_v2(examples, annotations)

    assert result["per_baseline"] == {"b1": 50.0}
    assert result["per_category"] == {"Creative Tasks": 100.0}
