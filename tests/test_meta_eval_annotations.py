"""Canonical annotation rows and physical-battle aggregation."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import judgearena.evaluate as evaluate
from judgearena.benchmarks.meta_eval.annotate import (
    _battle_texts,
    aggregate_battle_preferences,
    annotate_sample,
)
from judgearena.prompts.registry import resolve_judge_prompt


def _sample():
    return pd.DataFrame(
        [
            {
                "battle_id": "arena:q1",
                "model_a": "alpha",
                "model_b": "beta",
                "conversation_a": [
                    {"role": "user", "content": "Same prompt"},
                    {"role": "assistant", "content": "Alpha answer"},
                ],
                "conversation_b": [
                    {"role": "user", "content": "Same prompt"},
                    {"role": "assistant", "content": "Beta answer"},
                ],
            }
        ]
    )


def _config(swap_mode):
    return SimpleNamespace(
        judge=SimpleNamespace(swap_mode=swap_mode, strip_thinking_before_judging=False),
        generation=SimpleNamespace(truncate_judge_input_chars=8192),
        run=SimpleNamespace(use_tqdm=False),
    )


def test_annotation_swaps_and_aggregates_battles(monkeypatch):
    sample = pd.concat([_sample()] * 3, ignore_index=True)
    sample["battle_id"] = ["complete", "partial", "missing"]
    for column in ("conversation_a", "conversation_b"):
        sample[column] = sample[column].map(
            lambda turns: np.asarray(turns, dtype=object)
        )
    responses = iter(
        [
            ["score_A: 9\nscore_B: 1", "score_A: 5\nscore_B: 6", "unparseable"],
            ["score_A: 1\nscore_B: 7", "unparseable", "unparseable"],
        ]
    )
    monkeypatch.setattr(evaluate, "do_inference", lambda **kwargs: next(responses))
    rows = annotate_sample(
        sample,
        _config("both"),
        judge_chat_model=object(),
        resolved_prompt=resolve_judge_prompt(preset="meta-eval-pair-score"),
    )

    assert rows["orientation"].tolist() == ["direct"] * 3 + ["reversed"] * 3
    rendered = rows["judge_input"]
    assert rendered[0].index("Alpha answer") < rendered[0].index("Beta answer")
    assert rendered[3].index("Beta answer") < rendered[3].index("Alpha answer")
    assert rows.loc[[0, 3], "pref"].tolist() == pytest.approx(
        [0.01798620996, 0.04742587318]
    )
    assert json.loads(rows.loc[3, "parsed_scores_json"]) == {"A": 1.0, "B": 7.0}
    evidence = rows.loc[
        2, ["pref", "parsed_label", "parsed_scores_json", "parsed_details_json"]
    ]
    assert evidence.isna().all()
    battles = aggregate_battle_preferences(rows, swap_mode="both").set_index(
        "battle_id"
    )
    assert battles.loc["complete", "pref"] == pytest.approx(0.03270604157)
    assert battles.loc[["partial", "missing"], "pref"].isna().all()


def test_annotation_preserves_parser_label_and_details(monkeypatch):
    output = (
        '{"ordered_models": [{"model": "M", "rank": 1}, {"model": "m", "rank": 2}]}'
    )
    monkeypatch.setattr(evaluate, "do_inference", lambda **kwargs: [output])
    rows = annotate_sample(
        _sample(),
        _config("fixed"),
        judge_chat_model=object(),
        resolved_prompt=resolve_judge_prompt(preset="meta-eval-alpaca-eval-json"),
    )
    assert rows.loc[0, "parsed_label"] == "M"
    assert json.loads(rows.loc[0, "parsed_details_json"]) == {"ranks": {"M": 1, "m": 2}}
    battles = aggregate_battle_preferences(rows, swap_mode="fixed")
    assert battles.columns.tolist() == ["battle_id", "pref"]
    assert battles["pref"].tolist() == [1.0]


def test_aggregate_rejects_incomplete_orientation_sets():
    annotations = pd.DataFrame(
        {"battle_id": ["q1"], "orientation": ["direct"], "pref": [0.2]}
    )
    with pytest.raises(ValueError, match="expected.*direct.*reversed"):
        aggregate_battle_preferences(annotations, swap_mode="both")


def test_conversation_validation_requires_matching_prompts():
    sample = _sample()
    sample.at[0, "conversation_b"][0]["content"] = "Different prompt"
    with pytest.raises(ValueError, match="different user prompts"):
        _battle_texts(sample)


def test_three_way_choice_keeps_modal_tie_separate_from_soft_preference():
    annotations = pd.DataFrame(
        {
            "battle_id": ["q1", "q1"],
            "orientation": ["direct", "reversed"],
            "pref": [0.575, 0.575],
            "parsed_scores_json": [
                json.dumps({"A": 0.2, "B": 0.35, "tie": 0.45}),
                json.dumps({"A": 0.35, "B": 0.2, "tie": 0.45}),
            ],
            "parsed_details_json": ["{}", "{}"],
        }
    )

    battles = aggregate_battle_preferences(annotations, swap_mode="both")

    assert battles.to_dict("records") == [
        {"battle_id": "q1", "pref": pytest.approx(0.575), "hard_pref": 0.5}
    ]


def test_aggregate_uses_native_hard_distribution_after_swap():
    def scores(a, tie, both_bad, b):
        return json.dumps({"A": a, "tie": tie, "both_bad": both_bad, "B": b})

    annotations = pd.DataFrame(
        {
            "battle_id": ["q1", "q1"],
            "orientation": ["direct", "reversed"],
            "pref": [0.525, 0.525],
            "parsed_scores_json": [
                scores(0.2, 0.3, 0.25, 0.25),
                scores(0.25, 0.3, 0.25, 0.2),
            ],
            "parsed_details_json": [
                json.dumps({"hard_tie_threshold": 0.59}),
                json.dumps({"hard_tie_threshold": 0.59}),
            ],
        }
    )

    battles = aggregate_battle_preferences(annotations, swap_mode="both")

    assert battles.to_dict("records") == [
        {"battle_id": "q1", "pref": pytest.approx(0.525), "hard_pref": 1.0}
    ]


def test_aggregate_reorients_native_directional_score_levels():
    annotations = pd.DataFrame(
        {
            "battle_id": ["q1", "q1"],
            "orientation": ["direct", "reversed"],
            "pref": [0.425, 0.425],
            "parsed_scores_json": [
                json.dumps({"0": 0.1, "1": 0.5, "2": 0.1, "3": 0.2, "4": 0.1}),
                json.dumps({"0": 0.1, "1": 0.2, "2": 0.1, "3": 0.5, "4": 0.1}),
            ],
            "parsed_details_json": [
                json.dumps({"hard_preference_mode": "center_level"}),
                json.dumps({"hard_preference_mode": "center_level"}),
            ],
        }
    )

    battles = aggregate_battle_preferences(annotations, swap_mode="both")

    assert battles.to_dict("records") == [
        {"battle_id": "q1", "pref": pytest.approx(0.425), "hard_pref": 0.0}
    ]
