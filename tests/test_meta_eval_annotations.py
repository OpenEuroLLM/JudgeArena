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
