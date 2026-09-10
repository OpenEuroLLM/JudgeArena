"""Focused tests for canonical meta-evaluation annotation rows."""

from __future__ import annotations

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


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "battle_id": "arena:q1",
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


def _config(swap_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        judge=SimpleNamespace(swap_mode=swap_mode, strip_thinking_before_judging=False),
        generation=SimpleNamespace(truncate_judge_input_chars=8192),
        run=SimpleNamespace(use_tqdm=False),
    )


@pytest.mark.parametrize("swap_mode", ["fixed", "both"])
def test_annotation_parses_swaps_and_aggregates_physical_battles(
    monkeypatch, swap_mode
):
    sample = pd.concat([_sample()] * 3, ignore_index=True)
    sample["battle_id"] = ["complete", "partial", "missing"]
    for column in ("conversation_a", "conversation_b"):
        sample[column] = sample[column].map(
            lambda turns: np.asarray(turns, dtype=object)
        )
    prompt = resolve_judge_prompt(preset="meta-eval-pair-score")
    direct = ["score_A: 9\nscore_B: 1", "score_A: 5\nscore_B: 6", "unparseable"]
    reversed_outputs = ["score_A: 1\nscore_B: 7", "unparseable", "unparseable"]
    responses = iter([direct, reversed_outputs])
    rendered = []

    def fake_inference(*, inputs, **kwargs):
        rendered.extend(value.to_string() for value in inputs)
        return next(responses)

    monkeypatch.setattr(evaluate, "do_inference", fake_inference)
    rows = annotate_sample(
        sample, _config(swap_mode), judge_chat_model=object(), resolved_prompt=prompt
    )

    both = swap_mode == "both"
    assert rows["orientation"].tolist() == (
        ["direct"] * 3 + ["reversed"] * 3 if both else ["single"] * 3
    )
    assert rows["judge_input"].tolist() == rendered
    for i, value in enumerate(rendered):
        assert "Same prompt" in value
        assert (value.index("Alpha answer") < value.index("Beta answer")) == (i < 3)
    assert rows["judge_completion"].tolist() == direct + (
        reversed_outputs if both else []
    )
    direct_pref = prompt.parser.preference_from_scores(9, 1)
    partial_pref = prompt.parser.preference_from_scores(5, 6)
    assert pd.isna(rows.loc[2, "pref"])
    missing_evidence = rows.loc[
        2, ["parsed_label", "parsed_scores_json", "parsed_details_json"]
    ]
    assert missing_evidence.isna().all()
    assert json.loads(rows.loc[0, "parsed_scores_json"]) == {"A": 9.0, "B": 1.0}
    expected = direct_pref
    if both:
        reversed_pref = 1 - prompt.parser.preference_from_scores(1, 7)
        assert rows.loc[3, "pref"] == pytest.approx(reversed_pref)
        assert rows.loc[4:, "pref"].isna().all()
        assert json.loads(rows.loc[3, "parsed_scores_json"]) == {"A": 1.0, "B": 7.0}
        expected = (direct_pref + reversed_pref) / 2
    battles = aggregate_battle_preferences(rows, swap_mode=swap_mode).set_index(
        "battle_id"
    )
    assert rows.loc[:1, "pref"].tolist() == pytest.approx([direct_pref, partial_pref])
    assert battles.loc["complete", "pref"] == pytest.approx(expected)
    if both:
        assert pd.isna(battles.loc["partial", "pref"])
    else:
        assert battles.loc["partial", "pref"] == pytest.approx(partial_pref)
    assert pd.isna(battles.loc["missing", "pref"])


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
    assert rows.loc[0, "pref"] == 1.0
    assert rows.loc[0, "parsed_label"] == "M"
    assert json.loads(rows.loc[0, "parsed_details_json"]) == {"ranks": {"M": 1, "m": 2}}


def test_aggregate_rejects_incomplete_orientation_sets():
    annotations = pd.DataFrame(
        {"battle_id": ["q1"], "orientation": ["direct"], "pref": [0.2]}
    )

    with pytest.raises(ValueError, match="expected.*direct.*reversed"):
        aggregate_battle_preferences(annotations, swap_mode="both")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda sample: sample.at[0, "conversation_b"].pop(),
            "requires user and assistant turns",
        ),
        (
            lambda sample: sample.at[0, "conversation_a"][0].update(
                {"role": "assistant"}
            ),
            "requires user and assistant turns",
        ),
        (
            lambda sample: sample.at[0, "conversation_b"][0].update(
                {"content": "Different prompt"}
            ),
            "different user prompts",
        ),
    ],
)
def test_conversation_validation_requires_matching_user_assistant_pairs(
    mutate, message
):
    sample = _sample()
    mutate(sample)

    with pytest.raises(ValueError, match=message):
        _battle_texts(sample)
