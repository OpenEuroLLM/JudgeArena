"""Focused tests for canonical meta-evaluation annotation rows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.annotate as annotate_module
from judgearena.benchmarks.meta_eval.annotate import (
    _battle_texts,
    aggregate_battle_preferences,
    annotate_sample,
)
from judgearena.evaluate import JudgeAnnotation
from judgearena.prompts.parsing import ParsedPreference
from judgearena.prompts.registry import resolve_judge_prompt


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "battle_id": "arena:q1",
                "question_id": "q1",
                "model_a": "alpha",
                "model_b": "beta",
                "winner": "model_a",
                "lang": "en",
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
        judge=SimpleNamespace(
            swap_mode=swap_mode,
            strip_thinking_before_judging=False,
        ),
        generation=SimpleNamespace(truncate_judge_input_chars=8192),
        run=SimpleNamespace(use_tqdm=False),
    )


def _annotation(
    completion_a: str,
    completion_b: str,
    parsed: ParsedPreference | None,
) -> JudgeAnnotation:
    return JudgeAnnotation(
        instruction="Same prompt",
        completion_A=completion_a,
        completion_B=completion_b,
        judge_completion="judge output",
        judge_input="judge input",
        judge_top_logprobs={"token": -0.25},
        parsed=parsed,
    )


def _pass_row(battle_id: str, orientation: str, pref: float) -> dict[str, object]:
    return {
        "battle_id": battle_id,
        "orientation": orientation,
        "pref": pref,
    }


def test_battle_texts_accepts_parquet_array_conversations():
    sample = _sample()
    for column in ("conversation_a", "conversation_b"):
        sample.at[0, column] = np.asarray(sample.at[0, column], dtype=object)

    assert _battle_texts(sample) == (
        ["Same prompt"],
        ["Alpha answer"],
        ["Beta answer"],
    )


def test_annotation_uses_each_structured_parse_and_preserves_raw_evidence(
    monkeypatch,
):
    direct = _annotation(
        "Alpha answer",
        "Beta answer",
        ParsedPreference(
            preference=0.2,
            label="A",
            scores={"A": 9.0, "B": 1.0},
            details={"reason": "direct"},
        ),
    )
    reversed_pass = _annotation(
        "Beta answer",
        "Alpha answer",
        ParsedPreference(
            preference=0.8,
            label="B",
            scores={"A": 1.0, "B": 9.0},
            details={"reason": "reversed"},
        ),
    )

    def fake_judge_and_parse_prefs(**kwargs):
        assert kwargs["swap_mode"] == "both"
        return [direct], [reversed_pass], pd.Series([0.99, 0.01])

    monkeypatch.setattr(
        annotate_module, "judge_and_parse_prefs", fake_judge_and_parse_prefs
    )
    rows = annotate_sample(
        _sample(),
        _config("both"),
        judge_chat_model=object(),
        resolved_prompt=resolve_judge_prompt(preset="meta-eval-pair-score"),
    )

    assert list(rows.columns) == [
        "battle_id",
        "orientation",
        "pref",
        "judge_input",
        "judge_completion",
        "judge_top_logprobs_json",
        "parsed_label",
        "parsed_scores_json",
        "parsed_details_json",
    ]
    assert rows["orientation"].tolist() == ["direct", "reversed"]
    assert rows["pref"].tolist() == pytest.approx([0.99, 0.01])
    assert rows["parsed_label"].tolist() == ["A", "B"]
    assert json.loads(rows.loc[1, "parsed_scores_json"]) == {"A": 1.0, "B": 9.0}
    assert json.loads(rows.loc[1, "parsed_details_json"]) == {"reason": "reversed"}


def test_fixed_annotation_uses_single_orientation(monkeypatch):
    annotation = _annotation(
        "Alpha answer",
        "Beta answer",
        ParsedPreference(preference=0.75),
    )
    monkeypatch.setattr(
        annotate_module,
        "judge_and_parse_prefs",
        lambda **kwargs: ([annotation], None, pd.Series([0.75])),
    )

    rows = annotate_sample(
        _sample(),
        _config("fixed"),
        judge_chat_model=object(),
        resolved_prompt=resolve_judge_prompt(preset="meta-eval-pair-score"),
    )

    assert rows["orientation"].tolist() == ["single"]
    assert rows["pref"].tolist() == [0.75]


def test_aggregate_requires_every_pass_and_leaves_partial_evidence_raw():
    annotations = pd.DataFrame(
        [
            _pass_row("complete", "direct", 0.2),
            _pass_row("complete", "reversed", 0.4),
            _pass_row("partial", "direct", 0.5001),
            _pass_row("partial", "reversed", float("nan")),
            _pass_row("missing", "direct", float("nan")),
            _pass_row("missing", "reversed", float("nan")),
        ]
    )

    battles = aggregate_battle_preferences(annotations, swap_mode="both").set_index(
        "battle_id"
    )

    assert list(battles.reset_index().columns) == ["battle_id", "pref"]
    assert battles.loc["complete", "pref"] == pytest.approx(0.3)
    assert pd.isna(battles.loc["partial", "pref"])
    assert pd.isna(battles.loc["missing", "pref"])


def test_aggregate_rejects_incomplete_orientation_sets():
    annotations = pd.DataFrame([_pass_row("q1", "direct", 0.2)])

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
