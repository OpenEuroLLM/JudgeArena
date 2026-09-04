"""Focused tests for canonical meta-evaluation annotation rows."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.annotate as annotate_module
from judgearena.benchmarks.meta_eval.annotate import (
    aggregate_battle_preferences,
    annotate_sample,
    validate_battle_conversations,
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


def _pass_row(
    battle_id: str,
    orientation: str,
    pref: float,
    **overrides,
) -> dict[str, object]:
    row: dict[str, object] = {
        "battle_id": battle_id,
        "question_id": battle_id,
        "model_a": "a",
        "model_b": "b",
        "winner": "model_a",
        "lang": "en",
        "parse_ok": not pd.isna(pref),
        "pref": pref,
        "orientation": orientation,
    }
    row.update(overrides)
    return row


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
        # The shared function returns already-combined preferences. Annotation
        # production must instead read the structured result attached to each pass.
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

    assert rows["orientation"].tolist() == ["direct", "reversed"]
    assert rows["pref_judge_input"].tolist() == pytest.approx([0.2, 0.8])
    assert rows["pref"].tolist() == pytest.approx([0.2, 0.2])
    assert rows["completion_a"].tolist() == ["Alpha answer", "Alpha answer"]
    assert rows["completion_b"].tolist() == ["Beta answer", "Beta answer"]
    assert rows["judge_input_completion_a"].tolist() == [
        "Alpha answer",
        "Beta answer",
    ]
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


def test_annotation_rejects_unimplemented_random_swap_mode(monkeypatch):
    monkeypatch.setattr(
        annotate_module,
        "judge_and_parse_prefs",
        lambda **kwargs: pytest.fail("judge must not be called"),
    )

    with pytest.raises(ValueError, match="supports fixed or both"):
        annotate_sample(
            _sample(),
            _config("random"),
            judge_chat_model=object(),
            resolved_prompt=resolve_judge_prompt(preset="meta-eval-pair-score"),
        )


def test_aggregate_produces_one_physical_row_and_all_parse_statuses():
    annotations = pd.DataFrame(
        [
            _pass_row("complete", "direct", 0.2),
            _pass_row("complete", "reversed", 0.4),
            _pass_row("partial", "direct", 0.5001, winner="model_b"),
            _pass_row(
                "partial",
                "reversed",
                float("nan"),
                winner="model_b",
                parse_ok=False,
            ),
            _pass_row("missing", "direct", float("nan"), winner="tie", parse_ok=False),
            _pass_row(
                "missing", "reversed", float("nan"), winner="tie", parse_ok=False
            ),
        ]
    )

    battles = aggregate_battle_preferences(annotations, swap_mode="both").set_index(
        "battle_id"
    )

    assert len(battles) == 3
    assert battles.loc["complete", "pref"] == pytest.approx(0.3)
    assert battles.loc["complete", "pref_hard"] == 0.0
    assert battles.loc["complete", "parse_status"] == "complete"
    assert battles.loc["complete", "n_passes_parsed"] == 2
    assert battles.loc["partial", "pref"] == pytest.approx(0.5001)
    assert battles.loc["partial", "pref_hard"] == 1.0
    assert battles.loc["partial", "parse_status"] == "partial"
    assert battles.loc["partial", "n_passes_parsed"] == 1
    assert pd.isna(battles.loc["missing", "pref"])
    assert pd.isna(battles.loc["missing", "pref_hard"])
    assert battles.loc["missing", "parse_status"] == "missing"
    assert battles.loc["missing", "n_passes_expected"] == 2


def test_aggregate_rejects_incomplete_orientation_sets():
    annotations = pd.DataFrame([_pass_row("q1", "direct", 0.2)])

    with pytest.raises(ValueError, match="expected.*direct.*reversed"):
        aggregate_battle_preferences(annotations, swap_mode="both")


def test_aggregate_rejects_invalid_mode_and_conflicting_metadata():
    with pytest.raises(ValueError, match="supports fixed or both"):
        aggregate_battle_preferences(pd.DataFrame(), swap_mode="random")

    annotations = pd.DataFrame(
        [
            _pass_row("q1", "direct", 0.2),
            _pass_row("q1", "reversed", 0.2, model_b="other"),
        ]
    )
    with pytest.raises(ValueError, match="conflicting metadata.*model_b"):
        aggregate_battle_preferences(annotations, swap_mode="both")


@pytest.mark.parametrize(
    ("swap_mode", "direct", "reversed_passes", "message"),
    [
        ("fixed", [], None, "direct passes"),
        ("fixed", ["annotation"], ["annotation"], "returned reversed passes"),
        ("both", ["annotation"], None, "reversed passes"),
    ],
)
def test_annotation_validates_shared_judge_pass_contract(
    swap_mode, direct, reversed_passes, message, monkeypatch
):
    annotation = _annotation(
        "Alpha answer", "Beta answer", ParsedPreference(preference=0.25)
    )
    direct_annotations = [annotation for _ in direct]
    reversed_annotations = (
        None if reversed_passes is None else [annotation for _ in reversed_passes]
    )
    monkeypatch.setattr(
        annotate_module,
        "judge_and_parse_prefs",
        lambda **kwargs: (
            direct_annotations,
            reversed_annotations,
            pd.Series(dtype="float64"),
        ),
    )

    with pytest.raises(ValueError, match=message):
        annotate_sample(
            _sample(),
            _config(swap_mode),
            judge_chat_model=object(),
            resolved_prompt=resolve_judge_prompt(preset="meta-eval-pair-score"),
        )


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
        validate_battle_conversations(sample)
