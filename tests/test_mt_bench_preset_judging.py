from __future__ import annotations

import pandas as pd
import pytest

from judgearena.mt_bench.preset_judging import (
    _build_mt_bench_preset_items,
    _select_preset_prompt,
    judge_mt_bench_with_preset,
)
from judgearena.prompts.registry import FASTCHAT_PAIRWISE_PROMPT_PRESET


class SequenceJudge:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = []

    def batch(self, inputs, **_kwargs):
        self.calls.append(inputs)
        batch_outputs = self.outputs[: len(inputs)]
        self.outputs = self.outputs[len(inputs) :]
        return batch_outputs


def _questions_df(category: str = "writing") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": [category],
            "turn_1": ["Q1"],
            "turn_2": ["Q2"],
            "reference_turn_1": ["R1"],
            "reference_turn_2": ["R2"],
        },
        index=pd.Index([1], name="instruction_index"),
    )


def _completions_df(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "completion_turn_1": [f"{prefix}1"],
            "completion_turn_2": [f"{prefix}2"],
        },
        index=pd.Index([1], name="instruction_index"),
    )


def test_select_preset_prompt_rejects_delegated_preset():
    with pytest.raises(ValueError, match="delegated"):
        _select_preset_prompt(
            "writing",
            multi_turn=False,
            prompt_preset=FASTCHAT_PAIRWISE_PROMPT_PRESET,
            provide_explanation=False,
        )


@pytest.mark.parametrize(
    ("category", "multi_turn", "expected_name", "expected_ref_based"),
    [
        ("writing", False, "default-single", False),
        ("writing", True, "default-multi", False),
        ("math", False, "default-single_ref", True),
        ("math", True, "default-multi_ref", True),
    ],
)
def test_select_preset_prompt_variants(
    category: str,
    multi_turn: bool,
    expected_name: str,
    expected_ref_based: bool,
):
    prompt = _select_preset_prompt(
        category,
        multi_turn=multi_turn,
        prompt_preset="default",
        provide_explanation=False,
    )

    assert prompt.name == expected_name
    assert prompt.ref_based is expected_ref_based
    input_marker = "Conversation with User" if multi_turn else "[User Question]"
    assert input_marker in prompt.user_prompt_template
    assert "# Your output" in prompt.user_prompt_template


def test_build_mt_bench_preset_items_adds_turn_and_reference_kwargs():
    items = _build_mt_bench_preset_items(
        questions=_questions_df(category="math"),
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
        eval_single=True,
        eval_multi=True,
        truncate_input_chars=None,
        prompt_preset="default",
        provide_explanation=False,
    )

    assert [item.turn for item in items] == [1, 2]
    assert items[0].prompt_kwargs == {
        "question": "Q1",
        "answer_a": "A1",
        "answer_b": "B1",
        "ref_answer_1": "R1",
    }
    assert items[1].prompt_kwargs == {
        "question_1": "Q1",
        "question_2": "Q2",
        "answer_a_1": "A1",
        "answer_a_2": "A2",
        "answer_b_1": "B1",
        "answer_b_2": "B2",
        "ref_answer_1": "R1",
        "ref_answer_2": "R2",
    }


def test_judge_mt_bench_with_preset_parses_and_inverts_swapped_scores():
    judge = SequenceJudge(
        [
            "score_A: 10\nscore_B: 0",
            "score_A: 0\nscore_B: 10",
        ]
    )

    prefs, annotations, metadata = judge_mt_bench_with_preset(
        judge_chat_model=judge,
        judge_model="judge",
        questions=_questions_df(category="writing"),
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
        model_a="model-a",
        model_b="model-b",
        turns_mode="single",
        swap_mode="both",
        truncate_input_chars=None,
        use_tqdm=False,
        prompt_preset="default",
    )

    assert len(judge.calls) == 2
    assert len(prefs) == 2
    assert prefs.iloc[0] == pytest.approx(prefs.iloc[1])
    assert prefs.iloc[0] < 0.5
    assert annotations[0]["model_A"] == "model-a"
    assert annotations[1]["model_A"] == "model-b"
    assert annotations[1]["swapped"] is True
    assert "B1" in annotations[1]["user_prompt"]
    assert metadata == [
        {"question_id": 1, "category": "writing", "turn": 1},
        {"question_id": 1, "category": "writing", "turn": 1},
    ]
