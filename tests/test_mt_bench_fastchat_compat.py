from __future__ import annotations

import pandas as pd
import pytest

from judgearena.mt_bench.fastchat_compat import (
    _conservative_winner,
    _map_verdict_to_winner,
    _parse_fastchat_verdict,
    _select_prompt,
    judge_mt_bench_pairwise_fastchat,
)


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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Explanation [[A]]", "A"),
        ("Explanation [[B]]", "B"),
        ("Explanation [[C]]", "tie"),
        ("No bracketed verdict", "error"),
    ],
)
def test_parse_fastchat_verdict(raw: str, expected: str):
    assert _parse_fastchat_verdict(raw) == expected


def test_map_verdict_and_conservative_winner():
    assert _map_verdict_to_winner("A", swapped=False) == "model_A"
    assert _map_verdict_to_winner("A", swapped=True) == "model_B"
    assert _conservative_winner("model_A", "model_A") == ("model_A", False)
    assert _conservative_winner("model_A", "model_B") == ("tie", True)
    assert _conservative_winner("error", "model_B") == ("error", False)


@pytest.mark.parametrize(
    ("category", "multi_turn", "expected_name", "expected_ref_based"),
    [
        ("writing", False, "pair-v2", False),
        ("writing", True, "pair-v2-multi-turn", False),
        ("math", False, "pair-math-v1", True),
        ("math", True, "pair-math-v1-multi-turn", True),
    ],
)
def test_select_prompt_variants(
    category: str,
    multi_turn: bool,
    expected_name: str,
    expected_ref_based: bool,
):
    prompt = _select_prompt(category, multi_turn=multi_turn)

    assert prompt.name == expected_name
    assert prompt.ref_based is expected_ref_based
    input_marker = "Conversation with User" if multi_turn else "[User Question]"
    assert input_marker in prompt.user_prompt_template


def test_judge_mt_bench_pairwise_fastchat_swap_mode_both_is_conservative():
    judge = SequenceJudge(["[[A]]", "[[B]]"])

    prefs, annotations, metadata, num_inconsistent = judge_mt_bench_pairwise_fastchat(
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
    )

    assert num_inconsistent == 0
    assert len(judge.calls) == 2
    assert prefs.tolist() == [0.0]
    assert annotations[0]["g1_winner"] == "model_A"
    assert annotations[0]["g2_winner"] == "model_A"
    assert annotations[0]["final_winner"] == "model_A"
    assert "B1" in annotations[0]["g2_user_prompt"]
    assert metadata == [{"question_id": 1, "category": "writing", "turn": 1}]
