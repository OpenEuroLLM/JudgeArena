from __future__ import annotations

import pandas as pd
import pytest

from judgearena.prompts.mt_bench import (
    _select_prompt,
)
from judgearena.prompts.parsing import FastChatVerdictParser

REFERENCE_CATEGORIES = ("math", "reasoning", "coding", "arena-hard-200")


@pytest.mark.parametrize(
    ("raw", "preference", "label"),
    [
        ("Explanation [[A]]", 0.0, "A"),
        ("Explanation [[B]]", 1.0, "B"),
        ("Explanation [[C]]", 0.5, "tie"),
        ("No bracketed verdict", None, None),
        ("<think>[[A]]</think>Concise [[B]]", 1.0, "B"),
        ("[[B]] then [[A]]", 0.0, "A"),
    ],
)
def test_parse_fastchat_verdict(raw, preference, label):
    from judgearena.prompts.parsing import ParsedPreference

    expected = None if preference is None else ParsedPreference(preference, label)
    assert FastChatVerdictParser().parse_result(raw) == expected


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
    prompt = _select_prompt(
        category,
        multi_turn=multi_turn,
        reference_categories=REFERENCE_CATEGORIES,
    )

    assert prompt.variant_name == expected_name
    assert ("{ref_answer_1}" in prompt.user_prompt_template) is expected_ref_based
    input_marker = "Conversation with User" if multi_turn else "[User Question]"
    assert input_marker in prompt.user_prompt_template


@pytest.mark.parametrize(
    "direct,reverse,expected,inconsistent",
    [
        (0, 0, 0, False),
        (0, 1, 0.5, True),
        (0.5, 0.5, 0.5, False),
        (0, None, None, False),
        (None, 0, None, False),
    ],
)
def test_swapped_agreement_requires_both_verdicts(
    direct, reverse, expected, inconsistent
):
    from judgearena.battles import agreement_outcomes

    frame = agreement_outcomes(
        pd.DataFrame(
            {
                "instruction_index": ["battle", "battle"],
                "orientation": ["direct", "reversed"],
                "pref": [direct, reverse],
            }
        )
    )
    assert len(frame) == 1
    assert (
        pd.isna(frame.iloc[0].pref)
        if expected is None
        else frame.iloc[0].pref == expected
    )
    assert bool(frame.iloc[0].inconsistent) == inconsistent
