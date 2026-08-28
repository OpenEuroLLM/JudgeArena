import math

import pytest

from judgearena.prompts.parsing import (
    JUDGE_PARSERS,
    PairScore,
    weighted_token_preference,
)
from judgearena.prompts.registry import resolve_judge_prompt
from judgearena.utils import strip_thinking_tags

parse_arena_hard_verdict = JUDGE_PARSERS["arena-hard-verdict"]
parse_alpaca_eval_token = JUDGE_PARSERS["alpaca-eval-token"]


def test_pair_score():
    s = """
Answer: Model B
Explanation: While both models technically "failed" to provide a correct answer in the sense that they did not simply list 5 countries starting with S, Model A's response is clearly irrelevant and unhelpful. In contrast, although verbose, Model B actually attempted to fulfill the instruction.
Confidence: 0.85
Score_a: 0
Score_b: 1
"""
    score = PairScore()
    assert score.parse_model_raw(s) == pytest.approx(0.5744425168116589)


def test_pair_score2():
    s = """
Here is my judgement:

```
confidence: 0.99
score A: 10
score B: -5
```

In this case, Model A provided a correct and relevant response, listing two countries that start with S. On the other hand, Model B's response was completely irrelevant to the question asked, indicating a lack of understanding or ability to address the topic at hand. Therefore, Model A is significantly better than Model B in this scenario.
"""
    score = PairScore()
    assert score.parse_model_raw(s) == pytest.approx(0.010986942630593188)


@pytest.mark.parametrize(
    ("score_a", "score_b", "expected"),
    [(10_000, -10_000, 0.0), (-10_000, 10_000, 1.0), (10_000, 10_000, 0.5)],
)
def test_pair_score_is_bounded_for_extreme_scores(score_a, score_b, expected):
    preference = PairScore().preference_from_scores(score_a, score_b)

    assert math.isfinite(preference)
    assert 0.0 <= preference <= 1.0
    assert preference == expected


def test_regexp():
    raw_text = "Score of Assistant A: 0\nScore of Assistant B: 1\n```"

    scorer = PairScore()
    pref = scorer.parse_model_raw(raw_text)
    assert pref is not None
    assert pref == pytest.approx(0.5744425168116589)

    print(pref)


def test_default_prompt_preset_renders_answer_labels():
    resolved = resolve_judge_prompt(
        preset="default",
    )

    assert isinstance(resolved.parser, PairScore)
    assert "<|The Start of Assistant A's Answer|>" in resolved.user_prompt_template


def test_pair_score_ignores_scores_inside_thinking_tags():
    raw_text = """
    <think>
    Early draft:
    score_A: 2
    score_B: 1
    </think>
    Explanation: Assistant B is clearly better overall.
    score_A: 0
    score_B: 10
    """

    scorer = PairScore()
    pref = scorer.parse_model_raw(raw_text)

    assert pref is not None
    assert pref == pytest.approx(0.9525741268224333)


def test_pair_score_score_mode_ignores_bracketed_verdict_after_thinking():
    raw_text = """
    <think>
    score_A: 0
    score_B: 10
    </think>
    Concise verdict only.
    [[B]]
    """

    scorer = PairScore()

    assert scorer.parse_model_raw(raw_text) is None


def test_strip_thinking_tags_handles_closing_tag_without_opening_tag():
    raw_text = (
        "Reasoning that started implicitly and kept going.\n"
        "Still reasoning.\n"
        "</think>\n"
        "Final answer."
    )

    assert strip_thinking_tags(raw_text) == "Final answer."


@pytest.mark.parametrize(
    "judgment, expected",
    [
        ("My final verdict is tie: [[A=B]]", 0.5),
        ("Assistant A is significantly better: [[A>>B]]", 0.0),
        ("[[A>B]]", 0.25),
        ("[[B<A]]", 0.25),
        ("[[B<<A]]", 0.0),
        ("some explanation...\n[[B>A]]", 0.75),
        ("[[B>>A]]", 1.0),
        ("[[B=A]]", 0.5),  # symmetric spelling accepted by upstream
        ("[[A<B]]", 0.75),
        ("[[A<<B]]", 1.0),
        ("[A<<B]", 1.0),  # v2 single-bracket fallback
        ("[[A=B]] ... repeated [[A=B]]", 0.5),  # duplicates of one label are fine
        ("[[A>B]] but wait [[B>A]]", 0.75),  # last label wins, as upstream
        ("[[a>b]]", 0.25),  # matching is case-insensitive, as upstream
        ("no verdict here", None),
    ],
)
def test_parse_arena_hard_verdict(judgment, expected):
    assert parse_arena_hard_verdict(judgment) == expected


@pytest.mark.parametrize(
    "judgment, expected",
    [
        ("m", 0.0),
        ("M", 1.0),
        (" m\n", 0.0),  # whitespace-only padding is tolerated
        ("mM", None),
        ("model m", None),
        ("", None),
    ],
)
def test_parse_alpaca_eval_token(judgment, expected):
    assert parse_alpaca_eval_token(judgment) == expected


@pytest.mark.parametrize(
    "top_logprobs, expected",
    [
        ({"m": math.log(0.25), "M": math.log(0.75)}, 0.75),
        ({"M": -0.5}, 1.0),  # missing verdict token counts as -inf
        ({"m": -0.5}, 0.0),
        ({"x": -0.1}, None),  # neither verdict token present
        ({}, None),
    ],
)
def test_weighted_token_preference(top_logprobs, expected):
    result = weighted_token_preference(top_logprobs, ("m", "M"))
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_alpaca_eval_token_weights_by_logprobs_when_present():
    assert parse_alpaca_eval_token(
        "M", top_logprobs={"m": math.log(0.25), "M": math.log(0.75)}
    ) == pytest.approx(0.75)
    # Without logprobs the sampled token decides.
    assert parse_alpaca_eval_token("M") == 1.0


def test_alpaca_eval_preset_resolves_token_parser():
    resolved = resolve_judge_prompt(preset="alpaca-eval")

    assert resolved.parser is parse_alpaca_eval_token
    assert resolved.parser.requires_top_logprobs is True
    assert resolved.system_prompt.startswith("You are a highly efficient assistant")
    assert '"model_identifier": "m"' in resolved.user_prompt_template
    assert "{completion_A}" in resolved.user_prompt_template


def test_arena_hard_preset_resolves_verdict_parser():
    from judgearena.prompts.registry import resolve_judge_prompt

    resolved = resolve_judge_prompt(preset="arena-hard")

    assert resolved.parser is parse_arena_hard_verdict
    assert resolved.system_prompt.startswith(
        "Please act as an impartial judge and evaluate the quality"
    )
    assert "[[A>>B]]" in resolved.system_prompt
    assert "<|The Start of Assistant A's Answer|>" in resolved.user_prompt_template
