import math

import pytest

from judgearena.prompts.parsing import (
    JUDGE_PARSERS,
    PairScore,
    ParsedPreference,
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
    ("score_a", "score_b", "expected"), [(10_000, -10_000, 0.0), (-10_000, 10_000, 1.0)]
)
def test_pair_score_does_not_overflow(score_a, score_b, expected):
    assert PairScore().preference_from_scores(score_a, score_b) == expected


def test_pair_score_returns_structured_preference():
    raw_text = "Score of Assistant A: 6\nScore of Assistant B: 8"
    parser = PairScore()
    parsed = parser.parse_result(raw_text)

    assert parsed.preference == pytest.approx(0.6456563062257954)
    assert parsed.scores == {"A": 6.0, "B": 8.0}
    assert parser(raw_text) == parser.parse_model_raw(raw_text) == parsed.preference


@pytest.mark.parametrize("preference", [-0.1, math.nan])
def test_parsed_preference_rejects_invalid_values(preference):
    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        ParsedPreference(preference=preference)


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
    ("judgment", "expected"),
    [("[[A=B]]", 0.5), ("[A<<B]", 1.0), ("no verdict here", None)],
)
def test_parse_arena_hard_verdict(judgment, expected):
    assert parse_arena_hard_verdict(judgment) == expected


def test_official_parsers_preserve_structured_evidence():
    arena = parse_arena_hard_verdict.parse_result("[[A>B]] then [[b>a]]")
    assert arena is not None
    assert (arena.preference, arena.label) == (0.75, "B>A")  # Last label wins.

    logprobs = {"m": math.log(0.25), "M": math.log(0.75)}
    alpaca = parse_alpaca_eval_token.parse_result("M", top_logprobs=logprobs)
    assert alpaca is not None
    assert alpaca.preference == pytest.approx(0.75)
    assert alpaca.scores == logprobs
    assert parse_alpaca_eval_token("M", top_logprobs={"M": -0.5}) == 1.0


@pytest.mark.parametrize("top_logprobs", [None, {"x": -0.1}])
def test_alpaca_eval_token_does_not_fall_back_to_text(top_logprobs):
    assert parse_alpaca_eval_token("M", top_logprobs=top_logprobs) is None


def test_official_presets_resolve_their_parsers():
    alpaca = resolve_judge_prompt(preset="alpaca-eval")
    assert alpaca.parser is parse_alpaca_eval_token
    assert alpaca.parser.requires_top_logprobs is True
    assert (
        resolve_judge_prompt(preset="arena-hard-creative").parser
        is parse_arena_hard_verdict
    )
