import hashlib
import math

import pytest

from judgearena.prompts.parsing import (
    JUDGE_PARSERS,
    JudgeParser,
    PairScore,
    ParsedPreference,
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


def test_pair_score_returns_structured_preference():
    raw_text = "Score of Assistant A: 6\nScore of Assistant B: 8"

    parser = PairScore()
    parsed = parser.parse_result(raw_text)

    assert parsed is not None
    assert parsed.preference == pytest.approx(0.6456563062257954)
    assert parsed.scores == {"A": 6.0, "B": 8.0}
    assert parsed.label is None
    assert parsed.details == {}
    assert parser(raw_text) == parsed.preference


class LegacyScalarParser(JudgeParser):
    name = "legacy"

    def __call__(self, judge_completion, *, top_logprobs=None):
        return 0.75


def test_legacy_scalar_parser_gets_a_structured_result():
    parsed = LegacyScalarParser().parse_result("ignored")

    assert parsed is not None
    assert parsed.preference == 0.75
    assert parsed.scores == {}


@pytest.mark.parametrize("preference", [-0.1, 1.1, math.inf, math.nan])
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


def test_alpaca_eval_token_requires_and_weights_logprobs():
    assert parse_alpaca_eval_token(
        "M", top_logprobs={"m": math.log(0.25), "M": math.log(0.75)}
    ) == pytest.approx(0.75)
    for top_logprobs in (None, {}):
        with pytest.raises(ValueError, match="requires first-token top logprobs"):
            parse_alpaca_eval_token("M", top_logprobs=top_logprobs)


def test_alpaca_eval_preset_resolves_token_parser():
    resolved = resolve_judge_prompt(preset="alpaca-eval")

    assert resolved.parser is parse_alpaca_eval_token
    assert resolved.parser.requires_top_logprobs is True
    assert hashlib.sha256(resolved.system_prompt.encode()).hexdigest() == (
        "247a5fed0ca2aafd7b99cc3b1dc174723695dcedbc5772fe97a16f2a5549f131"
    )
    assert hashlib.sha256(resolved.user_prompt_template.encode()).hexdigest() == (
        "eec8eae642cfce39f6f6820f1c46bcfb10e5923686a57ee8195ade94ae6b0b6b"
    )


@pytest.mark.parametrize(
    ("preset", "system_sha256"),
    [
        (
            "arena-hard",
            "03f68010febd7a6405102ef882b4dd5a9700c56b2e1ff286d3b38f5d3a929bbf",
        ),
        (
            "arena-hard-creative",
            "171489efbca7f19fb520e1b3c23783c76af7c723e770fe897a55f391684aa358",
        ),
    ],
)
def test_arena_hard_presets_match_upstream_bytes(preset, system_sha256):
    resolved = resolve_judge_prompt(preset=preset)

    assert resolved.parser is parse_arena_hard_verdict
    assert hashlib.sha256(resolved.system_prompt.encode()).hexdigest() == system_sha256
    assert hashlib.sha256(resolved.user_prompt_template.encode()).hexdigest() == (
        "82fc205856141b8751263b96f25b345b4f0be39c1548ba067f5ac0e3f224b5c9"
    )
