import math

import pytest

from judgearena.prompts.parsing import PairScore, ParsedPreference
from judgearena.utils import strip_thinking_tags


def test_pair_score_returns_structured_preference():
    raw_text = "Score of Assistant A: 6\nScore of Assistant B: 8"
    parser = PairScore()
    parsed = parser.parse_result(raw_text)

    assert parsed.preference == pytest.approx(0.6456563062257954)
    assert parsed.scores == {"A": 6.0, "B": 8.0}
    assert parser(raw_text) == parser.parse_model_raw(raw_text) == parsed.preference


def test_pair_score_accepts_signed_scores():
    raw_text = "```\nconfidence: 0.99\nscore A: 10\nscore B: -5\n```"
    assert PairScore()(raw_text) == pytest.approx(0.010986942630593188)


@pytest.mark.parametrize(
    ("score_a", "score_b", "expected"), [(10_000, -10_000, 0.0), (-10_000, 10_000, 1.0)]
)
def test_pair_score_does_not_overflow(score_a, score_b, expected):
    assert PairScore().preference_from_scores(score_a, score_b) == expected


@pytest.mark.parametrize("preference", [-0.1, math.nan])
def test_parsed_preference_rejects_invalid_values(preference):
    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        ParsedPreference(preference=preference)


def test_pair_score_ignores_scores_inside_thinking_tags():
    raw_text = "<think>score_A: 2\nscore_B: 1</think>\nscore_A: 0\nscore_B: 10"
    assert PairScore()(raw_text) == pytest.approx(0.9525741268224333)


def test_pair_score_rejects_verdict_without_visible_scores():
    raw_text = "<think>score_A: 0\nscore_B: 10</think>\n[[B]]"
    assert PairScore()(raw_text) is None


def test_strip_thinking_tags_handles_closing_tag_without_opening_tag():
    assert (
        strip_thinking_tags("Implicit reasoning.\n</think>\nFinal answer.")
        == "Final answer."
    )
