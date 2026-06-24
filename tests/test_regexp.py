from judgearena.evaluate import PairScore
from judgearena.prompts.registry import resolve_judge_prompt
from judgearena.utils import strip_thinking_tags


def test_pair_score():
    s = """
Answer: Model B
Explanation: While both models technically "failed" to provide a correct answer in the sense that they did not simply list 5 countries starting with S, Model A's response is clearly irrelevant and unhelpful. In contrast, although verbose, Model B actually attempted to fulfill the instruction.
Confidence: 0.85
Score_a: 0
Score_b: 1
"""
    score = PairScore()
    assert score.parse_model_raw(s) == 0.5744425168116589


def test_pair_score2():
    # score B: -5 is outside the rubric range [0, 10] and is treated as a mis-grab.
    # parse_model_raw returns None when either side cannot be parsed.
    s = """
Here is my judgement:

```
confidence: 0.99
score A: 10
score B: -5
```

In this case, Model A provided a correct and relevant response, listing two countries that start with S. On the other hand, Model B's response was completely irrelevant to the question asked, indicating a lack of understanding or ability to agree to the topic at hand. Therefore, Model A is significantly better than Model B in this scenario.
"""
    score = PairScore()
    assert score.parse_model_raw(s) is None


def test_regexp():
    raw_text = "Score of Assistant A: 0\nScore of Assistant B: 1\n```"

    scorer = PairScore()
    pref = scorer.parse_model_raw(raw_text)
    assert pref is not None
    assert pref == 0.5744425168116589

    print(pref)


def test_default_prompt_preset_renders_answer_labels():
    resolved = resolve_judge_prompt(
        preset="default",
        provide_explanation=False,
    )

    assert resolved.parser_mode == "score"
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
    assert pref == 0.9525741268224333


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


# --- Range-validation tests (rubric: 0-10) ---


def test_parse_raw_scores_valid_in_range():
    """Standard valid scores stay intact."""
    sa, sb = PairScore.parse_raw_scores("score_a: 8\nscore_b: 9")
    assert sa == 8.0
    assert sb == 9.0


def test_parse_raw_scores_misgrab_out_of_range_returns_none():
    """A completion where the regex grabs a year (2024) as score_a must be rejected.

    The text is crafted so the loose regex r'score.*?a[": *\\n]*(-?\\d+)' matches
    '2024' before the real score line, replicating the real mis-grab scenario.
    """
    # The phrase 'scored a: 2024' comes before the real label, so the loose
    # regex grabs 2024 for side A.  Side B has a valid score (7).
    text = (
        "In 2023, assistant scored a: 2024 exceptional result.\n"
        "score_b: 7\n"
    )
    sa, sb = PairScore.parse_raw_scores(text)
    assert sa is None, f"Expected None for out-of-range score_a (2024), got {sa}"
    assert sb == 7.0


def test_parse_raw_scores_zero_kept_negative_rejected():
    """Zero is valid (lower rubric boundary); negative values are rejected."""
    sa, sb = PairScore.parse_raw_scores("score_a: 0\nscore_b: -5")
    assert sa == 0.0, f"Expected 0.0 for score_a, got {sa}"
    assert sb is None, f"Expected None for out-of-range score_b (-5), got {sb}"
