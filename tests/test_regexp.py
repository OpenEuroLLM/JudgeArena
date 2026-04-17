from judgearena.evaluate import PairScore
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
    assert score.parse_model_raw(s) == 0.010986942630593188


def test_regexp():
    raw_text = "Score of Assistant A: 0\nScore of Assistant B: 1\n```"

    scorer = PairScore()
    pref = scorer.parse_model_raw(raw_text)
    assert pref is not None
    assert pref == 0.5744425168116589

    print(pref)


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


def test_pair_score_score_mode_does_not_parse_bracketed_verdicts():
    scorer = PairScore()

    assert scorer.parse_model_raw("Explanation: ok\n[[A]]") is None
    assert scorer.parse_model_raw("Explanation: ok\n[[B]]") is None
    assert scorer.parse_model_raw("Explanation: ok\n[[C]]") is None


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


def test_pair_score_verdict_mode_uses_bracketed_verdicts():
    raw_text = "score_A: 10\nscore_B: 0\n[[B]]"

    scorer = PairScore(parser_mode="verdict")

    assert scorer.parse_model_raw(raw_text) == 1.0


def test_pair_score_verdict_mode_does_not_parse_score_only_outputs():
    raw_text = "score_A: 10\nscore_B: 0"

    scorer = PairScore(parser_mode="verdict")

    assert scorer.parse_model_raw(raw_text) is None
