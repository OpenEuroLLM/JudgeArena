from judgearena.evaluate import PairScore
from judgearena.judge_prompt_presets import resolve_pairwise_judge_prompt


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


def test_pair_score_verdict_mode_uses_bracketed_verdicts():
    raw_text = "score_A: 10\nscore_B: 0\n[[B]]"
    scorer = PairScore(parser_mode="verdict")

    assert scorer.parse_model_raw(raw_text) == 1.0


def test_pair_score_verdict_mode_does_not_parse_score_only_outputs():
    raw_text = "score_A: 10\nscore_B: 0"
    scorer = PairScore(parser_mode="verdict")

    assert scorer.parse_model_raw(raw_text) is None


def test_localized_prompt_preset_renders_translated_labels():
    resolved = resolve_pairwise_judge_prompt(
        prompt_preset="m-arena-hard-v2-localized-zh",
        provide_explanation=False,
    )

    assert resolved.parser_mode == "score"
    assert "<|助手 A 的 回答 开始|>" in resolved.user_prompt_template
