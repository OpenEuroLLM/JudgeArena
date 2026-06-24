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


CRITERIA = [
    "adherence",
    "helpfulness",
    "factuality",
    "completeness",
    "clarity",
    "fluency",
]


def test_criteria_parser_averages_all_axes():
    text = (
        "adherence: A=5 B=3\n"
        "helpfulness: A=4 B=2\n"
        "factuality: A=5 B=4\n"
        "completeness: A=4 B=3\n"
        "clarity: A=5 B=5\n"
        "fluency: A=3 B=4\n"
    )
    ps = PairScore(parser_mode="criteria", criteria_names=CRITERIA)
    sa, sb = ps.parse_raw_scores(text)
    assert sa == (5 + 4 + 5 + 4 + 5 + 3) / 6
    assert sb == (3 + 2 + 4 + 3 + 5 + 4) / 6


def test_criteria_parser_survives_one_mangled_axis():
    # 'helpfulness' line is garbled; the other 5 axes still parse.
    text = (
        "adherence: A=5 B=3\n"
        "helpfulness: <model refused to score>\n"
        "factuality: A=5 B=4\n"
        "completeness: A=4 B=3\n"
        "clarity: A=5 B=5\n"
        "fluency: A=3 B=4\n"
    )
    ps = PairScore(parser_mode="criteria", criteria_names=CRITERIA)
    sa, sb = ps.parse_raw_scores(text)
    assert sa == (5 + 5 + 4 + 5 + 3) / 5
    assert sb == (3 + 4 + 3 + 5 + 4) / 5


def test_criteria_parser_returns_none_when_nothing_parses():
    ps = PairScore(parser_mode="criteria", criteria_names=CRITERIA)
    assert ps.parse_raw_scores("no scores here") == (None, None)


def test_criteria_parser_rejects_out_of_range():
    # 7 is outside [1,5] -> that axis's A is dropped, not clamped.
    text = "adherence: A=7 B=3\nfluency: A=4 B=2\n"
    ps = PairScore(parser_mode="criteria", criteria_names=["adherence", "fluency"])
    sa, sb = ps.parse_raw_scores(text)
    assert sa == 4.0  # only fluency A survived
    assert sb == (3 + 2) / 2  # both B values in range


def test_parse_criteria_axes_returns_per_axis_values():
    text = "adherence: A=7 B=3\nfluency: A=4 B=2\n"
    axes = PairScore.parse_criteria_axes(text, ["adherence", "fluency"])
    # out-of-range A on adherence -> None; everything else preserved per axis
    assert axes == {
        "adherence_A": None,
        "adherence_B": 3.0,
        "fluency_A": 4.0,
        "fluency_B": 2.0,
    }


def test_criteria_axis_columns_frame_has_one_column_per_axis_side():
    from judgearena.evaluate import criteria_axis_columns

    completions = [
        "adherence: A=5 B=3\nfluency: A=4 B=2\n",
        "garbled output with no scores",
    ]
    frame = criteria_axis_columns(completions, ["adherence", "fluency"])
    assert list(frame.columns) == [
        "adherence_A",
        "adherence_B",
        "fluency_A",
        "fluency_B",
    ]
    assert len(frame) == 2
    assert frame.iloc[0]["adherence_A"] == 5.0 and frame.iloc[0]["fluency_B"] == 2.0
    # unparseable row -> all None
    assert frame.iloc[1].isna().all()


def test_criteria_parser_produces_preference():
    ps = PairScore(parser_mode="criteria", criteria_names=CRITERIA)
    pref = ps.parse_model_raw(
        "adherence: A=5 B=3\nhelpfulness: A=4 B=2\nfactuality: A=5 B=4\n"
        "completeness: A=4 B=3\nclarity: A=5 B=5\nfluency: A=3 B=4\n"
    )
    assert pref is not None and 0.0 <= pref <= 1.0


def test_run_path_builds_criteria_scorer_from_resolved_prompt():
    # Mirrors how the run path constructs the scorer from a resolved prompt.
    resolved = resolve_judge_prompt(preset="criteria")
    ps = PairScore(
        parser_mode=resolved.parser_mode,
        criteria_names=list(resolved.criteria_names),
    )
    pref = ps.parse_model_raw(
        "adherence: A=5 B=3\nhelpfulness: A=4 B=2\nfactuality: A=5 B=4\n"
        "completeness: A=4 B=3\nclarity: A=5 B=5\nfluency: A=3 B=4\n"
    )
    assert pref is not None and 0.0 <= pref <= 1.0


def test_strip_thinking_tags_handles_closing_tag_without_opening_tag():
    raw_text = (
        "Reasoning that started implicitly and kept going.\n"
        "Still reasoning.\n"
        "</think>\n"
        "Final answer."
    )

    assert strip_thinking_tags(raw_text) == "Final answer."
