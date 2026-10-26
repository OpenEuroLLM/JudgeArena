"""Canonical annotation rows and physical-battle aggregation."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

from judgearena.benchmarks.execution import judge_prepared_comparisons
from judgearena.benchmarks.meta_eval.annotate import (
    aggregate_battle_preferences,
    prepare_reference_comparisons,
)
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.cache.sqlite import JudgementCache
from judgearena.models import prepare_model
from judgearena.prompts.registry import resolve_judge_prompt


def _sample():
    return pd.DataFrame(
        [
            {
                "battle_id": "arena:q1",
                "instruction": "Same prompt",
                "completion_a": "Alpha answer",
                "completion_b": "Beta answer",
                "model_a": "alpha",
                "model_b": "beta",
                "conversation_a": [
                    {"role": "user", "content": "Same prompt"},
                    {"role": "assistant", "content": "Alpha answer"},
                ],
                "conversation_b": [
                    {"role": "user", "content": "Same prompt"},
                    {"role": "assistant", "content": "Beta answer"},
                ],
            }
        ]
    )


def _config(swap_mode):
    return SimpleNamespace(
        judge=SimpleNamespace(swap_mode=swap_mode, strip_thinking_before_judging=False),
        generation=SimpleNamespace(truncate_judge_input_chars=8192),
        run=SimpleNamespace(use_tqdm=False),
    )


@pytest.mark.parametrize("strip_thinking", [False, True])
def test_annotation_swaps_and_aggregates_battles(strip_thinking):
    sample = pd.concat([_sample()] * 3, ignore_index=True)
    sample["battle_id"] = ["complete", "partial", "missing"]
    sample.index = [8, 3, 11]
    for column in ("conversation_a", "conversation_b"):
        sample[column] = sample[column].map(
            lambda turns: np.asarray(turns, dtype=object)
        )
    judge = FakeListLLM(
        responses=[
            "score_A: 9\nscore_B: 1",
            "score_A: 5\nscore_B: 6",
            "unparseable",
            "score_A: 1\nscore_B: 7",
            "unparseable",
            "unparseable",
        ]
    )
    sample["completion_a"] = "<think>private reasoning</think>Alpha answer"
    cfg = _config("both")
    cfg.judge.strip_thinking_before_judging = strip_thinking
    rows = judge_prepared_comparisons(
        cfg,
        judge,
        prepare_reference_comparisons(sample),
        resolve_judge_prompt(preset="meta-eval-pair-score"),
        swap_mode="both",
        use_tqdm=False,
    )

    assert rows["orientation"].tolist() == ["direct"] * 3 + ["reversed"] * 3
    assert rows["battle_id"].tolist() == ["complete", "partial", "missing"] * 2
    rendered = rows["judge_input"]
    assert ("private reasoning" in rendered[0]) is not strip_thinking
    assert rendered[0].index("Alpha answer") < rendered[0].index("Beta answer")
    assert rendered[3].index("Beta answer") < rendered[3].index("Alpha answer")
    assert rows.loc[[0, 3], "pref"].tolist() == pytest.approx(
        [0.01798620996, 0.04742587318]
    )
    assert rows.loc[3, "parsed"]["scores"] == {"A": 1.0, "B": 7.0}
    evidence = rows.loc[2, ["pref", "parsed"]]
    assert evidence.isna().all()
    battles = aggregate_battle_preferences(rows, swap_mode="both").set_index(
        "battle_id"
    )
    assert battles.loc["complete", "pref"] == pytest.approx(0.03270604157)
    assert battles.loc[["partial", "missing"], "pref"].isna().all()


def test_annotation_preserves_parser_label_and_details(tmp_path):
    output = (
        '{"ordered_models": [{"model": "M", "rank": 1}, {"model": "m", "rank": 2}]}'
    )
    judge = prepare_model(
        "Dummy/judge", cache=JudgementInferenceCache(tmp_path, "meta-eval")
    )
    judge.factory = lambda: FakeListLLM(responses=[output])
    sample = _sample().drop(columns=["conversation_a", "conversation_b"])
    sample["instruction_id"] = "upstream-id"
    rows = judge_prepared_comparisons(
        _config("fixed"),
        judge,
        prepare_reference_comparisons(sample),
        resolve_judge_prompt(preset="meta-eval-alpaca-eval-json"),
        swap_mode="fixed",
        use_tqdm=False,
    )
    assert rows.loc[0, "parsed"]["label"] == "M"
    assert rows.loc[0, "parsed"]["details"] == {"ranks": {"M": 1, "m": 2}}
    battles = aggregate_battle_preferences(rows, swap_mode="fixed")
    assert battles["pref"].tolist() == [1.0]
    with JudgementCache(next(tmp_path.rglob("judgements.db"))) as cache:
        cached = cache.query()
    assert cached[
        ["instruction_id", "model_a", "model_b", "orientation"]
    ].values.tolist() == [["arena:q1", "alpha", "beta", "direct"]]


def test_aggregate_rejects_incomplete_orientation_sets():
    annotations = pd.DataFrame(
        {"battle_id": ["q1"], "orientation": ["direct"], "pref": [0.2]}
    )
    with pytest.raises(ValueError, match="expected.*direct.*reversed"):
        aggregate_battle_preferences(annotations, swap_mode="both")


def test_conversation_validation_requires_matching_prompts():
    sample = _sample()
    sample.at[0, "conversation_b"][0]["content"] = "Different prompt"
    with pytest.raises(ValueError, match="different user prompts"):
        prepare_reference_comparisons(sample)
