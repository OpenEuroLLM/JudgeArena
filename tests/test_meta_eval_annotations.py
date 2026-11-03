"""Canonical annotation rows and physical-battle aggregation."""

import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

from judgearena.benchmarks.meta_eval.annotate import (
    prepare_reference_comparisons,
)
from judgearena.benchmarks.meta_eval.runner import _prepare_arena_battles
from judgearena.benchmarks.meta_eval.scoring import aggregate_battle_preferences
from judgearena.benchmarks.pairwise.judging import prepare_pairwise_judging
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.cache.sqlite import JudgementCache
from judgearena.config import RunConfig
from judgearena.datasets.arena_battles import normalize_battles
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
    return RunConfig(
        task="meta-eval-comparia",
        judge={
            "model": "Dummy/judge",
            "swap_mode": swap_mode,
            "strip_thinking_before_judging": False,
        },
        generation={"truncate_judge_input_chars": 8192},
        run={"use_tqdm": False, "no_log_file": True},
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
    comparisons = prepare_reference_comparisons(sample)
    execute, _ = prepare_pairwise_judging(
        cfg, resolve_judge_prompt(preset="meta-eval-pair-score"), comparisons
    )
    rows, _, _ = execute(judge, comparisons)

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


@pytest.mark.parametrize("provided_id", [None, "adapter:comparison-1"])
def test_annotation_preserves_parser_details_and_replays_battle_ids(
    tmp_path, provided_id
):
    output = (
        '{"ordered_models": [{"model": "M", "rank": 1}, {"model": "m", "rank": 2}]}'
    )
    sample = _sample().drop(columns="battle_id")
    sample = sample.assign(instruction_index="q1", reference_pref=0.0)
    if provided_id is not None:
        sample["battle_id"] = provided_id
    sample = _prepare_arena_battles(
        normalize_battles(sample.assign(question_id="q1"), "arena"), task="test"
    )
    sample["instruction_id"] = "upstream-id"
    sample["conversation_a"] = "opaque source metadata"
    for replay in (False, True):
        judge = prepare_model(
            "Dummy/judge", cache=JudgementInferenceCache(tmp_path, "meta-eval")
        )
        judge.factory = (
            (lambda: pytest.fail("cache replay must not construct a provider"))
            if replay
            else lambda: FakeListLLM(responses=[output])
        )
        comparisons = prepare_reference_comparisons(sample)
        execute, _ = prepare_pairwise_judging(
            _config("fixed"),
            resolve_judge_prompt(preset="meta-eval-alpaca-eval-json"),
            comparisons,
        )
        rows, _, _ = execute(judge, comparisons)
    assert rows.loc[0, "conversation_a"] == "opaque source metadata"
    assert rows.loc[0, "parsed"]["label"] == "M"
    assert rows.loc[0, "parsed"]["details"] == {"ranks": {"M": 1, "m": 2}}
    battles = aggregate_battle_preferences(rows, swap_mode="fixed")
    assert battles["pref"].tolist() == [1.0]
    with JudgementCache(next(tmp_path.rglob("judgements.db"))) as cache:
        cached = cache.query()
    assert cached[
        ["instruction_id", "model_a", "model_b", "orientation"]
    ].values.tolist() == [[provided_id or "arena:q1", "alpha", "beta", "direct"]]


def test_aggregate_rejects_incomplete_orientation_sets():
    annotations = pd.DataFrame(
        {"battle_id": ["q1"], "orientation": ["direct"], "pref": [0.2]}
    )
    with pytest.raises(ValueError, match="expected.*direct.*reversed"):
        aggregate_battle_preferences(annotations, swap_mode="both")
