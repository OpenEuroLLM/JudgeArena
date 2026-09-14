"""Focused tests for meta-evaluation schema, prompts, parsers, and sampling."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

import judgearena.evaluate as evaluate_module
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.prompts.parsing import JUDGE_PARSERS
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import TaskSpec


def test_sampling_preserves_the_only_bridge_and_each_models_quota():
    rows = [
        (f"{a}{b}-{i}", a, b) for a, b in (("a", "b"), ("c", "d")) for i in range(6)
    ]
    # The dense pairs can meet their quotas without connecting to each other.
    rows += [("bridge", "b", "c"), ("rare", "rare", "a")]
    battles = pd.DataFrame(rows, columns=["battle_id", "model_a", "model_b"])
    top, pool = select_top_models(battles, top_models=4)
    sample = sample_battles_per_model(pool, top, battles_per_model=4, seed=7)
    shuffled = sample_battles_per_model(
        pool.sample(frac=1, random_state=42), top, battles_per_model=4, seed=7
    )

    counts = pd.concat([sample["model_a"], sample["model_b"]]).value_counts()
    assert set(counts.index) == set(top) == {"a", "b", "c", "d"}
    assert counts.min() >= 4
    assert sample["battle_id"].is_unique
    assert "bridge" in set(sample["battle_id"])
    pd.testing.assert_frame_equal(sample, shuffled)


def test_sampling_rejects_disconnected_top_model_pool():
    battles = pd.DataFrame(
        [{"model_a": "a", "model_b": "b"}, {"model_a": "c", "model_b": "d"}]
    )

    with pytest.raises(MetaEvalSamplingError, match="disconnected"):
        select_top_models(battles, top_models=4)


def test_sampling_rejects_insufficient_quota():
    battles = pd.DataFrame({"battle_id": ["ab"], "model_a": ["a"], "model_b": ["b"]})
    with pytest.raises(MetaEvalSamplingError, match="Insufficient unique battles"):
        sample_battles_per_model(battles, ["a", "b"], battles_per_model=2, seed=0)


def test_meta_eval_protocol_requires_no_baseline_and_compatible_metrics():
    definition = get_packaged_task("meta-eval-comparia").spec.model_dump()
    protocol = definition["protocol"]
    metric = protocol["scoring"]["metrics"][0]
    metric["metric"] = "pairwise_win_rate"
    with pytest.raises(ValidationError, match="unsupported meta-evaluation metric"):
        TaskSpec.model_validate(definition)

    metric["metric"] = "meta_eval_agreement"
    protocol["baseline"] = {"strategy": "runtime_required"}
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(definition)


def test_meta_eval_pair_score_preserves_scores():
    result = JUDGE_PARSERS["meta-eval-score"].parse_result(
        '<think>ignore score_A: 0</think>\n"score_A": 9,\n"score_B": 1'
    )

    assert result is not None
    assert result.preference == pytest.approx(0.01798620996)
    assert result.scores == {"A": 9.0, "B": 1.0}


def test_meta_eval_pair_score_rejects_fractional_scores():
    assert (
        JUDGE_PARSERS["meta-eval-score"].parse_result("score_A: 8.9\nscore_B: 8")
        is None
    )


def test_alpaca_eval_json_rejects_incomplete_rankings():
    assert (
        JUDGE_PARSERS["alpaca-eval-json"].parse_result(
            '{"ordered_models": [{"model": "m", "rank": 1}]}'
        )
        is None
    )


def test_meta_eval_alpaca_prompt_embeds_json_safe_inputs(monkeypatch):
    captured = []

    def fake_do_inference(**kwargs):
        captured.extend(kwargs["inputs"])
        return ["unparsed"]

    monkeypatch.setattr(evaluate_module, "do_inference", fake_do_inference)
    instruction = 'Say "hi"\r\nnext \\ path {curly} café'
    outputs = ['A says "yes"\nC:\\tmp {a}', 'B says "no"\r\nD:\\tmp {b}']
    evaluate_module.annotate_battles(
        judge_chat_model=object(),
        instructions=[instruction],
        completions_A=outputs[:1],
        completions_B=outputs[1:],
        prompt_preset="meta-eval-alpaca-eval-json",
        truncate_input_chars=None,
    )

    rendered = captured[0].messages[-1].content
    prompt_json = rendered.split("## Prompt\n\n", 1)[1].split(
        "\n\n## Model Outputs", 1
    )[0]
    outputs_json = rendered.split("## Model Outputs\n\n", 1)[1]
    outputs_json = outputs_json.split("\n\n## Task", 1)[0].split("\n\n", 1)[1]
    assert json.loads(prompt_json) == {"instruction": instruction}
    assert [row["output"] for row in json.loads(outputs_json)] == outputs
