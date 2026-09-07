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


def test_sampling_rejects_self_comparisons_and_insufficient_quota():
    self_comparison = pd.DataFrame(
        [
            {"battle_id": "self", "model_a": "a", "model_b": "a"},
            {"battle_id": "ab", "model_a": "a", "model_b": "b"},
        ]
    )
    with pytest.raises(MetaEvalSamplingError, match="self-comparisons"):
        select_top_models(self_comparison, top_models=2)
    battles = pd.DataFrame(
        [{"battle_id": f"q{i}", "model_a": "a", "model_b": "b"} for i in range(2)]
    )
    top, top_pool = select_top_models(battles, top_models=2)
    with pytest.raises(MetaEvalSamplingError, match="Insufficient unique battles"):
        sample_battles_per_model(top_pool, top, battles_per_model=3, seed=0)


def test_meta_eval_protocol_requires_no_baseline_and_compatible_metrics():
    definition = get_packaged_task("meta-eval-comparia").spec.model_dump()
    protocol = definition["protocol"]
    assert protocol["baseline"]["strategy"] == "none"

    metric = protocol["scoring"]["metrics"][0]
    metric["metric"] = "pairwise_win_rate"
    with pytest.raises(ValidationError, match="unsupported meta-evaluation metric"):
        TaskSpec.model_validate(definition)

    metric["metric"] = "meta_eval_agreement"
    metric["group_by"] = ["lang"]
    with pytest.raises(ValidationError, match="does not support group_by"):
        TaskSpec.model_validate(definition)

    metric["group_by"] = []
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


@pytest.mark.parametrize(
    "completion",
    [
        "score_A: 8.9\nscore_B: 8",
        "score_A: -1\nscore_B: 8",
        "score_A: 11\nscore_B: 8",
        "score_A: 0009\nscore_B: 1",
        "score_A: 8\nscore_B: 3.0",
    ],
)
def test_meta_eval_pair_score_rejects_invalid_scores(completion):
    assert JUDGE_PARSERS["meta-eval-score"].parse_result(completion) is None


def test_alpaca_eval_json_preserves_complete_ranks():
    result = JUDGE_PARSERS["alpaca-eval-json"].parse_result(
        '```json\n{"ordered_models": [{"model": "M", "rank": 1}, '
        '{"model": "m", "rank": 2}]}\n```'
    )

    assert result is not None
    assert result.preference == 1.0
    assert result.label == "M"
    assert result.details == {"ranks": {"M": 1, "m": 2}}


@pytest.mark.parametrize(
    "completion",
    [
        "[]",
        '{"ordered_models": ["m", "M"]}',
        '{"ordered_models": [{"model": "m", "rank": 1}]}',
        '{"ordered_models": [{"model": "m", "rank": 1}, {"model": "M", "rank": 1}]}',
        '{"ordered_models": [{"model": "m", "rank": true}, {"model": "M", "rank": 2}]}',
    ],
)
def test_alpaca_eval_json_rejects_malformed_rankings(completion):
    assert JUDGE_PARSERS["alpaca-eval-json"].parse_result(completion) is None


@pytest.mark.parametrize(
    "preset", ["meta-eval-alpaca-eval-json", "meta-eval-alpaca-eval-pair-score"]
)
def test_meta_eval_alpaca_prompts_embed_json_safe_inputs(preset, monkeypatch):
    captured_inputs = []

    def fake_do_inference(**kwargs):
        captured_inputs.extend(kwargs["inputs"])
        return ["unparsed"]

    monkeypatch.setattr(evaluate_module, "do_inference", fake_do_inference)
    instruction = 'Say "hi"\r\nnext \\ path {curly} café'
    completion_a = 'A says "yes"\nC:\\tmp {a}'
    completion_b = 'B says "no"\r\nD:\\tmp {b}'

    evaluate_module.annotate_battles(
        judge_chat_model=object(),
        instructions=[instruction],
        completions_A=[completion_a],
        completions_B=[completion_b],
        prompt_preset=preset,
        truncate_input_chars=None,
    )

    rendered = captured_inputs[0].messages[-1].content
    prompt_json = rendered.split("## Prompt\n\n", 1)[1].split(
        "\n\n## Model Outputs", 1
    )[0]
    outputs_json = rendered.split("## Model Outputs\n\n", 1)[1]
    outputs_json = outputs_json.split("\n\n## Task", 1)[0].split("\n\n", 1)[1]

    assert json.loads(prompt_json) == {"instruction": instruction}
    assert json.loads(outputs_json) == [
        {
            "model": "m" if preset.endswith("json") else "model A",
            "output": completion_a,
        },
        {
            "model": "M" if preset.endswith("json") else "model B",
            "output": completion_b,
        },
    ]
