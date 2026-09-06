"""Focused tests for meta-evaluation schema, prompts, parsers, and sampling."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

import judgearena.evaluate as evaluate_module
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    comparison_components,
    count_battles_per_model,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.prompts.parsing import JUDGE_PARSERS
from judgearena.prompts.registry import resolve_judge_prompt
from judgearena.tasks.schema import MetaEvalProtocol, TaskSpec


def _battles() -> pd.DataFrame:
    models = ["m1", "m2", "m3"]
    rows = [
        {
            "battle_id": f"arena:q{i}",
            "question_id": f"q{i}",
            "model_a": models[i % 3],
            "model_b": models[(i + 1) % 3],
        }
        for i in range(18)
    ]
    rows.append(
        {
            "battle_id": "arena:rare",
            "question_id": "rare",
            "model_a": "rare",
            "model_b": "m1",
        }
    )
    return pd.DataFrame(rows)


def test_sampling_is_deterministic_connected_and_limited_to_top_models():
    battles = _battles()
    top, top_pool = select_top_models(battles, top_models=3)

    sample = sample_battles_per_model(top_pool, top, battles_per_model=4, seed=7)
    shuffled = sample_battles_per_model(
        top_pool.sample(frac=1, random_state=42),
        top,
        battles_per_model=4,
        seed=7,
    )

    assert set(top) == {"m1", "m2", "m3"}
    assert "rare" not in set(top_pool["question_id"])
    assert sample["battle_id"].is_unique
    assert all(count >= 4 for count in count_battles_per_model(sample).values())
    assert comparison_components(sample, top) == [frozenset(top)]
    pd.testing.assert_frame_equal(sample, shuffled)


def test_sampling_rejects_disconnected_top_model_pool():
    battles = pd.DataFrame(
        [
            {"model_a": "a", "model_b": "b"},
            {"model_a": "c", "model_b": "d"},
        ]
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


def _meta_task() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "meta-test",
        "task_version": 1,
        "description": "Meta-evaluation schema test.",
        "dataset": {
            "adapter": "arena_battles",
            "sources": {
                "battles": {
                    "type": "huggingface_dataset",
                    "repo_id": "example/battles",
                    "revision": "a" * 40,
                }
            },
            "fields": {"id": "question_id", "instruction": "conversation_a"},
        },
        "protocol": {
            "runner": "meta_eval",
            "arena": "Test Arena",
            "baseline": {"strategy": "none"},
            "judge": {"default_prompt_preset": "meta-eval-pair-score"},
            "scoring": {
                "metrics": [
                    {
                        "metric": "meta_eval_agreement",
                        "parameters": {"n_bootstraps": 10, "tie_tolerance": 0.01},
                    }
                ]
            },
        },
    }


def test_meta_eval_protocol_uses_no_baseline_and_current_scoring_schema():
    definition = _meta_task()
    task = TaskSpec.model_validate(definition)

    assert isinstance(task.protocol, MetaEvalProtocol)
    assert task.protocol.baseline.strategy == "none"
    assert task.protocol.scoring.metrics[0].metric == "meta_eval_agreement"

    metric = definition["protocol"]["scoring"]["metrics"][0]  # type: ignore[index]
    metric["metric"] = "pairwise_win_rate"
    with pytest.raises(ValueError, match="unsupported meta-evaluation metric"):
        TaskSpec.model_validate(definition)

    metric["metric"] = "meta_eval_agreement"
    metric["group_by"] = ["lang"]
    with pytest.raises(ValueError, match="does not support group_by"):
        TaskSpec.model_validate(definition)


def test_meta_eval_protocol_rejects_a_model_baseline():
    definition = _meta_task()
    definition["protocol"]["baseline"] = {  # type: ignore[index]
        "strategy": "runtime_required"
    }

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
    ("preset", "parser_name"),
    [
        ("meta-eval-pair-score", "meta-eval-score"),
        ("meta-eval-alpaca-eval-json", "alpaca-eval-json"),
        ("meta-eval-alpaca-eval-pair-score", "meta-eval-score"),
    ],
)
def test_meta_eval_prompt_presets_select_their_parser(preset, parser_name):
    resolved = resolve_judge_prompt(preset=preset)

    assert resolved.parser is JUDGE_PARSERS[parser_name]
    assert resolved.system_prompt
    assert resolved.user_prompt_template


@pytest.mark.parametrize(
    "preset",
    ["meta-eval-alpaca-eval-json", "meta-eval-alpaca-eval-pair-score"],
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
