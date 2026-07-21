"""Tests for the declarative WildBench protocol contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from judgearena.tasks.schema import WildBenchProtocol


def _protocol(mode: str) -> dict[str, object]:
    baseline: dict[str, object]
    if mode == "score":
        baseline = {"strategy": "none"}
    else:
        baseline = {
            "strategy": "official_outputs",
            "source": "official_outputs",
            "references": ["reference-a", "reference-b"],
            "allow_runtime_override": True,
        }
    return {
        "runner": "wildbench",
        "mode": mode,
        "generation": {"mode": "conversation_chat"},
        "baseline": baseline,
        "judge": {
            "default_prompt": f"wildbench-{mode}-v2",
            "default_swap_mode": "fixed",
            "allowed_swap_modes": ["fixed", "both"],
            "default_temperature": 0.0,
            "max_words_to_evaluate": 1000,
        },
        "scoring": {"adapter": f"wildbench-{mode}-v2"},
    }


@pytest.mark.parametrize("mode", ["score", "reward"])
def test_wildbench_protocol_accepts_supported_modes(mode):
    protocol = WildBenchProtocol.model_validate(_protocol(mode))

    assert protocol.mode == mode
    assert protocol.generation.mode == "conversation_chat"
    assert protocol.judge.max_words_to_evaluate == 1000


def test_wildbench_score_rejects_official_baselines():
    definition = _protocol("score")
    definition["baseline"] = _protocol("reward")["baseline"]

    with pytest.raises(ValidationError, match="score mode does not use a baseline"):
        WildBenchProtocol.model_validate(definition)


def test_wildbench_reward_requires_reference_ids():
    definition = _protocol("reward")
    definition["baseline"]["references"] = []

    with pytest.raises(ValidationError, match="at least one baseline reference"):
        WildBenchProtocol.model_validate(definition)


def test_wildbench_score_rejects_length_penalty():
    definition = _protocol("score")
    definition["scoring"]["default_length_penalty_chars"] = 500

    with pytest.raises(ValidationError, match="does not support a length penalty"):
        WildBenchProtocol.model_validate(definition)
