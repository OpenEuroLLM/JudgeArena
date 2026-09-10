from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.mt_bench.runner as mt_bench_runner
from judgearena.benchmarks.elo.rating import fit_bradley_terry
from judgearena.benchmarks.mt_bench.preset_judging import (
    _build_mt_bench_preset_items,
    _select_preset_prompt,
    judge_mt_bench_with_preset,
)
from judgearena.benchmarks.mt_bench.runner import _build_mt_bench_battles
from judgearena.benchmarks.pairwise.scoring.metrics import (
    LengthControlledWinrateMetric,
)
from judgearena.config import RunConfig
from judgearena.prompts.registry import (
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    resolve_judge_prompt,
)
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import MetricSpec, ScoringSpec

REFERENCE_CATEGORIES = ("math", "reasoning", "coding", "arena-hard-200")


class SequenceJudge:
    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.calls = []

    def batch(self, inputs, **_kwargs):
        self.calls.append(inputs)
        batch_outputs = self.outputs[: len(inputs)]
        self.outputs = self.outputs[len(inputs) :]
        return batch_outputs


def _questions_df(category: str = "writing") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": [category],
            "turn_1": ["Q1"],
            "turn_2": ["Q2"],
            "reference_turn_1": ["R1"],
            "reference_turn_2": ["R2"],
        },
        index=pd.Index([1], name="instruction_index"),
    )


def _completions_df(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "completion_turn_1": [f"{prefix}1"],
            "completion_turn_2": [f"{prefix}2"],
        },
        index=pd.Index([1], name="instruction_index"),
    )


def test_select_preset_prompt_rejects_delegated_preset():
    with pytest.raises(ValueError, match="delegated"):
        _select_preset_prompt(
            "writing",
            multi_turn=False,
            reference_categories=REFERENCE_CATEGORIES,
            prompt_preset=FASTCHAT_PAIRWISE_PROMPT_PRESET,
        )


@pytest.mark.parametrize(
    ("category", "multi_turn", "expected_name", "expected_ref_based"),
    [
        ("writing", False, "default-single", False),
        ("writing", True, "default-multi", False),
        ("math", False, "default-single_ref", True),
        ("math", True, "default-multi_ref", True),
    ],
)
def test_select_preset_prompt_variants(
    category: str,
    multi_turn: bool,
    expected_name: str,
    expected_ref_based: bool,
):
    prompt = _select_preset_prompt(
        category,
        multi_turn=multi_turn,
        reference_categories=REFERENCE_CATEGORIES,
        prompt_preset="default",
    )

    assert prompt.name == expected_name
    assert prompt.ref_based is expected_ref_based
    input_marker = "Conversation with User" if multi_turn else "[User Question]"
    assert input_marker in prompt.user_prompt_template
    assert "# Your output" in prompt.user_prompt_template


def test_build_mt_bench_preset_items_adds_turn_and_reference_kwargs():
    items = _build_mt_bench_preset_items(
        questions=_questions_df(category="math"),
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
        eval_single=True,
        eval_multi=True,
        truncate_input_chars=None,
        reference_categories=REFERENCE_CATEGORIES,
        prompt_preset="default",
    )

    assert [item.turn for item in items] == [1, 2]
    assert items[0].prompt_kwargs == {
        "question": "Q1",
        "answer_a": "A1",
        "answer_b": "B1",
        "ref_answer_1": "R1",
    }
    assert items[1].prompt_kwargs == {
        "question_1": "Q1",
        "question_2": "Q2",
        "answer_a_1": "A1",
        "answer_a_2": "A2",
        "answer_b_1": "B1",
        "answer_b_2": "B2",
        "ref_answer_1": "R1",
        "ref_answer_2": "R2",
    }


def test_judge_mt_bench_with_preset_parses_and_inverts_swapped_scores():
    judge = SequenceJudge(
        [
            "score_A: 10\nscore_B: 0",
            "score_A: 0\nscore_B: 10",
        ]
    )

    prefs, annotations, metadata = judge_mt_bench_with_preset(
        judge_chat_model=judge,
        judge_model="judge",
        questions=_questions_df(category="writing"),
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
        model_a="model-a",
        model_b="model-b",
        turns_mode="single",
        swap_mode="both",
        truncate_input_chars=None,
        use_tqdm=False,
        reference_categories=REFERENCE_CATEGORIES,
        prompt_preset="default",
    )

    assert len(judge.calls) == 2
    assert len(prefs) == 2
    assert prefs.iloc[0] == pytest.approx(prefs.iloc[1])
    assert prefs.iloc[0] < 0.5
    assert annotations[0]["model_A"] == "model-a"
    assert annotations[0]["parsed"]["scores"] == {"A": 10.0, "B": 0.0}
    assert annotations[0]["preference"] < 0.5
    assert annotations[1]["model_A"] == "model-b"
    assert annotations[1]["parsed"]["scores"] == {"A": 0.0, "B": 10.0}
    assert annotations[1]["preference"] > 0.5
    assert annotations[1]["swapped"] is True
    assert "B1" in annotations[1]["user_prompt"]
    assert metadata == [
        {
            "question_id": 1,
            "category": "writing",
            "turn": 1,
            "orientation": "direct",
        },
        {
            "question_id": 1,
            "category": "writing",
            "turn": 1,
            "orientation": "reversed",
        },
    ]
    battles = _build_mt_bench_battles(
        cfg=SimpleNamespace(
            model=SimpleNamespace(name="model-a", baseline="model-b"),
            judge=SimpleNamespace(model="judge"),
        ),
        prefs=prefs,
        combined_metadata=metadata,
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
    )
    metric = LengthControlledWinrateMetric().calculate(battles)
    assert metric == {"num_pairs": 1, "num_scored": 1, "winrate": None}


def test_fixed_preset_judgment_builds_one_single_orientation_battle():
    prefs, _, metadata = judge_mt_bench_with_preset(
        judge_chat_model=SequenceJudge(["score_A: 10\nscore_B: 0"]),
        judge_model="judge",
        questions=_questions_df(category="writing"),
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
        model_a="model-a",
        model_b="model-b",
        turns_mode="single",
        swap_mode="fixed",
        truncate_input_chars=None,
        use_tqdm=False,
        reference_categories=REFERENCE_CATEGORIES,
        prompt_preset="default",
    )

    assert metadata[0]["orientation"] == "single"
    battles = _build_mt_bench_battles(
        cfg=SimpleNamespace(
            model=SimpleNamespace(name="model-a", baseline="model-b"),
            judge=SimpleNamespace(model="judge"),
        ),
        prefs=prefs,
        combined_metadata=metadata,
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
    )
    assert LengthControlledWinrateMetric().calculate(battles) == {
        "num_pairs": 1,
        "num_scored": 1,
        "winrate": None,
    }


def test_mt_bench_battles_preserve_preferences_and_turn_ids():
    prefs = pd.Series([0.2, 0.8, 0.5, np.nan])
    metadata = [
        {"question_id": question_id, "turn": turn}
        for question_id, turn in [(1, 1), (1, 2), (2, 1), (2, 2)]
    ]
    battles = _build_mt_bench_battles(
        cfg=SimpleNamespace(
            model=SimpleNamespace(name="candidate", baseline="reference"),
            judge=SimpleNamespace(model="judge"),
        ),
        prefs=prefs,
        combined_metadata=metadata,
        completions_a=pd.DataFrame(
            {"completion_turn_1": ["a1", "a2"], "completion_turn_2": ["a1b", "a2b"]},
            index=[1, 2],
        ),
        completions_b=pd.DataFrame(
            {"completion_turn_1": ["b1", "b2"], "completion_turn_2": ["b1b", "b2b"]},
            index=[1, 2],
        ),
    )

    pd.testing.assert_series_equal(battles["pref"], prefs.rename("pref"))
    pd.testing.assert_series_equal(
        battles["pref_hard"], pd.Series([0.0, 1.0, 0.5, np.nan], name="pref_hard")
    )
    assert {
        "instruction_index",
        "model",
        "baseline",
        "completion_model",
        "completion_baseline",
        "orientation",
        "pref",
    } <= set(battles)
    assert battles["instruction_index"].tolist() == [
        "1:turn-1",
        "1:turn-2",
        "2:turn-1",
        "2:turn-2",
    ]


@pytest.mark.parametrize("seed", [17, 29])
def test_mt_bench_hard_bootstraps_use_run_seed(monkeypatch, tmp_path, seed):
    monkeypatch.setattr(
        mt_bench_runner, "write_run_metadata_safely", lambda **_kwargs: None
    )
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "candidate", "baseline": "reference"},
        judge={"model": "judge", "prompt_preset": "default", "swap_mode": "fixed"},
        run={"seed": seed, "use_tqdm": False},
    )
    protocol = get_packaged_task(cfg.task).spec.protocol.model_copy(
        update={
            "scoring": ScoringSpec(
                metrics=(
                    MetricSpec(
                        metric="bradley_terry",
                        parameters={"soft": False, "n_bootstraps": 3},
                    ),
                )
            )
        }
    )

    mt_bench_runner._run_mt_bench_preset(
        cfg=cfg,
        protocol=protocol,
        res_folder=tmp_path,
        result_name="result",
        questions_df=_questions_df(),
        completions_a=_completions_df("A"),
        completions_b=_completions_df("B"),
        judge_chat_model=SequenceJudge(
            ["score A: 10 score B: 0", "score A: 0 score B: 10"]
        ),
        resolved_prompt=resolve_judge_prompt(preset="default"),
        started_at_utc=datetime.now(UTC),
    )

    battles = pd.DataFrame(
        {"model_a": "candidate", "model_b": "reference", "pref": [0.0, 1.0]}
    )
    rng = np.random.default_rng(seed)
    expected = [
        fit_bradley_terry(
            battles.sample(
                n=len(battles), replace=True, random_state=int(rng.integers(0, 2**31))
            )
        )
        for _ in range(3)
    ]
    saved = json.loads((tmp_path / "results-result.json").read_text())
    metric = saved["metrics"]["bradley_terry"]
    assert metric["method"] == "ELO"
    assert metric["bootstrap_ratings"] == expected


def test_select_preset_prompt_forwards_named_parser(tmp_path, monkeypatch):
    from judgearena.prompts.parsing import JUDGE_PARSERS

    sentinel = object()
    monkeypatch.setitem(JUDGE_PARSERS, "sentinel", sentinel)
    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("system")
    user_file.write_text(
        "{user_prompt} {completion_A} {completion_B}\n# Your output\nscores"
    )

    prompt = _select_preset_prompt(
        "writing",
        multi_turn=False,
        reference_categories=REFERENCE_CATEGORIES,
        system_file=str(system_file),
        user_file=str(user_file),
        parser="sentinel",
    )

    assert prompt.parse is sentinel
