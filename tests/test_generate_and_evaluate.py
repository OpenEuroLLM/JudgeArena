import pandas as pd
import pytest

import openjury.generate_and_evaluate as generate_and_evaluate
import openjury.mt_bench.mt_bench_utils as mt_bench_pipeline
from openjury.eval_utils import JudgeAnnotationResult
from openjury.generate_and_evaluate import (
    main as main_generate_and_eval,
    CliArgs,
)


@pytest.fixture(autouse=True)
def mock_external_data_and_cache(monkeypatch):
    single_turn_instructions = pd.DataFrame(
        {
            "instruction": [f"Synthetic instruction {i}" for i in range(20)],
        },
        index=pd.Index(range(20), name="instruction_index"),
    )

    # Mix of general and NEED_REF_CATS categories to exercise both code paths.
    categories = ["writing", "math", "reasoning", "coding", "roleplay",
                   "writing", "math", "reasoning", "coding", "roleplay",
                   "writing", "math", "reasoning", "coding", "roleplay",
                   "writing", "math", "reasoning", "coding", "roleplay"]
    ref_turn_1 = [
        f"Reference answer turn 1 for q{i}" if cat in ("math", "reasoning", "coding") else None
        for i, cat in enumerate(categories)
    ]
    ref_turn_2 = [
        f"Reference answer turn 2 for q{i}" if cat in ("math", "reasoning", "coding") else None
        for i, cat in enumerate(categories)
    ]
    mt_bench_questions = pd.DataFrame(
        {
            "category": categories,
            "turn_1": [f"Synthetic MT-Bench turn 1 question {i}" for i in range(20)],
            "turn_2": [f"Synthetic MT-Bench turn 2 follow-up {i}" for i in range(20)],
            "reference_turn_1": ref_turn_1,
            "reference_turn_2": ref_turn_2,
        },
        index=pd.Index(range(20), name="instruction_index"),
    )
    mt_bench_questions["instruction"] = mt_bench_questions["turn_1"]

    def _load_instructions(dataset: str, n_instructions: int | None = None) -> pd.DataFrame:
        df = mt_bench_questions if dataset == "mt-bench" else single_turn_instructions
        return df.head(n_instructions) if n_instructions is not None else df

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_instructions",
        _load_instructions,
    )
    monkeypatch.setattr(
        mt_bench_pipeline,
        "load_instructions",
        _load_instructions,
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_contexts",
        lambda dataset: single_turn_instructions.loc[:, "instruction"],
    )

    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: None,
    )

    def _run_without_cache(fun, **_kwargs):
        return fun()

    monkeypatch.setattr(
        generate_and_evaluate, "cache_function_dataframe", _run_without_cache
    )
    monkeypatch.setattr(
        mt_bench_pipeline, "cache_function_dataframe", _run_without_cache
    )


@pytest.mark.parametrize(
    "dataset", ["alpaca-eval", "fluency-french", "m-arena-hard-EU"]
)
def test_generate_and_evaluate_context_completion(dataset: str, tmp_path):
    prefs = main_generate_and_eval(
        CliArgs(
            dataset=dataset,
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            result_folder=str(tmp_path),
            # default for swap_mode is "fixed"
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref >= 0.9


def test_generate_and_evaluate_correct_order_bias(tmp_path):
    """Test the correction for model order bias.
    
    In this test, a judge that is totally biased towards model B should be corrected to be neutral.
    Since the judge favors model B regardless of the order and the completions, the average
    preference should be 0.5.
    """
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            swap_mode="both",
            result_folder=str(tmp_path),
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref == pytest.approx(0.5)


def test_main_non_mt_bench_reuses_make_judge_annotation(monkeypatch, tmp_path):
    captured = {"calls": 0, "kwargs": None}

    def _make_judge_annotation_stub(**kwargs):
        captured["calls"] += 1
        captured["kwargs"] = kwargs
        return JudgeAnnotationResult(
            annotations=[{"judge_completion": "score A: 0 score B: 10"}],
            annotations_reversed=[],
            metadata_for_annotations=[{"instruction_index": 0}],
            metadata_for_reversed_annotations=[],
            preferences=pd.Series([1.0]),
            combined_metadata=[{"instruction_index": 0}],
        )

    monkeypatch.setattr(
        generate_and_evaluate,
        "_make_judge_annotation",
        _make_judge_annotation_stub,
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=1,
            result_folder=str(tmp_path),
        )
    )

    assert captured["calls"] == 1
    assert captured["kwargs"]["swap_mode"] == "fixed"
    assert captured["kwargs"]["metadata"] == [{"instruction_index": 0}]
    assert prefs.tolist() == [1.0]


def test_format_mt_bench_turn_2_uses_conversation_blocks():
    questions = pd.DataFrame(
        {
            "category": ["math", "writing"],
            "turn_1": ["Math question turn 1", "Writing question turn 1"],
            "turn_2": ["Math question turn 2", "Writing question turn 2"],
            "reference_turn_1": ["Math reference turn 1", None],
            "reference_turn_2": ["Math reference turn 2", None],
        },
        index=pd.Index([0, 1], name="instruction_index"),
    )
    completions_a = pd.DataFrame(
        {
            "completion_turn_1": ["A1 math", "A1 writing"],
            "completion_turn_2": ["A2 math", "A2 writing"],
        },
        index=pd.Index([0, 1], name="instruction_index"),
    )
    completions_b = pd.DataFrame(
        {
            "completion_turn_1": ["B1 math", "B1 writing"],
            "completion_turn_2": ["B2 math", "B2 writing"],
        },
        index=pd.Index([0, 1], name="instruction_index"),
    )

    turn_1_inputs, turn_2_inputs = generate_and_evaluate.format_mt_bench_for_evaluation(
        questions=questions,
        completions_A=completions_a,
        completions_B=completions_b,
        turns_mode="both",
        truncate_input_chars=8192,
    )
    (
        instructions_turn_1,
        _completions_a_turn_1,
        _completions_b_turn_1,
        _metadata_turn_1,
    ) = turn_1_inputs
    (
        instructions_turn_2,
        completions_a_turn_2,
        completions_b_turn_2,
        _metadata_turn_2,
    ) = turn_2_inputs

    assert "Please focus on which assistant provides a better answer to the second user question." in instructions_turn_2[0]
    assert "<|The Start of Reference Answer|>" in instructions_turn_2[0]
    assert "Math reference turn 1" in instructions_turn_2[0]
    assert "Math reference turn 2" in instructions_turn_2[0]
    assert "<|The Start of Reference Answer|>" not in instructions_turn_2[1]

    assert "### User:\nMath question turn 1" in completions_a_turn_2[0]
    assert "### Assistant:\nA1 math" in completions_a_turn_2[0]
    assert "### User:\nMath question turn 2" in completions_a_turn_2[0]
    assert "### Assistant:\nA2 math" in completions_a_turn_2[0]

    assert "### User:\nMath question turn 1" in completions_b_turn_2[0]
    assert "### Assistant:\nB1 math" in completions_b_turn_2[0]
    assert "### User:\nMath question turn 2" in completions_b_turn_2[0]
    assert "### Assistant:\nB2 math" in completions_b_turn_2[0]

    assert instructions_turn_1[1] == "Writing question turn 1"
    assert "[MT-Bench | Turn 1]" in instructions_turn_1[0]


def test_mt_bench_pairwise(tmp_path):
    """Test MT-Bench pipeline through score-based parsing."""
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="mt-bench",
            model_A="Dummy/answer for turn 1 and turn 2",
            model_B="Dummy/another answer",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=5,
            result_folder=str(tmp_path),
        )
    )

    assert all(p < 0.5 for p in prefs)
    assert len(prefs) == 10  # two turns per question


def test_mt_bench_swap_mode(tmp_path):
    """Test that MT-Bench swap mode doubles the annotations and corrects bias."""
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="mt-bench",
            model_A="Dummy/answer A",
            model_B="Dummy/answer B",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=3,
            swap_mode="both",
            result_folder=str(tmp_path),
        )
    )

    assert len(prefs) == 12  # (3 questions * 2 turns) * 2 swap directions
    assert float(sum(prefs) / len(prefs)) == pytest.approx(0.5)


def test_mt_bench_single_turn_only(tmp_path):
    """Test MT-Bench single-turn-only evaluation (--mt_bench_turns single)."""
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="mt-bench",
            model_A="Dummy/answer A",
            model_B="Dummy/answer B",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=5,
            mt_bench_turns="single",
            result_folder=str(tmp_path),
        )
    )

    assert all(p < 0.5 for p in prefs)
    assert len(prefs) == 5  # one annotation per question, turn 1 only


def test_mt_bench_multi_turn_only(tmp_path):
    """Test MT-Bench multi-turn-only evaluation (--mt_bench_turns multi)."""
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="mt-bench",
            model_A="Dummy/answer A",
            model_B="Dummy/answer B",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            mt_bench_turns="multi",
            result_folder=str(tmp_path),
        )
    )

    assert all(p > 0.5 for p in prefs)
    assert len(prefs) == 5  # one annotation per question, turn 2 only


def test_mt_bench_fastchat_fixed_verdicts(tmp_path):
    """FastChat-compatible MT-Bench judging uses [[A]]/[[B]]/[[C]] parsing."""
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="mt-bench",
            model_A="Dummy/answer A",
            model_B="Dummy/answer B",
            judge_model="Dummy/[[A]]",
            n_instructions=5,
            mt_bench_compatibility="fastchat",
            result_folder=str(tmp_path),
        )
    )

    assert len(prefs) == 10  # two turns per question
    assert all(p < 0.5 for p in prefs)


def test_mt_bench_fastchat_conservative_swap_mode(tmp_path):
    """FastChat-compatible swap_mode='both' is conservative (tie if inconsistent)."""
    prefs = main_generate_and_eval(
        CliArgs(
            dataset="mt-bench",
            model_A="Dummy/answer A",
            model_B="Dummy/answer B",
            judge_model="Dummy/[[A]]",  # position-A biased judge
            n_instructions=3,
            swap_mode="both",
            mt_bench_compatibility="fastchat",
            result_folder=str(tmp_path),
        )
    )

    # Conservative swap runs both orders, but returns one resolved verdict per match.
    assert len(prefs) == 6  # 3 questions * 2 turns
    assert all(p == pytest.approx(0.5) for p in prefs)