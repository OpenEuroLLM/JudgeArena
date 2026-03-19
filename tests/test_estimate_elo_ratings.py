import math

import numpy as np
import pandas as pd
import pytest

import openjury.estimate_elo_ratings as estimate_elo_ratings
from openjury.estimate_elo_ratings import CliEloArgs, compute_bradley_terry, main
from openjury.evaluate import JudgeAnnotation, judge_and_parse_prefs
from openjury.utils import make_model

N_BATTLES = 30
ARENA_MODELS = ["arena_model_alpha", "arena_model_beta", "arena_model_gamma"]


def _make_conversation(content_user: str, content_assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": content_user},
        {"role": "assistant", "content": content_assistant},
    ]


@pytest.fixture
def synthetic_arena_df() -> pd.DataFrame:
    """Synthetic arena DataFrame matching the schema produced by load_arena_dataframe."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(N_BATTLES):
        ma, mb = rng.choice(ARENA_MODELS, size=2, replace=False)
        winner = rng.choice(["model_a", "model_b", "tie"])
        lang = rng.choice(["en", "fr"])
        rows.append(
            {
                "question_id": f"q{i}",
                "tstamp": 1700000000 + i,
                "model_a": ma,
                "model_b": mb,
                "winner": winner,
                "conversation_a": _make_conversation(
                    f"Instruction {i}", f"Response A {i}"
                ),
                "conversation_b": _make_conversation(
                    f"Instruction {i}", f"Response B {i}"
                ),
                "benchmark": "TestArena",
                "lang": lang,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def mock_external_deps(monkeypatch, synthetic_arena_df):
    monkeypatch.setattr(
        estimate_elo_ratings,
        "load_arena_dataframe",
        lambda arena: synthetic_arena_df,
    )

    def mock_generate(instructions, model, **kwargs):
        return pd.DataFrame(
            {
                "completion": [
                    f"Synthetic completion {i}" for i in range(len(instructions))
                ],
                "instruction_index": range(len(instructions)),
            }
        )

    monkeypatch.setattr(estimate_elo_ratings, "generate_instructions", mock_generate)

    def _run_without_cache(fun, **_kwargs):
        return fun()

    monkeypatch.setattr(
        estimate_elo_ratings, "cache_function_dataframe", _run_without_cache
    )


def _default_args(**kwargs) -> CliEloArgs:
    defaults = dict(
        arena="ComparIA",
        model="Dummy/my model",
        judge_model="Dummy/score A: 0 score B: 10",
        n_instructions=10,
        n_bootstraps=3,
    )
    defaults.update(kwargs)
    return CliEloArgs(**defaults)


# --- compute_bradley_terry unit tests ---


def test_bradley_terry_clear_winner():
    """Model A always beats B → A gets a higher ELO."""
    records = [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 10 + [
        {"model_a": "B", "model_b": "A", "winner": "model_b"}
    ] * 10
    ratings = compute_bradley_terry(pd.DataFrame(records), winner_col="winner")
    assert ratings["A"] > ratings["B"]


def test_bradley_terry_all_ties():
    """All ties → ratings should be equal."""
    records = [{"model_a": "A", "model_b": "B", "winner": "tie"}] * 20
    ratings = compute_bradley_terry(pd.DataFrame(records), winner_col="winner")
    assert abs(ratings["A"] - ratings["B"]) < 1.0


def test_bradley_terry_baseline():
    """Baseline model is anchored at baseline_rating."""
    records = [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 10
    ratings = compute_bradley_terry(
        pd.DataFrame(records),
        winner_col="winner",
        baseline_model="B",
        baseline_rating=1000,
    )
    assert ratings["B"] == pytest.approx(1000.0)
    assert ratings["A"] > 1000.0


# --- main() integration tests ---


def test_main_returns_summary():
    result = main(_default_args())
    assert set(result.keys()) >= {
        "num_wins",
        "num_losses",
        "num_ties",
        "winrate",
        "bootstrap_ratings",
        "model_name",
    }


def test_main_winrate_in_valid_range():
    result = main(_default_args())
    assert 0.0 <= result["winrate"] <= 1.0


def test_main_winrate_depends_on_judge():
    """A judge biased toward one position should yield different winrates depending on direction."""
    # With seed=0 and n=10 our model is always placed in position B, so:
    # judge favouring B → all wins; judge favouring A → all losses
    result_wins = main(_default_args(judge_model="Dummy/score A: 0 score B: 10"))
    result_loses = main(_default_args(judge_model="Dummy/score A: 10 score B: 0"))
    assert result_wins["winrate"] > result_loses["winrate"]


def test_main_language_filter_reduces_battles():
    """Filtering to a single language should use fewer battles than no filter."""
    result_all = main(_default_args(n_instructions=None))
    result_en = main(_default_args(n_instructions=None, languages=["en"]))
    total_all = (
        result_all["num_wins"] + result_all["num_losses"] + result_all["num_ties"]
    )
    total_en = result_en["num_wins"] + result_en["num_losses"] + result_en["num_ties"]
    assert total_en < total_all


def test_main_model_in_bootstrap_ratings():
    """Our model should appear in the bootstrap ELO leaderboard."""
    result = main(_default_args())
    model_name = result["model_name"]
    assert all(model_name in r for r in result["bootstrap_ratings"])


def test_main_n_instructions_limits_battles():
    """n_instructions caps the number of judged battles."""
    result_5 = main(_default_args(n_instructions=5))
    result_10 = main(_default_args(n_instructions=10))
    total_5 = (
        result_5["num_wins"]
        + result_5["num_losses"]
        + result_5["num_ties"]
        + result_5["num_missing"]
    )
    total_10 = (
        result_10["num_wins"]
        + result_10["num_losses"]
        + result_10["num_ties"]
        + result_10["num_missing"]
    )
    assert total_5 == 5
    assert total_10 == 10


def test_main_swap_mode_forwarded_to_judge(monkeypatch):
    """swap_mode from CliEloArgs must be forwarded to judge_and_parse_prefs.

    Regression test: previously run_judge() called judge_and_parse_prefs without
    swap_mode, so --swap_mode both was silently ignored.
    """
    captured = {}

    def spy_judge(
        judge_chat_model,
        instructions,
        completions_A,
        completions_B,
        swap_mode="fixed",
        **kwargs,
    ):
        captured["swap_mode"] = swap_mode
        n = len(instructions)
        dummy = JudgeAnnotation(
            judge_completion="score A: 0 score B: 10",
            instruction="",
            completion_A="",
            completion_B="",
        )
        return [dummy] * n, None, pd.Series([1.0] * n)

    monkeypatch.setattr(estimate_elo_ratings, "judge_and_parse_prefs", spy_judge)
    main(_default_args(swap_mode="both"))
    assert captured.get("swap_mode") == "both"


def test_judge_and_parse_prefs_none_prefs_swap_mode_both():
    """swap_mode='both' must not raise when judge output is unparseable (None prefs).

    Regression test: previously '1 - prefs_reversed' raised TypeError when
    prefs_reversed contained None values from an unparseable judge completion.
    """
    judge = make_model("Dummy/no scores here at all")
    instructions = ["Q1", "Q2", "Q3"]
    completions_A = ["A1", "A2", "A3"]
    completions_B = ["B1", "B2", "B3"]

    _, _, prefs = judge_and_parse_prefs(
        judge_chat_model=judge,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        swap_mode="both",
    )
    # All prefs should be NaN (unparseable → nan), not raise
    assert all(math.isnan(p) for p in prefs)
