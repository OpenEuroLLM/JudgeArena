import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.elo.runner as estimate_elo_ratings
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    fit_bradley_terry,
    winner_to_pref,
)
from judgearena.benchmarks.elo.runner import run_elo
from judgearena.config import RunConfig
from judgearena.evaluate import judge_and_parse_prefs
from judgearena.models import make_model
from judgearena.tasks.registry import get_packaged_task

N_BATTLES = 30
ARENA_MODELS = ["arena_model_alpha", "arena_model_beta", "arena_model_gamma"]


def _make_conversation(content_user: str, content_assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": content_user},
        {"role": "assistant", "content": content_assistant},
    ]


def _arena_df(n_battles: int) -> pd.DataFrame:
    """Synthetic arena DataFrame matching the schema produced by load_arena_dataframe."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_battles):
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


@pytest.fixture
def synthetic_arena_df() -> pd.DataFrame:
    return _arena_df(N_BATTLES)


@pytest.fixture(autouse=True)
def mock_external_deps(monkeypatch, synthetic_arena_df):
    monkeypatch.setattr(
        estimate_elo_ratings, "load_battles", lambda _task: synthetic_arena_df
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


def _default_args(
    *,
    result_folder,
    task="elo-comparia",
    model="Dummy/my model",
    judge_model="Dummy/score A: 0 score B: 10",
    n_instructions=10,
    languages=None,
    swap_mode="fixed",
    strip_thinking_before_judging=False,
    calibrate_temperature=False,
    battle_thinking_token_budget=None,
) -> RunConfig:
    return RunConfig(
        task=task,
        model={"name": model},
        judge={
            "model": judge_model,
            "swap_mode": swap_mode,
            "strip_thinking_before_judging": strip_thinking_before_judging,
            "battle_thinking_token_budget": battle_thinking_token_budget,
        },
        generation={"n_instructions": n_instructions},
        elo={
            "n_bootstraps": 3,
            "languages": languages,
            "calibrate_temperature": calibrate_temperature,
        },
        run={"result_folder": result_folder},
    )


# --- fit_bradley_terry unit tests ---


def _records_with_pref(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["pref"] = df["winner"].map(winner_to_pref)
    return df


def test_bradley_terry_all_ties():
    """All ties → ratings should be equal."""
    records = [{"model_a": "A", "model_b": "B", "winner": "tie"}] * 20
    ratings = fit_bradley_terry(_records_with_pref(records))
    assert abs(ratings["A"] - ratings["B"]) < 1.0


def test_bradley_terry_baseline():
    """Baseline model is anchored at baseline_rating."""
    records = [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 10
    ratings = fit_bradley_terry(
        _records_with_pref(records), baseline_model="B", baseline_rating=1000
    )
    assert ratings["B"] == pytest.approx(1000.0)
    assert ratings["A"] > 1000.0


# --- run_elo() integration tests ---


def run_elo_with_task(cfg: RunConfig) -> dict:
    return run_elo(cfg, get_packaged_task(cfg.task))


def test_run_elo_winrate_depends_on_judge(tmp_path):
    """A judge biased toward one position should yield different winrates depending on direction."""
    # With seed=0 and n=10 our model is always placed in position B, so:
    # judge favouring B → all wins; judge favouring A → all losses
    result_wins = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), judge_model="Dummy/score A: 0 score B: 10"
        )
    )
    result_loses = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), judge_model="Dummy/score A: 10 score B: 0"
        )
    )
    assert result_wins["winrate"] > result_loses["winrate"]


def test_run_elo_language_filter_reduces_battles(tmp_path):
    """Filtering to a single language should use fewer battles than no filter."""
    result_all = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), n_instructions=None)
    )
    result_en = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), n_instructions=None, languages=["en"]
        )
    )
    total_all = (
        result_all["num_wins"] + result_all["num_losses"] + result_all["num_ties"]
    )
    total_en = result_en["num_wins"] + result_en["num_losses"] + result_en["num_ties"]
    assert total_en < total_all


def test_run_elo_limits_battles_and_reports_bootstrap_ratings(tmp_path):
    result = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), n_instructions=5)
    )
    assert (
        sum(
            result[key] for key in ("num_wins", "num_losses", "num_ties", "num_missing")
        )
        == 5
    )
    assert 0.0 <= result["winrate"] <= 1.0
    assert len(result["bootstrap_ratings"]) == 3
    assert all(
        result["model_name"] in ratings for ratings in result["bootstrap_ratings"]
    )


def test_run_elo_forwards_judge_settings(monkeypatch, tmp_path):
    # Regressions: swap/strip flags and the preset parser must reach judging.
    from judgearena.prompts.parsing import JUDGE_PARSERS

    captured = {}
    real_judge = estimate_elo_ratings.judge_and_parse_prefs

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_judge(*args, **kwargs)

    monkeypatch.setattr(estimate_elo_ratings, "judge_and_parse_prefs", spy)
    run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            swap_mode="both",
            strip_thinking_before_judging=True,
        )
    )
    assert captured["swap_mode"] == "both"
    assert captured["strip_thinking_before_judging"] is True
    assert captured["parse"] is JUDGE_PARSERS["score"]


def test_run_elo_thinking_budget_capped_by_max_out_tokens(monkeypatch, tmp_path):
    captured = {}

    def spy_generate(instructions, model, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "completion": ["c"] * len(instructions),
                "instruction_index": range(len(instructions)),
            }
        )

    monkeypatch.setattr(estimate_elo_ratings, "generate_instructions", spy_generate)
    cfg = _default_args(
        result_folder=str(tmp_path),
        model="VLLM/Qwen/Qwen3.5-9B",
        battle_thinking_token_budget=10**9,
    )
    run_elo_with_task(cfg)
    assert captured["thinking_token_budget"] == cfg.model.max_out_tokens


def test_judge_and_parse_prefs_retains_structured_result():
    judge = make_model("Dummy/Score_A: 6\nScore_B: 8")

    annotations, _, prefs = judge_and_parse_prefs(
        judge_chat_model=judge,
        instructions=["Q"],
        completions_A=["A"],
        completions_B=["B"],
    )

    assert prefs.tolist() == pytest.approx([0.6456563062257954])
    assert annotations[0].parsed is not None
    assert annotations[0].parsed.scores == {"A": 6.0, "B": 8.0}


def test_judge_and_parse_prefs_none_prefs_swap_mode_both():
    """swap_mode='both' must not raise when judge output is unparseable (None prefs).

    Regression test: previously '1 - prefs_reversed' raised TypeError when
    prefs_reversed contained None values from an unparseable judge completion.
    """
    judge = make_model("Dummy/no scores here at all")
    instructions = ["Q"]
    completions_A = ["A"]
    completions_B = ["B"]

    _, _, prefs = judge_and_parse_prefs(
        judge_chat_model=judge,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        swap_mode="both",
    )
    # All prefs should be NaN (unparseable → nan), not raise
    assert prefs.isna().tolist() == [True, True]


def test_arena_anchor_battles_filters_and_preserves_index():
    # Anchors are rebuilt on recompute, so this primitive must drop under-
    # represented models (< 500 battles), keep provenance, and preserve the
    # arena row labels (calibration looks up conversations via df_arena_all.loc[i]).
    n = 500
    df_all = pd.DataFrame(
        {
            "model_a": ["x"] * n + ["rare"],
            "model_b": ["y"] * n + ["x"],
            "winner": ["model_a", "model_b"] * (n // 2) + ["model_a"],
            "conversation_a": [["q"]] * (n + 1),  # extra column must be ignored
        },
        index=range(1000, 1000 + n + 1),
    )
    out = arena_anchor_battles(df_all)

    # x, y have >= 500 battles -> kept; 'rare' (1 battle) -> its row dropped
    assert set(out["model_a"]) | set(out["model_b"]) == {"x", "y"}
    assert list(out.index) == list(range(1000, 1000 + n))  # labels preserved, rare gone
    assert (out["source"] == "human").all()
    assert out.loc[1000, "pref"] == 0.0 and out.loc[1001, "pref"] == 1.0


def test_elo_language_variant_resolves_and_filters(tmp_path):
    variant = get_packaged_task("elo-lmarena-140k-en")
    assert variant is not None
    assert variant.selection is not None
    assert variant.selection.values == ("en",)

    result_en = run_elo(
        _default_args(
            result_folder=str(tmp_path), task="elo-lmarena-140k-en", n_instructions=None
        ),
        variant,
    )
    result_all = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), task="elo-lmarena-140k", n_instructions=None
        )
    )
    total_en = result_en["num_wins"] + result_en["num_losses"] + result_en["num_ties"]
    total_all = (
        result_all["num_wins"] + result_all["num_losses"] + result_all["num_ties"]
    )
    assert 0 < total_en < total_all


def test_run_elo_temperature_calibration_builds_judge(monkeypatch, tmp_path):
    """Regression: the calibration path constructs its own judge model and once
    crashed on a duplicate max_tokens kwarg; nothing else exercises it. The
    MLE fit itself is mocked."""
    captured = {}

    def fake_calibrate(delta_s, y):
        captured["n_pairs"] = len(delta_s)
        return 0.42

    monkeypatch.setattr(estimate_elo_ratings, "calibrate_temperature", fake_calibrate)
    # Anchor battles require models with >= 500 appearances; the default
    # 30-battle fixture leaves the calibration pool empty.
    monkeypatch.setattr(
        estimate_elo_ratings, "load_battles", lambda _task: _arena_df(900)
    )

    result = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), calibrate_temperature=True)
    )

    assert captured["n_pairs"] >= 10
    assert 0.0 <= result["winrate"] <= 1.0


def test_extract_instruction_text_tolerates_moderated_turns():
    from judgearena.arenas_utils import _extract_instruction_text

    assert _extract_instruction_text({"content": None}) == ""
    assert _extract_instruction_text({"content": "plain"}) == "plain"
    assert (
        _extract_instruction_text(
            {"content": [{"type": "text", "text": None}, {"type": "image"}, None]}
        )
        == ""
    )
