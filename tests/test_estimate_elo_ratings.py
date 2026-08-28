import math
from types import SimpleNamespace

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
from judgearena.evaluate import JudgeAnnotation, judge_and_parse_prefs
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
        estimate_elo_ratings,
        "load_battles",
        lambda _task: synthetic_arena_df,
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


def _default_args(*, result_folder: str, **kwargs) -> RunConfig:
    task = kwargs.pop("task", "elo-comparia")
    arena = kwargs.pop("arena", None)
    model = kwargs.pop("model", "Dummy/my model")
    judge_model = kwargs.pop("judge_model", "Dummy/score A: 0 score B: 10")
    n_instructions = kwargs.pop("n_instructions", 10)
    n_bootstraps = kwargs.pop("n_bootstraps", 3)
    languages = kwargs.pop("languages", None)
    swap_mode = kwargs.pop("swap_mode", "fixed")
    strip_thinking_before_judging = kwargs.pop("strip_thinking_before_judging", False)
    prompt_preset = kwargs.pop("prompt_preset", None)
    calibrate_temperature = kwargs.pop("calibrate_temperature", False)
    battle_thinking_token_budget = kwargs.pop("battle_thinking_token_budget", None)
    assert not kwargs, f"unexpected kwargs: {kwargs}"
    judge: dict[str, object] = {
        "model": judge_model,
        "swap_mode": swap_mode,
        "strip_thinking_before_judging": strip_thinking_before_judging,
    }
    if battle_thinking_token_budget is not None:
        judge["battle_thinking_token_budget"] = battle_thinking_token_budget
    if prompt_preset is not None:
        judge["prompt_preset"] = prompt_preset
    return RunConfig(
        task=task,
        model={"name": model},
        judge=judge,
        generation={"n_instructions": n_instructions},
        elo={
            "arena": arena,
            "n_bootstraps": n_bootstraps,
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


def test_bradley_terry_clear_winner():
    """Model A always beats B → A gets a higher ELO."""
    records = [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 10 + [
        {"model_a": "B", "model_b": "A", "winner": "model_b"}
    ] * 10
    ratings = fit_bradley_terry(_records_with_pref(records))
    assert ratings["A"] > ratings["B"]


def test_bradley_terry_all_ties():
    """All ties → ratings should be equal."""
    records = [{"model_a": "A", "model_b": "B", "winner": "tie"}] * 20
    ratings = fit_bradley_terry(_records_with_pref(records))
    assert abs(ratings["A"] - ratings["B"]) < 1.0


def test_bradley_terry_baseline():
    """Baseline model is anchored at baseline_rating."""
    records = [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 10
    ratings = fit_bradley_terry(
        _records_with_pref(records),
        baseline_model="B",
        baseline_rating=1000,
    )
    assert ratings["B"] == pytest.approx(1000.0)
    assert ratings["A"] > 1000.0


def test_bradley_terry_soft_matches_hard():
    """Soft prefs ∈ {0, 0.5, 1} must give the same fit as hard winner labels."""
    records = (
        [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 7
        + [{"model_a": "A", "model_b": "B", "winner": "model_b"}] * 3
        + [{"model_a": "A", "model_b": "B", "winner": "tie"}] * 2
    )
    df = _records_with_pref(records)
    hard = fit_bradley_terry(df, pref_col="pref")
    # Passing the same column twice (continuous == quantised here) must match.
    df["pref_soft"] = df["pref"].astype(float)
    soft = fit_bradley_terry(df, pref_col="pref_soft")
    assert hard["A"] == pytest.approx(soft["A"], abs=1e-3)
    assert hard["B"] == pytest.approx(soft["B"], abs=1e-3)


# --- run_elo() integration tests ---


def run_elo_with_task(cfg: RunConfig) -> dict:
    return run_elo(cfg, get_packaged_task(cfg.task))


def test_run_elo_returns_summary(tmp_path):
    result = run_elo_with_task(_default_args(result_folder=str(tmp_path)))
    assert set(result.keys()) >= {
        "num_wins",
        "num_losses",
        "num_ties",
        "winrate",
        "bootstrap_ratings",
        "model_name",
    }


def test_run_elo_winrate_in_valid_range(tmp_path):
    result = run_elo_with_task(_default_args(result_folder=str(tmp_path)))
    assert 0.0 <= result["winrate"] <= 1.0


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


def test_run_elo_model_in_bootstrap_ratings(tmp_path):
    """Our model should appear in the bootstrap ELO leaderboard."""
    result = run_elo_with_task(_default_args(result_folder=str(tmp_path)))
    model_name = result["model_name"]
    assert all(model_name in r for r in result["bootstrap_ratings"])


def test_run_elo_n_instructions_limits_battles(tmp_path):
    """n_instructions caps the number of judged battles."""
    result_5 = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), n_instructions=5)
    )
    result_10 = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), n_instructions=10)
    )
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


def test_run_elo_swap_mode_forwarded_to_judge(monkeypatch, tmp_path):
    """swap_mode from the run config must be forwarded to judge_and_parse_prefs.

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
    run_elo_with_task(_default_args(result_folder=str(tmp_path), swap_mode="both"))
    assert captured.get("swap_mode") == "both"


def _spy_judge_capturing(captured):
    def spy_judge(
        judge_chat_model,
        instructions,
        completions_A,
        completions_B,
        swap_mode="fixed",
        strip_thinking_before_judging=False,
        **kwargs,
    ):
        captured["strip_thinking_before_judging"] = strip_thinking_before_judging
        n = len(instructions)
        dummy = JudgeAnnotation(
            judge_completion="score A: 0 score B: 10",
            instruction="",
            completion_A="",
            completion_B="",
        )
        return [dummy] * n, None, pd.Series([1.0] * n)

    return spy_judge


def test_run_elo_strip_thinking_forwarded_to_judge(monkeypatch, tmp_path):
    """strip_thinking_before_judging from the run config must reach the judge.

    Regression test: the Elo entrypoint accepted the flag but never forwarded it
    to judge_and_parse_prefs, so reasoning traces were judged verbatim.
    """
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "judge_and_parse_prefs", _spy_judge_capturing(captured)
    )
    run_elo_with_task(
        _default_args(result_folder=str(tmp_path), strip_thinking_before_judging=True)
    )
    assert captured.get("strip_thinking_before_judging") is True


def test_run_elo_strip_thinking_defaults_off(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "judge_and_parse_prefs", _spy_judge_capturing(captured)
    )
    run_elo_with_task(_default_args(result_folder=str(tmp_path)))
    assert captured.get("strip_thinking_before_judging") is False


def _spy_generate_capturing(captured):
    def spy_generate(instructions, model, **kwargs):
        captured["gen_kwargs"] = kwargs
        return pd.DataFrame(
            {
                "completion": [f"c{i}" for i in range(len(instructions))],
                "instruction_index": range(len(instructions)),
            }
        )

    return spy_generate


def test_run_elo_thinking_budget_injected_for_thinking_model(monkeypatch, tmp_path):
    """battle_thinking_token_budget must reach generation for VLLM thinking models."""
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "generate_instructions", _spy_generate_capturing(captured)
    )
    run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            model="VLLM/Qwen/Qwen3.5-9B",
            battle_thinking_token_budget=128,
        )
    )
    assert captured["gen_kwargs"].get("thinking_token_budget") == 128


def test_run_elo_thinking_budget_capped_by_max_out_tokens(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "generate_instructions", _spy_generate_capturing(captured)
    )
    cfg = _default_args(
        result_folder=str(tmp_path),
        model="VLLM/Qwen/Qwen3.5-9B",
        battle_thinking_token_budget=10**9,
    )
    run_elo_with_task(cfg)
    assert (
        captured["gen_kwargs"].get("thinking_token_budget") == cfg.model.max_out_tokens
    )


def test_run_elo_thinking_budget_absent_for_nonthinking_model(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "generate_instructions", _spy_generate_capturing(captured)
    )
    run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            model="Dummy/my model",
            battle_thinking_token_budget=128,
        )
    )
    assert "thinking_token_budget" not in captured["gen_kwargs"]


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


def test_run_elo_forwards_resolved_parser(tmp_path, monkeypatch):
    from judgearena.prompts.parsing import JUDGE_PARSERS

    captured = {}
    real = estimate_elo_ratings.judge_and_parse_prefs

    def spy(*args, **kwargs):
        captured["parse"] = kwargs.get("parse")
        return real(*args, **kwargs)

    monkeypatch.setattr(estimate_elo_ratings, "judge_and_parse_prefs", spy)
    run_elo_with_task(_default_args(result_folder=str(tmp_path)))

    # The default preset's registered parser instance, not a fresh fallback.
    assert captured["parse"] is JUDGE_PARSERS["score"]


def test_run_elo_keeps_non_pairscore_soft_preferences(tmp_path, monkeypatch):
    def fake_judge_and_parse_prefs(**kwargs):
        annotations = [
            JudgeAnnotation(
                instruction=instruction,
                completion_A=completion_a,
                completion_B=completion_b,
                judge_completion="M",
                judge_input="prompt",
            )
            for instruction, completion_a, completion_b in zip(
                kwargs["instructions"],
                kwargs["completions_A"],
                kwargs["completions_B"],
                strict=True,
            )
        ]
        return annotations, None, pd.Series([0.75] * len(annotations))

    monkeypatch.setattr(
        estimate_elo_ratings,
        "judge_and_parse_prefs",
        fake_judge_and_parse_prefs,
    )
    result = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            prompt_preset="alpaca-eval",
        )
    )

    assert result["elo_n_bootstraps"] > 0
    assert result["model_name"] in result["mean_ratings"]


def test_judge_cache_identity_covers_resolved_protocol_settings():
    def prompt(*, parser=None, system="system", user="{completion_A} {completion_B}"):
        return SimpleNamespace(
            preset_name="default",
            parser=parser or estimate_elo_ratings.PairScore(temperature=0.3),
            system_prompt=system,
            user_prompt_template=user,
        )

    base = {
        "judge_model": "OpenRouter/judge-a",
        "resolved_prompt": prompt(),
        "swap_mode": "fixed",
        "judge_model_kwargs": {"temperature": 0.0, "top_logprobs": 5},
        "strip_thinking_before_judging": False,
        "truncate_input_chars": 4096,
        "run_seed": 0,
    }
    variants = [
        {**base, "judge_model": "OpenRouter/judge-b"},
        {**base, "resolved_prompt": prompt(system="different system")},
        {**base, "resolved_prompt": prompt(user="different {completion_A}")},
        {
            **base,
            "resolved_prompt": prompt(
                parser=estimate_elo_ratings.PairScore(temperature=0.5)
            ),
        },
        {**base, "swap_mode": "both"},
        {**base, "judge_model_kwargs": {"temperature": 0.2, "top_logprobs": 5}},
        {**base, "judge_model_kwargs": {"temperature": 0.0, "top_logprobs": 10}},
        {**base, "strip_thinking_before_judging": True},
        {**base, "truncate_input_chars": 2048},
        {**base, "run_seed": 1},
    ]

    base_hash = estimate_elo_ratings._judge_cache_identity_hash(**base)

    assert all(
        estimate_elo_ratings._judge_cache_identity_hash(**variant) != base_hash
        for variant in variants
    )
    assert (
        estimate_elo_ratings._judge_cache_identity_hash(
            **{
                **base,
                "judge_model_kwargs": {"top_logprobs": 5, "temperature": 0.0},
            }
        )
        == base_hash
    )
