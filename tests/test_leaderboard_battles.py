from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from judgearena.leaderboard.battles import judge_pool_battles, sample_pool_battles
from judgearena.leaderboard.panel import Panel


def _pool():
    meta = {"pool_models": ["m1", "m2"], "baseline_model": "m1"}
    comps = pd.DataFrame({
        "model": ["m1", "m1", "m2", "m2"],
        "instruction": ["q1", "q2", "q1", "q2"],
        "lang": ["en", "en", "en", "en"],
        "completion": ["m1q1", "m1q2", "m2q1", "m2q2"],
    })
    return Panel(meta=meta, battles=pd.DataFrame()), comps


def test_sample_pool_battles_shape_and_connectivity():
    panel, comps = _pool()
    new_comps = pd.DataFrame({"model": ["new", "new"], "instruction": ["q1", "q2"],
                              "lang": ["en", "en"], "completion": ["nq1", "nq2"]})
    specs = sample_pool_battles("new", new_comps, panel, comps, n_per_pair=2, seed=0)
    assert set(specs.columns) >= {"instruction", "lang", "opponent", "new_completion", "opp_completion"}
    assert set(specs["opponent"]) == {"m1", "m2"}        # battles vs both pool members
    assert len(specs) == 4                                # 2 opponents x 2 instructions
    row = specs[(specs.opponent == "m2") & (specs.instruction == "q1")].iloc[0]
    assert row["new_completion"] == "nq1" and row["opp_completion"] == "m2q1"


def test_sample_pool_battles_disconnected_raises():
    panel, comps = _pool()
    new_comps = pd.DataFrame({"model": ["new"], "instruction": ["zzz"], "lang": ["en"],
                              "completion": ["x"]})  # no shared instruction with the pool
    with pytest.raises(ValueError):
        sample_pool_battles("new", new_comps, panel, comps, n_per_pair=2, seed=0)


def _make_specs():
    return pd.DataFrame({
        "instruction": ["q1", "q2"],
        "lang": ["en", "fr"],
        "opponent": ["m1", "m2"],
        "new_completion": ["new_ans1", "new_ans2"],
        "opp_completion": ["opp_ans1", "opp_ans2"],
    })


def _fake_judge_cfg():
    cfg = MagicMock()
    cfg.model = "fake-model"
    cfg.swap_mode = "fixed"
    cfg.provide_explanation = False
    return cfg


def _fake_resolved():
    resolved = MagicMock()
    resolved.system_prompt = None
    resolved.user_prompt_template = "{instruction} {completion_a} {completion_b}"
    resolved.preset_name = "fake_preset"
    return resolved


def _make_specs_with_positions(n=200, seed=42):
    """Build specs with randomized challenger_position (~50/50 A/B)."""
    rng = np.random.default_rng(seed)
    positions = rng.choice(["A", "B"], size=n)
    return pd.DataFrame({
        "instruction": [f"q{i}" for i in range(n)],
        "lang": ["en"] * n,
        "opponent": ["m1"] * n,
        "new_completion": [f"new_ans{i}" for i in range(n)],
        "opp_completion": [f"opp_ans{i}" for i in range(n)],
        "challenger_position": positions,
    })


def test_judge_pool_battles_has_completion_columns():
    """judge_pool_battles must return completion_a == new_completion and completion_b == opp_completion."""
    specs = _make_specs()
    # Add challenger_position (all "A" for this schema test)
    specs["challenger_position"] = "A"
    judge_cfg = _fake_judge_cfg()
    prefs = [0.3, 0.7]

    with (
        patch("judgearena.leaderboard.battles.resolve_run_judge_prompt", return_value=_fake_resolved()),
        patch("judgearena.leaderboard.battles.make_model", return_value=MagicMock()),
        patch("judgearena.leaderboard.battles.judge_and_parse_prefs", return_value=(None, None, prefs)),
    ):
        result = judge_pool_battles(specs, "new", judge_cfg=judge_cfg, scorer=None)

    assert "completion_a" in result.columns, "completion_a column missing from judge_pool_battles output"
    assert "completion_b" in result.columns, "completion_b column missing from judge_pool_battles output"
    assert list(result["completion_a"]) == list(specs["new_completion"]), \
        "completion_a should equal specs['new_completion'] (stored frame stays new=A)"
    assert list(result["completion_b"]) == list(specs["opp_completion"]), \
        "completion_b should equal specs['opp_completion'] (stored frame stays new=A)"


def test_sample_pool_battles_has_challenger_position():
    """sample_pool_battles must add a randomized challenger_position column."""
    panel, comps = _pool()
    new_comps = pd.DataFrame({"model": ["new", "new"], "instruction": ["q1", "q2"],
                              "lang": ["en", "en"], "completion": ["nq1", "nq2"]})
    specs = sample_pool_battles("new", new_comps, panel, comps, n_per_pair=2, seed=0)
    assert "challenger_position" in specs.columns, "challenger_position column missing from sample_pool_battles output"
    assert set(specs["challenger_position"]).issubset({"A", "B"}), \
        "challenger_position values must be 'A' or 'B'"


def test_position_bias_debiased():
    """A judge that ALWAYS prefers position A should yield ~50% new-model wins when
    challenger_position is randomized (~50/50 A/B), because the pref is un-flipped
    back to the new=model_a frame."""
    specs = _make_specs_with_positions(n=200, seed=42)
    judge_cfg = _fake_judge_cfg()

    n = len(specs)

    def biased_judge(**kwargs):
        # Always prefer whoever is in position A: P(B wins) = 0.0 for every row.
        return None, None, [0.0] * n

    with (
        patch("judgearena.leaderboard.battles.resolve_run_judge_prompt", return_value=_fake_resolved()),
        patch("judgearena.leaderboard.battles.make_model", return_value=MagicMock()),
        patch("judgearena.leaderboard.battles.judge_and_parse_prefs", side_effect=biased_judge),
    ):
        result = judge_pool_battles(specs, "new", judge_cfg=judge_cfg, scorer=None)

    # judge_pref_hard == 0.0 means new model wins (model_a wins), 1.0 means opponent wins.
    hard = result["judge_pref_hard"].dropna()
    new_wins = (hard == 0.0).sum()
    win_rate = new_wins / len(hard)
    assert 0.40 <= win_rate <= 0.60, (
        f"Expected ~50% new-model wins with randomized positions + biased judge, got {win_rate:.2%}. "
        "Position bias was NOT debiased."
    )


def test_position_bias_without_randomization():
    """Document the pre-fix bug: a position-A-biased judge gives ~100% new-model wins
    when all challenger_position=="A" (new model always shown in judge position A)."""
    specs = _make_specs_with_positions(n=200, seed=42)
    # Override: put new model in position A for every battle (the old bug)
    specs["challenger_position"] = "A"
    judge_cfg = _fake_judge_cfg()

    n = len(specs)

    def biased_judge(**kwargs):
        return None, None, [0.0] * n  # always prefer position A

    with (
        patch("judgearena.leaderboard.battles.resolve_run_judge_prompt", return_value=_fake_resolved()),
        patch("judgearena.leaderboard.battles.make_model", return_value=MagicMock()),
        patch("judgearena.leaderboard.battles.judge_and_parse_prefs", side_effect=biased_judge),
    ):
        result = judge_pool_battles(specs, "new", judge_cfg=judge_cfg, scorer=None)

    hard = result["judge_pref_hard"].dropna()
    new_wins = (hard == 0.0).sum()
    win_rate = new_wins / len(hard)
    assert win_rate >= 0.95, (
        f"Expected ~100% new-model wins when all positions=='A' + biased judge, got {win_rate:.2%}. "
        "This test documents the position-bias bug (all-A = inflated wins)."
    )


def test_judge_pool_battles_output_position_is_canonical_a():
    # Regression: prefs are un-flipped into the new=model_a=A frame, so the stored
    # "position" must be "A" (not the randomized challenger_position). assemble's
    # submission head-to-head reads battles["position"] with pref_to_win_a(judge_pref);
    # a renamed/ randomized column there silently breaks that attribution.
    specs = _make_specs_with_positions(n=20, seed=1)
    n = len(specs)

    with (
        patch("judgearena.leaderboard.battles.resolve_run_judge_prompt", return_value=_fake_resolved()),
        patch("judgearena.leaderboard.battles.make_model", return_value=MagicMock()),
        patch("judgearena.leaderboard.battles.judge_and_parse_prefs",
              side_effect=lambda **kw: (None, None, [0.0] * n)),
    ):
        result = judge_pool_battles(specs, "new", judge_cfg=_fake_judge_cfg(), scorer=None)

    assert "position" in result.columns
    assert set(result["position"]) == {"A"}
    assert "challenger_position" not in result.columns
