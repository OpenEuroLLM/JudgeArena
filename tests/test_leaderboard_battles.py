import pandas as pd
import pytest
from judgearena.leaderboard.panel import Panel
from judgearena.leaderboard.battles import sample_pool_battles


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
