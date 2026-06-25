"""Parquet round-trip + leaderboard summary tests for judgearena/battles.py."""

import numpy as np
import pandas as pd

from judgearena.battles import (
    EloReport,
    read_battles,
    summarize_bootstrap,
    write_battles,
)


def test_battles_parquet_round_trip(tmp_path):
    # Mirrors the persisted frame: an llm-judge row (judge_model set) and a
    # human row (judge_model null, pref NaN for a tie). pref_hard is a derived
    # column that the typed Battle view should drop.
    df = pd.DataFrame(
        {
            "model_a": ["m", "gpt"],
            "model_b": ["gpt", "claude"],
            "winner": ["model_a", "tie"],
            "pref": [0.2, np.nan],
            "pref_hard": [0.0, 0.5],
            "source": ["llm-judge", "human"],
            "judge_model": ["judgeX", None],
        }
    )
    path = tmp_path / "battles.parquet"
    write_battles(path, df)

    battles = read_battles(path)
    assert [b.source for b in battles] == ["llm-judge", "human"]
    assert battles[0].pref == 0.2 and battles[0].judge_model == "judgeX"
    assert battles[1].judge_model is None  # human-row null preserved
    assert not hasattr(battles[0], "pref_hard")  # derived column not in typed view


def test_summarize_bootstrap_sorts_and_bounds(tmp_path):
    boot = [{"a": 1000.0, "b": 1100.0}, {"a": 1010.0, "b": 1090.0}]
    ratings = summarize_bootstrap(boot, {"a": 2, "b": 2}, model_under_test="a")

    assert [r.model for r in ratings] == ["b", "a"]  # sorted high-to-low
    a = next(r for r in ratings if r.model == "a")
    assert a.source == "evaluated" and a.ci_low <= a.rating <= a.ci_high
    assert next(r for r in ratings if r.model == "b").source == "human"

    EloReport(
        arena="A", model="a", judge_model="j", n_bootstraps=2, seed=0, ratings=ratings
    ).write(tmp_path / "elo.json")
    assert (tmp_path / "elo.json").exists()
