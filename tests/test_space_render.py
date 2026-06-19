"""Tests for the Space render helpers (offline; no Gradio runtime)."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

from space.app import load_bundle
from space.render import (
    available_languages,
    distribution_fig,
    language_bar,
    language_table,
    overview_table,
)


def _bundle():
    return {
        "panel": {"judge_model": "j", "mae_vs_human": 10.0},
        "languages": ["en", "fr"],
        "rows": [
            {"rank": 1, "model": "strong", "elo": 1200.0, "ci_low": None,
             "ci_high": None, "n": 100, "is_submission": False, "winrate": None},
            {"rank": 2, "model": "cand #seed-1", "elo": 1050.0, "ci_low": 1040.0,
             "ci_high": 1060.0, "n": 50, "is_submission": True, "winrate": 0.55},
        ],
        "by_language": {
            "en": [
                {"rank": 1, "model": "strong", "elo": 1190.0, "n": 50, "is_submission": False},
                {"rank": 2, "model": "cand #seed-1", "elo": 1051.0, "n": 25, "is_submission": True},
            ],
            "fr": [
                {"rank": 1, "model": "strong", "elo": 1210.0, "n": 50, "is_submission": False},
            ],
        },
    }


def test_overview_table_columns():
    df = overview_table(_bundle())
    assert list(df.columns) == ["Rank", "Model", "ELO", "CI", "Win rate", "n", "Submission"]
    cand = df[df["Model"] == "cand #seed-1"].iloc[0]
    assert cand["CI"] == "[1040, 1060]"
    assert cand["Win rate"] == "55.0%"
    assert cand["Submission"] == "✓"
    strong = df[df["Model"] == "strong"].iloc[0]
    assert strong["CI"] == "" and strong["Win rate"] == "" and strong["Submission"] == ""


def test_available_languages():
    assert available_languages(_bundle()) == ["en", "fr"]


def test_language_table():
    df = language_table(_bundle(), "en")
    assert list(df.columns) == ["Rank", "Model", "ELO", "Submission"]
    assert list(df["Model"]) == ["strong", "cand #seed-1"]


def test_language_bar_is_figure():
    fig = language_bar(_bundle(), "en")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_distribution_fig_per_model():
    scores = pd.DataFrame({
        "model": ["a", "a", "b", "b"], "tag": [None, None, None, None],
        "lang": ["en", "fr", "en", "fr"], "judge_pref": [0.2, 0.3, 0.7, 0.8],
    })
    fig = distribution_fig(scores, ["a", "b"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # one histogram trace per model


def test_load_bundle_local(tmp_path):
    (tmp_path / "leaderboard.json").write_text(json.dumps({"rows": [], "languages": []}))
    pd.DataFrame({"model": ["a"], "tag": [None], "lang": ["en"], "judge_pref": [0.5]}).to_parquet(
        tmp_path / "scores.parquet", index=False
    )
    bundle, scores = load_bundle(local_dir=str(tmp_path))
    assert bundle == {"rows": [], "languages": []}
    assert list(scores.columns) == ["model", "tag", "lang", "judge_pref"]


def _bundle_4a():
    b = _bundle()  # has rows (strong anchor, cand #seed-1 submission), languages en/fr, by_language
    b["rows"][1]["winrate_per_lang"] = {"en": 0.6, "fr": 0.4}
    b["rows"][0]["winrate_per_lang"] = {}
    b["panel"]["kappa_per_language"] = {"en": 0.8, "fr": 0.3}
    b["calibration"] = {
        "mae": 12.3, "spearman": 0.9,
        "points": [
            {"model": "strong", "human_elo": 1200.0, "judge_elo": 1190.0, "judge_ci": [1180.0, 1200.0]},
            {"model": "weak", "human_elo": 1000.0, "judge_elo": 1010.0, "judge_ci": [1000.0, 1020.0]},
        ],
    }
    return b


def test_calibration_fig():
    from space.render import calibration_fig
    fig = calibration_fig(_bundle_4a())
    assert len(fig.data) >= 2  # points + y=x (and regression line)


def test_kappa_bar():
    from space.render import kappa_bar
    fig = kappa_bar(_bundle_4a())
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == ["en", "fr"]


def test_winrate_heatmap():
    from space.render import winrate_heatmap
    fig = winrate_heatmap(_bundle_4a())
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == ["cand #seed-1"]  # submissions only


def test_existing_figs_use_white_template():
    from space.render import TEMPLATE
    assert TEMPLATE == "plotly_white"
