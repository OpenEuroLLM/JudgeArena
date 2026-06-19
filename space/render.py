"""Pure render helpers for the leaderboard Space (no Gradio runtime)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def overview_table(bundle: dict) -> pd.DataFrame:
    rows = []
    for r in bundle.get("rows", []):
        ci = (
            f"[{r['ci_low']:.0f}, {r['ci_high']:.0f}]"
            if r.get("ci_low") is not None and r.get("ci_high") is not None
            else ""
        )
        winrate = f"{r['winrate']:.1%}" if r.get("winrate") is not None else ""
        rows.append(
            {
                "Rank": r["rank"],
                "Model": r["model"],
                "ELO": round(r["elo"], 1),
                "CI": ci,
                "Win rate": winrate,
                "n": r["n"],
                "Submission": "✓" if r["is_submission"] else "",
            }
        )
    return pd.DataFrame(
        rows, columns=["Rank", "Model", "ELO", "CI", "Win rate", "n", "Submission"]
    )


def available_languages(bundle: dict) -> list[str]:
    return list(bundle.get("languages", []))


def language_table(bundle: dict, lang: str) -> pd.DataFrame:
    rows = [
        {
            "Rank": r["rank"],
            "Model": r["model"],
            "ELO": round(r["elo"], 1),
            "Submission": "✓" if r["is_submission"] else "",
        }
        for r in bundle.get("by_language", {}).get(lang, [])
    ]
    return pd.DataFrame(rows, columns=["Rank", "Model", "ELO", "Submission"])


def language_bar(bundle: dict, lang: str) -> go.Figure:
    entries = bundle.get("by_language", {}).get(lang, [])
    fig = go.Figure(
        go.Bar(
            x=[e["model"] for e in entries],
            y=[e["elo"] for e in entries],
            marker_color=["#f59e0b" if e["is_submission"] else "#3b82f6" for e in entries],
        )
    )
    fig.update_layout(
        title=f"ELO by model — {lang}", yaxis_title="ELO", template="plotly_dark"
    )
    return fig


def distribution_fig(scores: pd.DataFrame, models: list[str]) -> go.Figure:
    fig = go.Figure()
    for model in models:
        sub = scores[scores["model"] == model]
        fig.add_trace(go.Histogram(x=sub["judge_pref"], name=model, opacity=0.6, nbinsx=20))
    fig.update_layout(
        barmode="overlay",
        title="Judge-preference distribution",
        xaxis_title="judge_pref  (0 = opponent wins, 1 = model wins)",
        template="plotly_dark",
    )
    return fig
