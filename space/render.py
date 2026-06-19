"""Pure render helpers for the leaderboard Space (no Gradio runtime)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEMPLATE = "plotly_white"


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
        title=f"ELO by model — {lang}", yaxis_title="ELO", template=TEMPLATE
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
        template=TEMPLATE,
    )
    return fig


def _kappa_color(k: float) -> str:
    if k is None or (isinstance(k, float) and pd.isna(k)):
        return "#9ca3af"
    if k >= 0.6:
        return "#22c55e"
    if k >= 0.4:
        return "#f59e0b"
    return "#ef4444"


def calibration_fig(bundle: dict) -> go.Figure:
    pts = bundle.get("calibration", {}).get("points", [])
    if not pts:
        return go.Figure().update_layout(title="No calibration data", template=TEMPLATE)
    x = [p["human_elo"] for p in pts]
    y = [p["judge_elo"] for p in pts]
    lo = [p["judge_ci"][0] for p in pts]
    hi = [p["judge_ci"][1] for p in pts]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers+text", text=[p["model"] for p in pts],
            textposition="top center", name="anchors",
            error_y=dict(
                type="data", symmetric=False,
                array=[h - v for h, v in zip(hi, y, strict=False)],
                arrayminus=[v - lo_i for v, lo_i in zip(y, lo, strict=False)],
            ),
        )
    )
    span_lo, span_hi = min(min(x), min(y)), max(max(x), max(y))
    fig.add_trace(go.Scatter(x=[span_lo, span_hi], y=[span_lo, span_hi], mode="lines",
                             line=dict(dash="dot", color="#9ca3af"), name="y = x"))
    if len(pts) >= 2:
        m, b = np.polyfit(x, y, 1)
        fig.add_trace(go.Scatter(x=[span_lo, span_hi], y=[m * span_lo + b, m * span_hi + b],
                                 mode="lines", line=dict(color="#ef4444"), name="fit"))
    cal = bundle.get("calibration", {})
    fig.update_layout(
        title=f"Human-ELO vs Judge-ELO  (MAE {cal.get('mae', float('nan')):.1f}, "
              f"Spearman ρ {cal.get('spearman', float('nan')):.2f})",
        xaxis_title="Human-ELO", yaxis_title="Judge-ELO (CI = panel-resampling)",
        template=TEMPLATE,
    )
    return fig


def kappa_bar(bundle: dict) -> go.Figure:
    kp = bundle.get("panel", {}).get("kappa_per_language", {})
    langs = sorted(kp)
    vals = [kp[lang] for lang in langs]
    fig = go.Figure(
        go.Bar(x=vals, y=langs, orientation="h",
               marker_color=[_kappa_color(v) for v in vals])
    )
    fig.update_layout(title="Judge–human agreement (Cohen's κ) by language",
                      xaxis_title="κ", template=TEMPLATE)
    return fig


def winrate_heatmap(bundle: dict) -> go.Figure:
    langs = bundle.get("languages", [])
    subs = [r for r in bundle.get("rows", []) if r.get("is_submission")]
    if not subs or not langs:
        return go.Figure().update_layout(title="No submissions yet", template=TEMPLATE)
    z = [[(r.get("winrate_per_lang") or {}).get(lang) for lang in langs] for r in subs]
    fig = go.Figure(
        go.Heatmap(z=z, x=langs, y=[r["model"] for r in subs], zmin=0, zmax=1,
                   colorscale="RdYlGn", colorbar=dict(title="win rate"))
    )
    fig.update_layout(title="Win rate by language", template=TEMPLATE)
    return fig
