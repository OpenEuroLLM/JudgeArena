"""judgearena-board: assemble and render the leaderboard ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from judgearena.estimate_elo_ratings import fit_bradley_terry
from judgearena.leaderboard.panel import Panel, load_panel
from judgearena.log import get_logger

logger = get_logger(__name__)

_BOARD_COLUMNS = ["model", "elo", "ci_low", "ci_high", "n", "is_submission"]


def build_board(
    panel: Panel, records: list[dict], *, lang: str | None = None
) -> pd.DataFrame:
    """Merge panel anchor ratings with submission records into a ranked ladder."""
    panel_hash = panel.meta.get("panel_hash")
    baseline_model = panel.meta.get("baseline_model")
    if baseline_model is None:
        logger.warning(
            "Panel has no baseline_model; absolute ELOs may drift between "
            "anchor and submission fits (relative order is still sound)."
        )
    method = panel.meta.get("scorer", {}).get("method", "soft")
    anchor_pref_col = "judge_pref_hard" if method == "hard" else "judge_pref"

    battles = panel.battles
    if lang is not None:
        battles = battles[battles["lang"] == lang]

    rows: list[dict] = []
    if len(battles):
        anchor_df = pd.DataFrame(
            {
                "model_a": battles["model_a"],
                "model_b": battles["model_b"],
                "pref": battles[anchor_pref_col],
            }
        )
        anchor_elo = fit_bradley_terry(
            anchor_df, pref_col="pref", baseline_model=baseline_model
        )
        counts = pd.concat([battles["model_a"], battles["model_b"]]).value_counts()
        for model, elo in anchor_elo.items():
            rows.append(
                {
                    "model": model, "elo": float(elo),
                    "ci_low": float("nan"), "ci_high": float("nan"),
                    "n": int(counts.get(model, 0)), "is_submission": False,
                }
            )

    for rec in records:
        if rec.get("panel_hash") != panel_hash:
            logger.warning(
                "Skipping %s: panel_hash mismatch (not this panel).", rec.get("model")
            )
            continue
        if lang is None:
            elo = rec.get("elo_overall", float("nan"))
            ci = rec.get("elo_ci", [float("nan"), float("nan")])
        else:
            elo = rec.get("elo_per_lang", {}).get(lang, float("nan"))
            ci = [float("nan"), float("nan")]
        rows.append(
            {
                "model": (
                    f"{rec.get('model')} #{rec['tag']}"
                    if rec.get("tag")
                    else rec.get("model")
                ),
                "elo": float(elo),
                "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                "n": int(rec.get("n_battles", 0)), "is_submission": True,
            }
        )

    board = pd.DataFrame(rows, columns=_BOARD_COLUMNS)
    board = board.sort_values(
        "elo", ascending=False, na_position="last"
    ).reset_index(drop=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    return board


def render_board(board: pd.DataFrame, fmt: str = "table") -> str:
    """Render a board DataFrame as ``table`` (default), ``markdown``, or ``csv``."""
    if fmt == "csv":
        return board.to_csv(index=False)

    def _ci(row) -> str:
        if pd.isna(row["ci_low"]):
            return ""
        return f"[{row['ci_low']:.0f}, {row['ci_high']:.0f}]"

    if fmt == "markdown":
        lines = ["| Rank | Model | ELO | CI | n |", "|---|---|---|---|---|"]
        for _, row in board.iterrows():
            mark = " *" if row["is_submission"] else ""
            lines.append(
                f"| {row['rank']} | {row['model']}{mark} | {row['elo']:.1f} | "
                f"{_ci(row)} | {row['n']} |"
            )
        return "\n".join(lines)

    lines = [f"{'Rank':>4}  {'Model':<44} {'ELO':>7}  {'n':>6}"]
    for _, row in board.iterrows():
        mark = " *" if row["is_submission"] else ""
        lines.append(
            f"{row['rank']:>4}  {str(row['model']) + mark:<44} "
            f"{row['elo']:>7.1f}  {row['n']:>6}"
        )
    return "\n".join(lines)


def main_board(argv: list[str] | None = None) -> None:
    """Print the leaderboard ladder for a panel from its local result records."""
    ap = argparse.ArgumentParser(prog="judgearena-board")
    ap.add_argument("--panel-dir", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--format", choices=["table", "markdown", "csv"], default="table")
    ap.add_argument("--lang", default=None)
    args = ap.parse_args(argv)

    panel = load_panel(args.panel_dir)
    panel_version = panel.meta.get("panel_version", "")
    results_root = Path(args.results_dir) / panel_version
    records = [
        json.loads(path.read_text())
        for path in sorted(results_root.glob("*/result.json"))
    ]
    board = build_board(panel, records, lang=args.lang)
    print(render_board(board, args.format))


if __name__ == "__main__":
    main_board()
