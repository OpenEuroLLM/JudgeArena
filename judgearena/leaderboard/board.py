"""judgearena-board: assemble and render the leaderboard ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from judgearena.leaderboard.anchors import compute_anchor_ratings
from judgearena.leaderboard.assemble import _board_rows
from judgearena.leaderboard.panel import Panel, load_panel
from judgearena.log import get_logger

logger = get_logger(__name__)


def build_board(
    panel: Panel, records: list[dict], *, lang: str | None = None
) -> pd.DataFrame:
    """Merge panel anchor ratings with submission records into a ranked ladder."""
    if panel.meta.get("baseline_model") is None:
        logger.warning(
            "Panel has no baseline_model; absolute ELOs may drift between "
            "anchor and submission fits (relative order is still sound)."
        )
    ratings = compute_anchor_ratings(panel)
    if lang is None:
        anchor_elo = ratings["overall"]
        anchor_counts = ratings["counts_overall"]
    else:
        anchor_elo = ratings["per_lang"].get(lang, {})
        anchor_counts = ratings["counts_per_lang"].get(lang, {})
    return _board_rows(
        anchor_elo, anchor_counts, records, panel.meta.get("panel_hash"), lang=lang
    )


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
