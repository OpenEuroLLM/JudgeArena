"""judgearena-submit: score a model against a frozen panel."""

from __future__ import annotations

import argparse
from pathlib import Path

from judgearena.config import JudgeArgs
from judgearena.estimate_elo_ratings import _slugify
from judgearena.leaderboard.panel import load_panel
from judgearena.leaderboard.score import generate_panel_completions, score_against_panel
from judgearena.log import get_logger

logger = get_logger(__name__)


def main_submit(argv: list[str] | None = None) -> Path:
    """Generate + judge a model against a frozen panel; write its ResultRecord."""
    ap = argparse.ArgumentParser(prog="judgearena-submit")
    ap.add_argument("--panel-dir", required=True, help="Directory of a frozen panel.")
    ap.add_argument("--model", required=True, help="{backend}/{path} model under eval.")
    ap.add_argument("--out", default="results", help="Root output directory.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstraps", type=int, default=100)
    ap.add_argument("--submitter", default=None)
    args = ap.parse_args(argv)

    panel = load_panel(args.panel_dir)
    gen = panel.meta.get("generation_params", {})
    completions = generate_panel_completions(
        panel,
        args.model,
        max_out_tokens=gen.get("max_out_tokens", 32768),
        truncate_all_input_chars=gen.get("truncate_all_input_chars", 8192),
    )
    record = score_against_panel(
        panel,
        completions,
        model=args.model,
        judge_cfg=JudgeArgs(model=panel.meta["judge_model"]),
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        generation_params=gen,
    )
    record.submitter = args.submitter
    out_dir = Path(args.out) / panel.meta["panel_version"] / _slugify(args.model)
    record.save(out_dir)
    logger.info("Wrote result to %s (ELO %.1f)", out_dir, record.elo_overall)
    return out_dir


if __name__ == "__main__":
    main_submit()
