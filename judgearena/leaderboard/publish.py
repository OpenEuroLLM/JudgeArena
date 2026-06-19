"""Publish leaderboard results to a Hugging Face dataset bundle."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from judgearena.estimate_elo_ratings import _winner_to_pref, fit_bradley_terry
from judgearena.leaderboard.board import build_board
from judgearena.leaderboard.panel import Panel, load_panel
from judgearena.log import get_logger

logger = get_logger(__name__)


def _record_label(rec: dict) -> str:
    return f"{rec['model']} #{rec['tag']}" if rec.get("tag") else rec["model"]


def _opt(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)


def _anchor_calibration(panel: Panel, *, n_bootstrap: int = 100, seed: int = 0) -> dict:
    """Human-ELO vs Judge-ELO for the anchor models (judge-side bootstrap CIs)."""
    battles = panel.battles
    if battles is None or len(battles) == 0:
        return {"mae": float("nan"), "spearman": float("nan"), "points": []}

    baseline = panel.meta.get("baseline_model")
    method = panel.meta.get("scorer", {}).get("method", "soft")
    judge_col = "judge_pref_hard" if method == "hard" else "judge_pref"

    human_df = pd.DataFrame(
        {
            "model_a": battles["model_a"],
            "model_b": battles["model_b"],
            "pref_hard": battles["human_winner"].map(_winner_to_pref),
        }
    )
    judge_df = pd.DataFrame(
        {
            "model_a": battles["model_a"],
            "model_b": battles["model_b"],
            "pref": battles[judge_col],
        }
    )
    human_elo = fit_bradley_terry(human_df, pref_col="pref_hard", baseline_model=baseline)
    judge_elo = fit_bradley_terry(judge_df, pref_col="pref", baseline_model=baseline)

    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        sample = judge_df.sample(
            n=len(judge_df), replace=True, random_state=int(rng.integers(0, 2**31))
        )
        ratings = fit_bradley_terry(sample, pref_col="pref", baseline_model=baseline)
        for model, value in ratings.items():
            boot.setdefault(model, []).append(value)

    points = []
    for model in sorted(m for m in human_elo if m in judge_elo):
        vals = boot.get(model, [])
        ci = (
            [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            if vals else [float("nan"), float("nan")]
        )
        points.append(
            {
                "model": model,
                "human_elo": float(human_elo[model]),
                "judge_elo": float(judge_elo[model]),
                "judge_ci": ci,
            }
        )

    mae = (
        float(np.mean([abs(p["human_elo"] - p["judge_elo"]) for p in points]))
        if points else float("nan")
    )
    if len(points) >= 2:
        h = pd.Series([p["human_elo"] for p in points])
        j = pd.Series([p["judge_elo"] for p in points])
        spearman = float(h.rank().corr(j.rank()))
    else:
        spearman = float("nan")
    return {"mae": mae, "spearman": spearman, "points": points}


def build_bundle(panel: Panel, records: list[dict]) -> dict:
    """Assemble the published leaderboard bundle dict from a panel + records."""
    overall = build_board(panel, records)
    by_label = {_record_label(r): r for r in records}

    rows = []
    for _, row in overall.iterrows():
        entry = {
            "rank": int(row["rank"]),
            "model": row["model"],
            "elo": float(row["elo"]),
            "ci_low": _opt(row["ci_low"]),
            "ci_high": _opt(row["ci_high"]),
            "n": int(row["n"]),
            "is_submission": bool(row["is_submission"]),
            "winrate": None,
            "winrate_per_lang": {},
        }
        if row["is_submission"]:
            rec = by_label.get(row["model"])
            if rec is not None:
                entry["winrate"] = rec.get("winrate_overall")
                entry["winrate_per_lang"] = rec.get("winrate_per_lang", {})
        rows.append(entry)

    languages = sorted((panel.meta.get("kappa_per_language") or {}).keys())
    by_language: dict[str, list[dict]] = {}
    for lang in languages:
        lb = build_board(panel, records, lang=lang)
        by_language[lang] = [
            {
                "rank": int(r["rank"]),
                "model": r["model"],
                "elo": float(r["elo"]),
                "n": int(r["n"]),
                "is_submission": bool(r["is_submission"]),
            }
            for _, r in lb.iterrows()
        ]

    meta = panel.meta
    return {
        "panel": {
            "panel_version": meta.get("panel_version"),
            "panel_hash": meta.get("panel_hash"),
            "judge_model": meta.get("judge_model"),
            "baseline_model": meta.get("baseline_model"),
            "mae_vs_human": meta.get("mae_vs_human"),
            "kappa_per_language": meta.get("kappa_per_language", {}),
            "scorer": meta.get("scorer", {}),
            "generated_utc": datetime.now(UTC).isoformat(),
        },
        "languages": languages,
        "rows": rows,
        "by_language": by_language,
        "calibration": _anchor_calibration(panel),
    }


def build_scores_frame(items: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    """Long-form (model, tag, lang, judge_pref) from each (record, battles)."""
    cols = ["model", "tag", "lang", "judge_pref"]
    frames = []
    for rec, battles in items:
        if battles is None or len(battles) == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "model": rec["model"],
                    "tag": rec.get("tag"),
                    "lang": battles["lang"].to_numpy(),
                    "judge_pref": battles["judge_pref"].to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def _upload_bundle(repo: str, bundle_dir: Path, *, token: str | None, create_pr: bool) -> str:
    """Upload the bundle folder to a HF dataset repo. Returns the PR/commit URL."""
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo, repo_type="dataset", exist_ok=True, token=token)
    return str(
        upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(bundle_dir),
            commit_message="Update leaderboard bundle",
            create_pr=create_pr,
            token=token,
        )
    )


def main_publish(argv: list[str] | None = None) -> Path:
    """Assemble the bundle from local results and upload it to a HF dataset."""
    ap = argparse.ArgumentParser(prog="judgearena-publish")
    ap.add_argument("--panel-dir", required=True)
    ap.add_argument("--repo", default=None, help="HF dataset repo id, e.g. user/leaderboard (required unless --dry-run).")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="Local bundle dir (default: a temp dir).")
    ap.add_argument("--push", action="store_true", help="Commit directly instead of a PR.")
    ap.add_argument("--dry-run", action="store_true", help="Build the bundle locally only; skip the HF upload.")
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.repo:
        ap.error("--repo is required unless --dry-run is given.")

    panel = load_panel(args.panel_dir)
    panel_version = panel.meta.get("panel_version", "")
    root = Path(args.results_dir) / panel_version

    items: list[tuple[dict, pd.DataFrame | None]] = []
    for result_path in sorted(root.glob("*/result.json")):
        d = result_path.parent
        rec = json.loads(result_path.read_text())
        battles_path = d / "battles.parquet"
        battles = pd.read_parquet(battles_path) if battles_path.exists() else None
        items.append((rec, battles))

    records = [rec for rec, _ in items]
    bundle = build_bundle(panel, records)
    scores = build_scores_frame(items)

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="lb-bundle-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(bundle, indent=2) + "\n")
    scores.to_parquet(out_dir / "scores.parquet", index=False)

    if args.dry_run:
        logger.info("Dry run: wrote bundle to %s (no upload).", out_dir)
        return out_dir

    token = os.environ.get(args.token_env)
    url = _upload_bundle(args.repo, out_dir, token=token, create_pr=not args.push)
    logger.info("Published bundle to %s (%s)", args.repo, url)
    return out_dir


if __name__ == "__main__":
    main_publish()
