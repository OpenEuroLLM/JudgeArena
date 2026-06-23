"""judgearena-submit: score a model against a frozen panel and open a PR."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download, upload_folder

from judgearena.config import JudgeArgs
from judgearena.estimate_elo_ratings import _slugify
from judgearena.evaluate import PairScore
from judgearena.leaderboard.anchors import save_anchor_caches
from judgearena.leaderboard.battles import judge_pool_battles, sample_pool_battles
from judgearena.leaderboard.pool import load_pool, save_pool
from judgearena.leaderboard.pool_fit import extend_pool, place_against_pool
from judgearena.leaderboard.score import generate_panel_completions
from judgearena.log import get_logger

logger = get_logger(__name__)


def _resolve_panel_version(repo_files: list[str]) -> str:
    """Latest panel version present in the dataset (numeric-aware on the suffix)."""
    from judgearena.leaderboard.assemble import latest_panel_version

    versions = {f.split("/")[1] for f in repo_files if f.startswith("panel/") and "/" in f[6:]}
    return latest_panel_version(sorted(versions))


def _download_panel(repo: str, version: str) -> Path:
    local = snapshot_download(
        repo_id=repo, repo_type="dataset", allow_patterns=[f"panel/{version}/*"]
    )
    return Path(local) / "panel" / version


def _bump_version(version: str) -> str:
    """Increment the trailing integer of a version string (e.g. 'v1' → 'v2', 'v10' → 'v11')."""
    m = re.search(r"(\d+)$", version)
    if not m:
        return version + "2"
    n = int(m.group(1))
    return version[: m.start()] + str(n + 1)


def main_submit(argv: list[str] | None = None) -> Path:
    """Generate + judge a model against a dataset panel; write + PR its record."""
    ap = argparse.ArgumentParser(prog="judgearena-submit")
    ap.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/leaderboard.")
    ap.add_argument("--model", required=True, help="{backend}/{path} model under eval.")
    ap.add_argument("--panel-version", default=None, help="Panel version (latest if omitted).")
    ap.add_argument("--out", default="results", help="Local output root.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-bootstraps", type=int, default=100)
    ap.add_argument("--submitter", default=None)
    ap.add_argument("--tag", default=None, help="Run tag to distinguish repeated submissions.")
    ap.add_argument("--dry-run", action="store_true", help="Write the record locally; skip the PR.")
    ap.add_argument("--into-pool", action="store_true", default=False,
                    help="Extend the reference pool (maintainer mode).")
    ap.add_argument("--n-per-pair", type=int, default=100,
                    help="Number of battles per pool-model pair.")
    args = ap.parse_args(argv)

    version = args.panel_version
    if version is None:
        files = HfApi().list_repo_files(repo_id=args.repo, repo_type="dataset")
        version = _resolve_panel_version(files)

    panel_dir = _download_panel(args.repo, version)

    # Shared work: load pool, generate completions, sample + judge battles.
    panel, pool_completions = load_pool(panel_dir)

    gen = panel.meta.get("generation_params", {})
    completions = generate_panel_completions(
        panel,
        args.model,
        max_out_tokens=gen.get("max_out_tokens", 32768),
        truncate_all_input_chars=gen.get("truncate_all_input_chars", 8192),
    )
    # generate_panel_completions returns one completion per battle row (in order);
    # build the per-(model, instruction) completions frame, one row per instruction.
    new_completions = (
        pd.DataFrame(
            {
                "model": args.model,
                "instruction": panel.battles["instruction"].to_numpy(),
                "lang": panel.battles["lang"].to_numpy(),
                "completion": completions,
            }
        )
        .drop_duplicates(subset=["instruction"])
        .reset_index(drop=True)
    )

    judge_cfg = JudgeArgs(model=panel.meta["judge_model"])
    scorer = PairScore(temperature=panel.meta.get("scorer", {}).get("temperature", 0.3))

    specs = sample_pool_battles(
        args.model, new_completions, panel, pool_completions,
        n_per_pair=args.n_per_pair, seed=args.seed,
    )
    new_battles = judge_pool_battles(
        specs, args.model, judge_cfg=judge_cfg, scorer=scorer,
        arena=panel.meta.get("arena"),
    )

    if args.into_pool:
        return _submit_into_pool(args, panel, pool_completions, new_completions, new_battles, version)
    else:
        return _submit_place(args, panel, new_completions, new_battles)


def _submit_place(args, panel, new_completions, new_battles) -> Path:
    """Default contributor path: place new model against frozen pool → record + PR."""
    record = place_against_pool(
        panel, args.model, new_battles,
        n_bootstraps=args.n_bootstraps, seed=args.seed,
    )
    record.submitter = args.submitter
    record.tag = args.tag

    slug = _slugify(args.model)
    if args.tag:
        slug = f"{slug}__{_slugify(args.tag)}"
    version = panel.meta["panel_version"]
    out_dir = Path(args.out) / version / slug
    record.save(out_dir)
    logger.info("Wrote record to %s (ELO %.1f)", out_dir, record.elo_overall)

    if args.dry_run:
        logger.info("Dry run: skipping PR upload.")
        return out_dir

    url = upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(out_dir),
        path_in_repo=f"records/{version}/{slug}",
        commit_message=f"Submit {args.model}" + (f" #{args.tag}" if args.tag else ""),
        create_pr=True,
    )
    logger.info("Opened submission PR: %s", url)
    return out_dir


def _submit_into_pool(args, panel, pool_completions, new_completions, new_battles, current_version: str) -> Path:
    """Maintainer path: extend the pool → bump version → save + upload (direct commit)."""
    new_version = _bump_version(current_version)
    new_panel, new_completions_merged = extend_pool(
        panel, pool_completions, args.model, new_completions, new_battles,
        bump_version=new_version,
    )

    out_dir = Path(args.out) / "panel" / new_version
    save_pool(new_panel, new_completions_merged, out_dir)
    save_anchor_caches(new_panel, out_dir)
    logger.info("Extended pool to %s at %s", new_version, out_dir)

    if args.dry_run:
        logger.info("Dry run: skipping pool upload.")
        return out_dir

    url = upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(out_dir),
        path_in_repo=f"panel/{new_version}",
        commit_message=f"Extend pool: add {args.model} → {new_version}",
        create_pr=False,
    )
    logger.info("Uploaded pool %s: %s", new_version, url)
    return out_dir


if __name__ == "__main__":
    main_submit()
