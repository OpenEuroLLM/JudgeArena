"""judgearena-submit: score a model against a frozen panel and open a PR."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, upload_folder

from judgearena.config import JudgeArgs
from judgearena.estimate_elo_ratings import _slugify
from judgearena.leaderboard.panel import load_panel
from judgearena.leaderboard.score import generate_panel_completions, score_against_panel
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
    args = ap.parse_args(argv)

    version = args.panel_version
    if version is None:
        files = HfApi().list_repo_files(repo_id=args.repo, repo_type="dataset")
        version = _resolve_panel_version(files)

    panel_dir = _download_panel(args.repo, version)
    panel = load_panel(panel_dir)

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
    record.tag = args.tag

    slug = _slugify(args.model)
    if args.tag:
        slug = f"{slug}__{_slugify(args.tag)}"
    out_dir = Path(args.out) / panel.meta["panel_version"] / slug
    record.save(out_dir)
    logger.info("Wrote result to %s (ELO %.1f)", out_dir, record.elo_overall)

    if args.dry_run:
        logger.info("Dry run: skipping PR upload.")
        return out_dir

    url = upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=str(out_dir),
        path_in_repo=f"records/{panel.meta['panel_version']}/{slug}",
        commit_message=f"Submit {args.model}" + (f" #{args.tag}" if args.tag else ""),
        create_pr=True,
    )
    logger.info("Opened submission PR: %s", url)
    return out_dir


if __name__ == "__main__":
    main_submit()
