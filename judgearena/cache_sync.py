"""Standalone CLI for synchronizing inference cache cells with Hugging Face."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from judgearena.cache_backfill import backfill_sources, log_report_summary, write_report
from judgearena.log import configure_logging, get_logger
from judgearena.store_sync import (
    DEFAULT_CACHE_REPO,
    fetch_remote_cells,
    iter_cell_dbs,
    push_cells,
    validate_path_filters,
)

logger = get_logger(__name__)


def _add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prefix",
        help="Exact repository path prefix (overrides task/provider/model/config).",
    )
    parser.add_argument("--task", help="Filter cells by benchmark task name.")
    parser.add_argument("--provider", help="Filter cells by provider, e.g. VLLM.")
    parser.add_argument(
        "--model",
        help="Filter cells by model path (slashes become '--' in folders).",
    )
    parser.add_argument("--config_hash", help="Filter cells by descriptor hash.")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store_root", required=True, help="Local store root.")
    parser.add_argument("--cache_hf_repo", default=DEFAULT_CACHE_REPO)
    parser.add_argument("--repo_type", default="dataset")
    parser.add_argument("--revision", default="main")
    parser.add_argument("-v", "--verbose", action="count", default=0)


def _resolve_prefix(args: argparse.Namespace) -> str | None:
    return validate_path_filters(
        prefix=args.prefix,
        task=args.task,
        provider=args.provider,
        model=args.model,
        config_hash=args.config_hash,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judgearena-cache",
        description="Fetch or push shared inference cache cells.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch",
        help="Discover and merge remote cells into the local store.",
    )
    _add_common(fetch)
    _add_filter_args(fetch)

    push = subparsers.add_parser(
        "push",
        help="Merge and upload local cells.",
    )
    _add_common(push)
    _add_filter_args(push)
    push.add_argument("--pushed_by", default=getpass.getuser())
    push.add_argument("--create_pr", action="store_true")
    push.add_argument(
        "--ensure_repo",
        action="store_true",
        help="Create the repository if it does not exist.",
    )
    push.add_argument(
        "--public",
        action="store_true",
        help="Create a public repository when used with --ensure_repo.",
    )

    backfill = subparsers.add_parser(
        "backfill",
        help="Backfill hosted judge outputs from saved run folders.",
    )
    backfill.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help="Run folders or parents containing saved judge annotations.",
    )
    backfill.add_argument("--store_root", required=True, help="Local store root.")
    backfill.add_argument(
        "--dry_run",
        action="store_true",
        help="Plan and report without writing inference rows.",
    )
    backfill.add_argument(
        "--report",
        type=Path,
        help="Optional path to write a JSON backfill report.",
    )
    backfill.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    configure_logging(getattr(args, "verbose", 0))

    if args.command == "backfill":
        report = backfill_sources(
            args.sources,
            args.store_root,
            dry_run=args.dry_run,
        )
        if args.report is not None:
            write_report(report, args.report)
        log_report_summary(report)
        return

    try:
        path_prefix = _resolve_prefix(args)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.command == "fetch":
        if path_prefix is None:
            logger.error(
                "Fetch requires a path filter. Provide --prefix or at least --task."
            )
            sys.exit(1)
        fetched = fetch_remote_cells(
            args.cache_hf_repo,
            args.store_root,
            path_prefix=path_prefix,
            repo_type=args.repo_type,
            revision=args.revision,
            strict=True,
        )
        if not fetched:
            logger.warning(
                "No remote cells matched prefix %r under %s",
                path_prefix,
                args.cache_hf_repo,
            )
        return

    db_paths = iter_cell_dbs(args.store_root, path_prefix=path_prefix)
    if not db_paths:
        logger.warning("No inference.db cells found under %s", args.store_root)
        return

    push_cells(
        args.cache_hf_repo,
        args.store_root,
        db_paths,
        pushed_by=args.pushed_by,
        repo_type=args.repo_type,
        revision=args.revision,
        create_pr=args.create_pr,
        ensure_repo=args.ensure_repo,
        private=not args.public,
        strict=True,
    )


if __name__ == "__main__":
    main()
