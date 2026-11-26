"""Local release creation and submission commands."""

import argparse
from pathlib import Path

import pandas as pd

from judgearena.benchmarks.pairwise.graph import validate_graph_task
from judgearena.config import ModelArgs, RunConfig, resolve_cli_config
from judgearena.datasets import load_task_data, resolve_dataset_filters
from judgearena.datasets.data import PreferenceData
from judgearena.leaderboard.evaluate import evaluate_release
from judgearena.leaderboard.prepare import sample_battles
from judgearena.leaderboard.release import create_completion_release, create_release
from judgearena.leaderboard.settings import LeaderboardSettings, rating_metrics
from judgearena.leaderboard.submissions import add_submission, submission_table
from judgearena.tasks.schema import RecordedOpponents


class LeaderboardCreateConfig(RunConfig):
    """Expose release settings alongside the ordinary run CLI fields."""

    leaderboard: LeaderboardSettings


def run_leaderboard_command(argv):
    parser = argparse.ArgumentParser(prog="judgearena leaderboard", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", allow_abbrev=False)
    create.add_argument("--config", "--config_path", type=Path)
    create.add_argument("--candidate", type=Path)
    create.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate",
        allow_abbrev=False,
        epilog="Candidate settings: --model.name, --model.temperature, --model.engine_kwargs, etc. CLI values override the optional config.",
    )
    evaluate.add_argument("--leaderboard", type=Path, required=True)
    evaluate.add_argument("--config", "--config_path", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--answers", type=Path)
    evaluate.add_argument("--store-root", type=Path)
    evaluate.add_argument(
        "--n-bootstraps", "--bootstraps", dest="n_bootstraps", type=int, default=200
    )
    evaluate.add_argument("--seed", type=int, default=42)
    add = commands.add_parser("add", allow_abbrev=False)
    add.add_argument("--leaderboard", type=Path, required=True)
    add.add_argument("--submission", type=Path, required=True)
    add.add_argument("--collection", type=Path, required=True)
    table = commands.add_parser("table", allow_abbrev=False)
    table.add_argument("--leaderboard", type=Path, required=True)
    table.add_argument("--collection", type=Path, required=True)
    table.add_argument("--columns", nargs="+")
    view = table.add_mutually_exclusive_group()
    view.add_argument(
        "--overall", action="store_true", help="Show overall metrics only"
    )
    view.add_argument(
        "--breakdown", metavar="FIELD", help="Show one metric breakdown, e.g. lang"
    )
    table.add_argument("--sort-by")
    table.add_argument("--descending", action="store_true")
    args, overrides = parser.parse_known_args(argv)
    if args.command in ("add", "table"):
        if overrides:
            parser.error(f"Unrecognized arguments: {' '.join(overrides)}")
        if args.command == "add":
            print(add_submission(args.leaderboard, args.submission, args.collection))
        else:
            print(
                submission_table(
                    args.leaderboard,
                    args.collection,
                    columns=args.columns,
                    overall=args.overall,
                    breakdown=args.breakdown,
                    sort_by=args.sort_by,
                    ascending=not args.descending,
                ).to_string(index=False)
            )
        return
    if args.command == "create":
        declaration = resolve_cli_config(
            overrides, config_path=args.config, settings_cls=LeaderboardCreateConfig
        )
        if declaration.get("stage", "evaluate") != "evaluate":
            raise ValueError(
                "Use leaderboard evaluate --stage generate for generation-only runs"
            )
        settings = LeaderboardSettings.model_validate(
            declaration.pop("leaderboard", {})
        )
        # Creation has no candidate; this identity is never sent to a backend.
        declaration.setdefault("model", {}).setdefault(
            "name", f"leaderboard:{settings.version}"
        )
        cfg = RunConfig(**declaration)
        runner = cfg.resolved_task.spec.protocol.runner
        if runner not in {"pairwise", "single_answer"}:
            raise ValueError(
                f"Leaderboard submissions do not support runner {runner!r}"
            )
        explicit_panel = args.candidate is not None or (
            runner == "pairwise" and "sampling" in settings.model_fields_set
        )
        if explicit_panel and cfg.model.baseline is not None:
            raise ValueError(
                "Choose either model.baseline or a leaderboard opponent panel."
            )
        graph = explicit_panel or (
            cfg.model.baseline is None
            and isinstance(cfg.resolved_task.spec.protocol.baseline, RecordedOpponents)
        )
        if graph:
            validate_graph_task(cfg.resolved_task.spec.protocol)
        if args.output.exists():
            raise FileExistsError(args.output)
        configured = rating_metrics(cfg)
        filters = resolve_dataset_filters(cfg.resolved_task, cfg.run.filters)
        heldout = None
        if args.candidate:
            battles = pd.read_parquet(args.candidate / "reference_battles.parquet")
            panel = pd.read_parquet(args.candidate / "panel.parquet")
            heldout_path = args.candidate / "heldout_battles.parquet"
            if heldout_path.exists():
                heldout = pd.read_parquet(heldout_path)
        else:
            data = load_task_data(cfg.resolved_task).select(filters)
            if not graph:
                if "sampling" in settings.model_fields_set:
                    raise ValueError(
                        "Completion releases use run.sampling.limit and run.filters, not graph sampling"
                    )
                if "labels" in settings.model_fields_set and settings.labels != "judge":
                    raise ValueError(
                        "Completion releases use judge labels for submissions"
                    )
                create_completion_release(
                    cfg, data, args.output, version=settings.version
                )
                print(f"Release written to {args.output.resolve()}")
                return
            if not isinstance(data, PreferenceData):
                raise ValueError(
                    "Graph sampling requires PreferenceData with comparisons"
                )
            battles, panel, heldout = sample_battles(
                data.battles,
                settings,
                seed=cfg.run.seed,
                baseline_models=[m.baseline_model for _, m in configured],
                languages=filters.get("lang"),
                require_connected=bool(configured),
            )
        print(
            f"Preparing {len(battles)} reference comparisons and {len(panel)} panel comparisons; labels={settings.labels}, swap_mode={cfg.judge.swap_mode}",
            flush=True,
        )
        create_release(
            cfg,
            battles,
            panel,
            args.output,
            version=settings.version,
            labels=settings.labels,
            heldout=heldout,
            preparation={
                **settings.model_dump(),
                "input": "candidate" if args.candidate else "task",
                "languages": sorted(panel.lang.unique().tolist()),
                "sampling": None
                if args.candidate
                else {
                    **settings.sampling.model_dump(),
                    "opponents": panel.opponent_model.drop_duplicates().tolist(),
                },
                "split_population": "provided candidate"
                if args.candidate
                else "selected dataset",
            },
        )
        print(f"Release written to {args.output.resolve()}")
    else:
        declaration = resolve_cli_config(
            overrides, config_path=args.config, allowed_sections={"model", "stage"}
        )
        result = evaluate_release(
            args.leaderboard,
            ModelArgs(**declaration.get("model", {})),
            args.output,
            answers=args.answers,
            stage=declaration.get("stage", "evaluate"),
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
            store_root=args.store_root,
        )
        print(
            result.to_string(index=False)
            if isinstance(result, pd.DataFrame)
            else result
        )
