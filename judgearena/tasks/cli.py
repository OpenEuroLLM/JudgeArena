"""Implement ``judgearena tasks list|show|validate|download`` inspection commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict

import yaml

from judgearena.tasks.registry import TaskDefinitionError, load_tasks, resolve_task
from judgearena.tasks.schema import ResolvedTaskSpec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="judgearena tasks")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List packaged tasks.")

    show = commands.add_parser("show", help="Show a packaged task definition.")
    show.add_argument("task")
    show.add_argument(
        "--resolved",
        action="store_true",
        help="Include inheritance and digest provenance.",
    )

    validate = commands.add_parser("validate", help="Validate packaged tasks.")
    validate.add_argument("task", nargs="?")
    download = commands.add_parser(
        "download", help="Prepare task data and scoring assets."
    )
    download.add_argument("task", nargs="*")
    download.add_argument(
        "--config_path",
        help="Use the effective dataset, baselines and metrics from a run YAML.",
    )
    return parser


def run_task_command(
    argv: Sequence[str], *, tasks: dict[str, ResolvedTaskSpec] | None = None
) -> None:
    """Run a task-registry command without starting an evaluation."""
    parser = _parser()
    args = parser.parse_args(list(argv))
    try:
        tasks = load_tasks() if tasks is None else tasks
        if args.command == "list":
            for resolved in tasks.values():
                spec = resolved.spec
                print(f"{spec.task}\tv{spec.task_version}\t{spec.description}")
        elif args.command == "show":
            resolved = _require(parser, tasks, args.task)
            output = resolved.model_dump()
            if resolved.selection is not None:
                output["_selection"] = asdict(resolved.selection)
            if args.resolved:
                output["_provenance"] = asdict(resolved.provenance)
            print(yaml.safe_dump(output, sort_keys=False).rstrip())
        elif args.command == "download":
            from judgearena.config import load_config
            from judgearena.utils.io import download_task

            if args.config_path:
                if args.task:
                    parser.error("Use task names or --config_path, not both.")
                cfg = load_config(args.config_path)
                download_task(
                    cfg.resolved_task,
                    model=cfg.model.name,
                    baseline=cfg.model.baseline,
                    filters=cfg.run.filters,
                )
                print(f"Prepared task {cfg.task!r}.")
            else:
                if not args.task:
                    parser.error("Provide task names or --config_path.")
                for name in args.task:
                    task = _require(parser, tasks, name)
                    download_task(task)
                    print(f"Prepared task {task.task!r}.")
        elif args.command == "validate":
            if args.task is not None:
                _require(parser, tasks, args.task)
                print(f"Validated task {args.task!r}.")
            else:
                print(f"Validated {len(tasks)} task(s).")
    except TaskDefinitionError as exc:
        parser.error(str(exc))


def _require(
    parser: argparse.ArgumentParser,
    tasks: dict[str, ResolvedTaskSpec],
    task_id: str,
) -> ResolvedTaskSpec:
    resolved = resolve_task(tasks, task_id)
    if resolved is None:
        known = ", ".join(sorted(tasks)) or "none"
        parser.error(f"unknown task {task_id!r}; registered tasks: {known}")
    return resolved
