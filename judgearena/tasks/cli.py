"""Implement ``judgearena tasks list|show|validate`` inspection commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict

import yaml

from judgearena.datasets.registry import resolve_download_adapter
from judgearena.paths import data_root
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
        "download", help="Download the pinned sources for packaged tasks."
    )
    download.add_argument("tasks", nargs="*")
    download.add_argument(
        "--all", action="store_true", help="Download sources for every packaged task."
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
        elif args.command == "validate":
            if args.task is not None:
                _require(parser, tasks, args.task)
                print(f"Validated task {args.task!r}.")
            else:
                print(f"Validated {len(tasks)} task(s).")
        elif args.command == "download":
            if args.all and args.tasks:
                parser.error("download accepts task IDs or --all, not both")
            if not args.all and not args.tasks:
                parser.error("download requires at least one task ID or --all")
            selected = (
                list(tasks.values())
                if args.all
                else [_require(parser, tasks, task_id) for task_id in args.tasks]
            )
            tables_path = data_root / "tables"
            for resolved in selected:
                adapter = resolve_download_adapter(resolved.spec.dataset.adapter)
                adapter.download(resolved, tables_path)
                print(f"Downloaded task {resolved.task!r}.")
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
