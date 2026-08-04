"""Implement ``judgearena tasks list|show|validate`` inspection commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict

import yaml

from judgearena.log import get_logger
from judgearena.tasks.loader import TaskDefinitionError
from judgearena.tasks.registry import load_tasks
from judgearena.tasks.schema import ResolvedTaskSpec

logger = get_logger(__name__)


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
            if args.resolved:
                output["_provenance"] = asdict(resolved.provenance)
            print(yaml.safe_dump(output, sort_keys=False).rstrip())
        elif args.command == "validate":
            if args.task is not None:
                _require(parser, tasks, args.task)
                logger.info("Validated task %r.", args.task)
            else:
                logger.info("Validated %d task(s).", len(tasks))
    except TaskDefinitionError as exc:
        parser.error(str(exc))


def _require(
    parser: argparse.ArgumentParser,
    tasks: dict[str, ResolvedTaskSpec],
    task_id: str,
) -> ResolvedTaskSpec:
    if task_id not in tasks:
        known = ", ".join(sorted(tasks)) or "none"
        parser.error(f"unknown task {task_id!r}; registered tasks: {known}")
    return tasks[task_id]
