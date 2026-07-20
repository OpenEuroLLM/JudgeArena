"""Implement ``judgearena tasks list|show|validate`` inspection commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict

import yaml

from judgearena.tasks.loader import TaskDefinitionError
from judgearena.tasks.registry import TaskRegistry, UnknownTaskError


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
    argv: Sequence[str], *, registry: TaskRegistry | None = None
) -> None:
    """Run a task-registry command without starting an evaluation."""
    parser = _parser()
    args = parser.parse_args(list(argv))
    registry = registry or TaskRegistry()
    try:
        if args.command == "list":
            for task in registry.list():
                print(f"{task.task}\tv{task.task_version}\t{task.description}")
        elif args.command == "show":
            resolved = registry.get(args.task)
            output = resolved.model_dump()
            if args.resolved:
                output["_provenance"] = asdict(resolved.provenance)
            print(yaml.safe_dump(output, sort_keys=False).rstrip())
        elif args.command == "validate":
            if args.task is not None:
                registry.get(args.task)
                print(f"Validated task {args.task!r}.")
            else:
                report = registry.validate_all()
                print(f"Validated {report.count} task(s).")
    except (TaskDefinitionError, UnknownTaskError) as exc:
        parser.error(str(exc))
