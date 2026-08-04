"""Unified CLI entrypoint for JudgeArena.

Task-management subcommands are handled before the existing model-driven run
configuration and benchmark dispatch.
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from judgearena.benchmarks.elo.runner import main as main_elo
from judgearena.benchmarks.runner import run_benchmark
from judgearena.config import build_run_config
from judgearena.constants import ELO_TASK_PREFIX
from judgearena.log import configure_logging, get_logger

logger = get_logger(__name__)


def _format_config_error(exc: ValidationError) -> str:
    """Render the first validation error as a single CLI-friendly line."""
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", str(exc))
    return f"judgearena: error: {loc}: {msg}" if loc else f"judgearena: error: {msg}"


def cli(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    # `judgearena tasks {list,show,validate}` inspects packaged task definitions
    # instead of running an evaluation, so it has its own argparse grammar (see
    # tasks/cli.py) and must be routed before build_run_config, which only parses
    # run/eval flags and would reject the subcommand form.
    if args[:1] == ["tasks"]:
        from judgearena.tasks.cli import run_task_command

        run_task_command(args[1:])
        return

    try:
        cfg = build_run_config(args)
    except ValidationError as exc:
        raise SystemExit(_format_config_error(exc)) from exc

    configure_logging(cfg.run.verbosity, log_file=cfg.run.log_file)
    logger.debug("Running with config: %s", cfg.model_dump())
    if cfg.task.startswith(ELO_TASK_PREFIX):
        main_elo(cfg)
    else:
        run_benchmark(cfg)


if __name__ == "__main__":
    cli()
