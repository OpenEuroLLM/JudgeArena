"""Unified CLI entrypoint for judgearena.

Builds a ``RunConfig`` from CLI flags (derived from the config model) and/or a
``--config_path`` YAML, then dispatches to the ELO or generate-and-judge flow
based on ``--task`` (``elo-`` prefix runs the ELO rating flow).
"""

from __future__ import annotations

from pydantic import ValidationError

from judgearena.config import build_run_config
from judgearena.estimate_elo_ratings import main as main_elo
from judgearena.generate_and_evaluate import main as main_generate_and_evaluate
from judgearena.log import configure_logging, get_logger
from judgearena.tasks import ELO_TASK_PREFIX

logger = get_logger(__name__)


def _format_config_error(exc: ValidationError) -> str:
    """Render the first validation error as a single CLI-friendly line."""
    first = exc.errors()[0]
    loc = ".".join(str(p) for p in first.get("loc", ()))
    msg = first.get("msg", str(exc))
    return f"judgearena: error: {loc}: {msg}" if loc else f"judgearena: error: {msg}"


def cli(argv: list[str] | None = None) -> None:
    try:
        cfg = build_run_config(argv)
    except ValidationError as exc:
        raise SystemExit(_format_config_error(exc)) from exc

    configure_logging(cfg.run.verbosity, log_file=cfg.run.log_file)
    logger.debug("Running with config: %s", cfg.model_dump())
    if cfg.task.startswith(ELO_TASK_PREFIX):
        main_elo(cfg)
    else:
        main_generate_and_evaluate(cfg)


if __name__ == "__main__":
    cli()
