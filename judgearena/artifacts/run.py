"""Shared lifecycle helpers for benchmark result directories."""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from logging import FileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from judgearena.artifacts.metadata import write_run_metadata
from judgearena.log import attach_file_handler, get_logger, make_run_log_path

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)
_run_handlers: ContextVar[list[FileHandler] | None] = ContextVar(
    "run_log_handlers", default=None
)


@contextmanager
def run_logging() -> Generator[None, None, None]:
    """Detach and close this run's automatic logs when its runner exits."""
    handlers: list[FileHandler] = []
    token = _run_handlers.set(handlers)
    try:
        yield
    finally:
        _run_handlers.reset(token)
        for handler in handlers:
            get_logger().removeHandler(handler)
            handler.close()


def safe_filename(value: str) -> str:
    """Return a filesystem-safe model, task, or benchmark name."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "value"


def prepare_run_directory(
    cfg: RunConfig, path: str | Path, *, attach_log: bool = True
) -> Path:
    """Create a result directory and persist its resolved configuration."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if attach_log and not cfg.run.no_log_file:
        previous_handlers = get_logger().handlers.copy()
        handler = attach_file_handler(make_run_log_path(output_dir))
        run_handlers = _run_handlers.get()
        if run_handlers is not None and handler not in previous_handlers:
            run_handlers.append(handler)

    from judgearena.config import dump_config

    dump_config(cfg, output_dir / "config.yaml")
    return output_dir


def write_run_metadata_safely(**kwargs):
    """Write reproducibility metadata without discarding completed results."""
    try:
        return write_run_metadata(**kwargs)
    except OSError as exc:
        logger.warning("Failed to write run metadata: %s", exc)
        return None
