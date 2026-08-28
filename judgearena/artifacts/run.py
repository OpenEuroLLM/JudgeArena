"""Shared lifecycle helpers for benchmark result directories."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from judgearena.artifacts.metadata import write_run_metadata
from judgearena.log import attach_file_handler, get_logger, make_run_log_path

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)
_RUN_LOG_CONTEXT: ContextVar[str | None] = ContextVar(
    "judgearena_run_log_context", default=None
)


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
        attach_file_handler(make_run_log_path(output_dir))

    from judgearena.config import dump_config

    dump_config(cfg, output_dir / "config.yaml")
    return output_dir


def atomic_write_path[T](path: str | Path, writer: Callable[[Path], T]) -> T:
    """Publish one file atomically after its writer completes successfully."""
    final_path = Path(path)
    temporary_path = final_path.with_name(f".{final_path.name}.{uuid4().hex}.tmp")
    try:
        result = writer(temporary_path)
        os.replace(temporary_path, final_path)
        return result
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_unique_run_directory(
    cfg: RunConfig,
    parent: str | Path,
    *,
    task: str,
) -> Path:
    """Reserve a unique invocation directory and persist config before work."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    for _ in range(3):
        output_dir = parent / f"{safe_filename(task)}-{timestamp}-{uuid4().hex}"
        try:
            output_dir.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        try:
            from judgearena.config import dump_config

            atomic_write_path(
                output_dir / "config.yaml",
                lambda temporary_path: dump_config(cfg, temporary_path),
            )
        except Exception:
            output_dir.rmdir()
            raise
        return output_dir
    raise RuntimeError("Could not reserve a unique run directory after 3 attempts.")


@contextmanager
def scoped_run_file_logging(cfg: RunConfig, output_dir: str | Path) -> Iterator[None]:
    """Attach one automatic run log for this scope, then remove and close it."""
    if cfg.run.no_log_file:
        yield
        return

    run_context = uuid4().hex
    handler = attach_file_handler(make_run_log_path(output_dir))
    handler.addFilter(lambda _record: _RUN_LOG_CONTEXT.get() == run_context)
    root_logger = get_logger()
    token = _RUN_LOG_CONTEXT.set(run_context)
    try:
        yield
    finally:
        _RUN_LOG_CONTEXT.reset(token)
        root_logger.removeHandler(handler)
        handler.close()


def write_run_metadata_safely(**kwargs):
    """Write reproducibility metadata without discarding completed results."""
    try:
        return write_run_metadata(**kwargs)
    except OSError as exc:
        logger.warning("Failed to write run metadata: %s", exc)
        return None
