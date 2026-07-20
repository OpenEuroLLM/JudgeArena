"""Shared lifecycle helpers for benchmark result directories."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from judgearena.artifacts.metadata import write_run_metadata
from judgearena.log import attach_file_handler, get_logger, make_run_log_path

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def slugify(value: str) -> str:
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


def write_run_metadata_safely(**kwargs):
    """Write reproducibility metadata without discarding completed results."""
    try:
        return write_run_metadata(**kwargs)
    except OSError as exc:
        logger.warning("Failed to write run metadata: %s", exc)
        return None
