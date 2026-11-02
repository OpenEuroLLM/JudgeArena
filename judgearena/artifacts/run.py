"""Shared lifecycle helpers for benchmark result directories."""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from logging import FileHandler
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts.metadata import to_jsonable, write_run_metadata
from judgearena.log import attach_file_handler, get_logger, make_run_log_path
from judgearena.prompts.registry import prompt_option_files

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
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    for attempt in count():
        output_dir = Path(path) if attempt == 0 else Path(f"{path}-{attempt}")
        try:
            output_dir.mkdir()
            break
        except FileExistsError:
            continue
    if attach_log and not cfg.run.no_log_file:
        previous_handlers = get_logger().handlers.copy()
        handler = attach_file_handler(make_run_log_path(output_dir))
        run_handlers = _run_handlers.get()
        if run_handlers is not None and handler not in previous_handlers:
            run_handlers.append(handler)

    from judgearena.config import dump_config

    saved_cfg = cfg
    prompt = cfg.judge.prompt
    if prompt is not None:
        values = prompt.model_dump()
        for owner, names in (
            (values, ("system_file", "user_file")),
            (values["options"], prompt_option_files(prompt.preset)),
        ):
            for name in names:
                source = owner.get(name)
                if source is not None:
                    source = Path(source)
                    target = output_dir / "prompts" / name / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    copyfile(source, target)
                    owner[name] = target.relative_to(output_dir)
        saved_cfg = cfg.model_copy(
            update={
                "judge": cfg.judge.model_copy(
                    update={"prompt": prompt.model_copy(update=values)}
                )
            }
        )
    dump_config(saved_cfg, output_dir / "config.yaml")
    return output_dir


def write_run_metadata_safely(**kwargs):
    """Write reproducibility metadata without discarding completed results."""
    try:
        return write_run_metadata(**kwargs)
    except OSError as exc:
        logger.warning("Failed to write run metadata: %s", exc)
        return None


def write_annotations(annotations: pd.DataFrame, path: str | Path) -> None:
    """Save rows, JSON-encoding nested values and heterogeneous scalar columns.

    Mixed ID columns are encoded too: json.loads restores each ID's original
    type. Uniform integer or string ID columns keep their native representation.
    """
    path = Path(path)
    if path.suffix not in (".csv", ".parquet"):
        raise ValueError("Annotations require a .csv or .parquet path.")
    frame = annotations.copy()
    for column in frame:
        if pd.api.types.infer_dtype(frame[column], skipna=True).startswith("mixed"):
            frame[column] = frame[column].map(
                lambda value: (
                    None
                    if pd.api.types.is_scalar(value) and pd.isna(value)
                    else json.dumps(
                        to_jsonable(value), ensure_ascii=False, allow_nan=False
                    )
                )
            )
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
