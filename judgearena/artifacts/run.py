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
import yaml

from judgearena.artifacts.metadata import to_jsonable, write_run_metadata
from judgearena.log import attach_file_handler, get_logger, make_run_log_path
from judgearena.prompts.parsing import resolve_judge_parser
from judgearena.prompts.spec import (
    config_prompts,
    expand_registered_prompt,
    prompt_file_fields,
)
from judgearena.utils.extensions import (
    component_references,
    python_object_reference,
    python_reference,
)

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

    saved = cfg.model_dump(mode="json")
    runtime_prompt = cfg.judge.prompt
    task_judge = cfg.resolved_task.spec.protocol.judge
    if (
        runtime_prompt is not None
        and runtime_prompt.preset is not None
        and not runtime_prompt.options
        and (
            runtime_prompt == task_judge.prompt
            or (
                task_judge.prompt is None
                and runtime_prompt.preset == task_judge.default_prompt_preset
            )
        )
    ):
        # Inherit the identical task selection, including its category routing.
        saved["judge"]["prompt"] = None
    for prompt in config_prompts(saved, expand_names=True):
        expand_registered_prompt(prompt)
        parser = (
            resolve_judge_parser(prompt["parser"]) if prompt.get("parser") else None
        )
        if parser is not None and (reference := python_object_reference(parser)):
            prompt["parser"] = reference
    references = list(component_references(saved))
    references.extend((prompt, "parser") for prompt in config_prompts(saved))
    for index, (owner, key) in enumerate(references):
        reference = python_reference(owner.get(key))
        if reference is not None:
            source, symbol = reference
            target = output_dir / "components" / str(index) / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, target)
            owner[key] = f"{target.relative_to(output_dir)}:{symbol}"
    for index, prompt in enumerate(config_prompts(saved)):
        for owner, name in prompt_file_fields(prompt):
            source = Path(owner[name])
            target = output_dir / "prompts" / str(index) / name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, target)
            owner[name] = str(target.relative_to(output_dir))
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(saved, sort_keys=False), encoding="utf-8"
    )
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
