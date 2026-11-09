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
from judgearena.prompts.spec import (
    config_prompts,
    prompt_file_fields,
    serialize_run_prompts,
)
from judgearena.utils.extensions import (
    component_references,
    external_component_reference,
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
    serialize_run_prompts(cfg, saved)
    from judgearena.benchmarks.scoring import _metric_type
    from judgearena.datasets.registry import dataset_adapter_type

    references = list(component_references(saved))
    for owner, key in references:
        if python_reference(owner[key]) is None:
            resolve = dataset_adapter_type if key == "adapter" else _metric_type
            try:
                reference = external_component_reference(resolve(owner[key]))
            except ValueError:
                # Invalid metrics are reported by scoring; still save the run.
                continue
            if reference is not None:
                if key == "metric" and not owner.get("name"):
                    owner["name"] = owner[key]
                owner[key] = reference
    references.extend((prompt, "parser") for prompt in config_prompts(saved))
    for index, (owner, key) in enumerate(references):
        reference = python_reference(owner.get(key))
        if reference is not None:
            if key == "metric" and not owner.get("name"):
                owner["name"] = owner[key]
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
    from judgearena.tasks.registry import _canonical_sha256

    resolved = saved["resolved_task"]
    resolved["provenance"]["resolved_sha256"] = _canonical_sha256(resolved["spec"])
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
