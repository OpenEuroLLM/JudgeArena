"""Filesystem, dataset-download, caching, and timing helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from judgearena.log import get_logger

logger = get_logger(__name__)


def _data_root_path() -> Path:
    raw = os.environ.get("JUDGEARENA_DATA") or os.environ.get("OPENJURY_DATA")
    if raw:
        return Path(raw).expanduser()
    return Path("~/judgearena-data/").expanduser()


data_root = _data_root_path()


def download_hf(name: str, local_path: Path):
    """Download the HF sources a packaged task declares in its YAML."""
    from judgearena.tasks.registry import get_packaged_task

    resolved_task = get_packaged_task(name)
    if resolved_task is None:
        raise ValueError(f"Unknown task {name!r}.")
    from judgearena.datasets.registry import resolve_download_adapter

    resolve_download_adapter(resolved_task.spec.dataset.adapter).download(
        resolved_task, local_path
    )


def read_df(filename: Path, **pandas_kwargs) -> pd.DataFrame:
    assert filename.exists(), f"Dataframe file not found at {filename}"
    if filename.name.endswith(".csv.zip") or filename.name.endswith(".csv"):
        return pd.read_csv(filename, **pandas_kwargs)
    else:
        assert filename.name.endswith(".parquet"), f"Unsupported extension {filename}"
        return pd.read_parquet(filename, **pandas_kwargs)


def safe_parse_int(env_var: str) -> int | None:
    """Parse an integer environment variable by name.

    Returns ``None`` when the variable is unset, blank, or malformed (a warning
    is logged for malformed values) so callers can fall back to a default
    instead of crashing at import time.
    """
    raw = os.getenv(env_var)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r; expected an integer.", env_var, raw)
        return None


def download_all():
    from judgearena.tasks.registry import load_tasks

    logger.info("Downloading all datasets in %s", data_root)
    local_path_tables = data_root / "tables"
    for task_id in load_tasks():
        download_hf(name=task_id, local_path=local_path_tables)


if __name__ == "__main__":
    download_all()
