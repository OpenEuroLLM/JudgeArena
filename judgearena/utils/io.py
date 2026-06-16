"""Filesystem, dataset-download, caching, and timing helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.log import get_logger

logger = get_logger(__name__)


def _data_root_path() -> Path:
    raw = os.environ.get("JUDGEARENA_DATA") or os.environ.get("OPENJURY_DATA")
    if raw:
        return Path(raw).expanduser()
    return Path("~/judgearena-data/").expanduser()


data_root = _data_root_path()


def download_hf(name: str, local_path: Path):
    local_path.mkdir(exist_ok=True, parents=True)
    # downloads the model from huggingface into `local_path` folder
    snapshot_download(
        repo_id="judge-arena/judge-arena-dataset",
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
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
    from judgearena.instruction_dataset.m_arenahard import M_ARENA_HARD_BASELINES

    logger.info("Downloading all datasets in %s", data_root)
    local_path_tables = data_root / "tables"
    for dataset in (
        "alpaca-eval",
        "arena-hard-v0.1",
        "arena-hard-v2.0",
        *M_ARENA_HARD_BASELINES,
    ):
        if is_arena_hard_dataset(dataset):
            download_arena_hard(dataset=dataset, local_tables_path=local_path_tables)
        else:
            download_hf(name=dataset, local_path=local_path_tables)

    snapshot_download(
        repo_id="geoalgo/multilingual-contexts-to-be-completed",
        repo_type="dataset",
        allow_patterns="*",
        local_dir=data_root / "contexts",
        force_download=False,
    )

    from judgearena.instruction_dataset.mt_bench import download_mt_bench

    download_mt_bench()


class Timeblock:
    """Timer context manager"""

    def __init__(self, name: str | None = None, verbose: bool = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start
        if self.verbose:
            logger.info("%s", self)

    def __str__(self):
        name = self.name if self.name else "block"
        msg = f"{name} took {self.duration} seconds"
        return msg


def cache_function_dataframe(
    fun: Callable[[], pd.DataFrame],
    cache_name: str,
    ignore_cache: bool = False,
    cache_path: Path | None = None,
    parquet: bool = False,
) -> pd.DataFrame:
    """
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv.zip`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :param parquet: whether to store the data in parquet, if not specified use csv.zip
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = data_root / "cache"

    if parquet:
        cache_file = cache_path / (cache_name + ".parquet")
    else:
        cache_file = cache_path / (cache_name + ".csv.zip")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        logger.info("Loading cache %s", cache_file)
        if parquet:
            return pd.read_parquet(cache_file)
        else:
            return pd.read_csv(cache_file)
    else:
        logger.info(
            "Cache %s not found or ignore_cache set to True, regenerating the file",
            cache_file,
        )
        with Timeblock("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            if parquet:
                # object cols cannot be saved easily in parquet; numpy arrays must be
                # deep-converted to plain Python so str() produces ast.literal_eval-safe
                # repr (no "array([...])" syntax, which breaks literal_eval)
                import numpy as np

                def _to_python(x):
                    """Recursively convert numpy arrays/scalars to Python lists/dicts."""
                    if isinstance(x, np.ndarray):
                        return [_to_python(i) for i in x]
                    if isinstance(x, dict):
                        return {k: _to_python(v) for k, v in x.items()}
                    if isinstance(x, list):
                        return [_to_python(i) for i in x]
                    return x

                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].apply(_to_python).astype(str)
                df.to_parquet(cache_file, index=False)
                return pd.read_parquet(cache_file)
            else:
                df.to_csv(cache_file, index=False)
                return pd.read_csv(cache_file)


if __name__ == "__main__":
    download_all()
