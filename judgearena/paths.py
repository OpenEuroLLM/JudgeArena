"""Filesystem paths and small file-IO helpers anchored at JudgeArena's data root.

This is a tiny leaf module so it can be imported by every other module
(including ``judgearena.instruction_dataset``) without pulling in the rest of
``judgearena.utils``, which would create an import cycle with the
``instruction_dataset`` package.

Symbols here are re-exported from :mod:`judgearena.utils` for backward
compatibility, so existing ``from judgearena.utils import data_root`` /
``from judgearena.utils import download_hf, read_df`` callers keep working.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.dataset_revisions import hf_revision


def _data_root_path() -> Path:
    raw = os.environ.get("JUDGEARENA_DATA") or os.environ.get("OPENJURY_DATA")
    if raw:
        return Path(raw).expanduser()
    return Path("~/judgearena-data/").expanduser()


data_root: Path = _data_root_path()


def download_hf(name: str, local_path: Path) -> None:
    """Download AlpacaEval-style instruction/output tables into ``local_path``."""
    local_path.mkdir(exist_ok=True, parents=True)
    repo_id = "geoalgo/llmjudge"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
        revision=hf_revision(repo_id),
    )


def read_df(filename: Path, **pandas_kwargs) -> pd.DataFrame:
    """Read a CSV/CSV-zip/parquet dataframe from disk."""
    assert filename.exists(), f"Dataframe file not found at {filename}"
    if filename.name.endswith(".csv.zip") or filename.name.endswith(".csv"):
        return pd.read_csv(filename, **pandas_kwargs)
    else:
        assert filename.name.endswith(".parquet"), f"Unsupported extension {filename}"
        return pd.read_parquet(filename, **pandas_kwargs)
