"""The reference-pool artifact: a Panel plus a per-(model, instruction) completions store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from judgearena.leaderboard.panel import Panel, load_panel, save_panel

POOL_COMPLETION_COLUMNS = ["model", "instruction", "lang", "completion"]


def save_pool(panel: Panel, completions: pd.DataFrame, directory: str | Path) -> Path:
    directory = Path(directory)
    out = save_panel(panel, directory)  # writes battles.parquet + panel.json (+ panel_hash)
    completions[POOL_COMPLETION_COLUMNS].to_parquet(out / "completions.parquet", index=False)
    return out


def load_pool(directory: str | Path) -> tuple[Panel, pd.DataFrame]:
    directory = Path(directory)
    panel = load_panel(directory)
    cpath = directory / "completions.parquet"
    completions = (
        pd.read_parquet(cpath) if cpath.exists()
        else pd.DataFrame(columns=POOL_COMPLETION_COLUMNS)
    )
    return panel, completions


def pool_models(panel: Panel) -> list[str]:
    return list(panel.meta.get("pool_models", []))


def anchor_models(panel: Panel) -> list[str]:
    return list(panel.meta.get("anchor_models", []))


def pool_completion(completions: pd.DataFrame, model: str, instruction: str) -> str | None:
    hit = completions[(completions["model"] == model) & (completions["instruction"] == instruction)]
    return None if hit.empty else str(hit.iloc[0]["completion"])
