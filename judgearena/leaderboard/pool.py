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


def select_panel_instructions(panel: Panel, instructions_per_lang: int, *, seed: int) -> pd.DataFrame:
    """≤k unique instructions per language from the panel, sampled deterministically."""
    import numpy as np

    uniq = panel.battles[["instruction", "lang"]].drop_duplicates(subset=["instruction"])
    rng = np.random.default_rng(seed)
    parts = []
    for lang in sorted(uniq["lang"].unique()):
        group = uniq[uniq["lang"] == lang]
        if len(group) > instructions_per_lang:
            idx = rng.choice(len(group), size=instructions_per_lang, replace=False)
            group = group.iloc[sorted(idx)]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)[["instruction", "lang"]]


def completions_from_battles(battles: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, instruction) completions unpacked from a battles table's inline columns."""
    if battles.empty:
        return pd.DataFrame(columns=POOL_COMPLETION_COLUMNS)
    a = battles[["model_a", "instruction", "lang", "completion_a"]].rename(
        columns={"model_a": "model", "completion_a": "completion"}
    )
    b = battles[["model_b", "instruction", "lang", "completion_b"]].rename(
        columns={"model_b": "model", "completion_b": "completion"}
    )
    out = pd.concat([a, b], ignore_index=True)
    out = out.drop_duplicates(subset=["model", "instruction"]).reset_index(drop=True)
    return out[POOL_COMPLETION_COLUMNS]
