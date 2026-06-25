"""Frozen leaderboard panel artifact: a battles table plus JSON metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PANEL_BATTLE_COLUMNS = [
    "battle_id",
    "lang",
    "model_a",
    "model_b",
    "instruction",
    "completion_a",
    "completion_b",
    "human_winner",
    "judge_pref",
    "judge_pref_hard",
    "challenger_opponent",
    "challenger_position",
    "axis_scores",  # JSON per-axis criteria scores ({} / null for single-score panels)
]


def panel_hash(battles: pd.DataFrame) -> str:
    """Order-independent sha256 hex of the battle rows."""
    payload = battles.sort_values("battle_id").to_json(orient="records")
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Panel:
    """A frozen evaluation panel: metadata + the anchor battles table."""

    meta: dict = field(default_factory=dict)
    battles: pd.DataFrame = field(default_factory=pd.DataFrame)


def save_panel(panel: Panel, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    panel.battles.to_parquet(directory / "battles.parquet", index=False)
    meta = {**panel.meta, "panel_hash": panel_hash(panel.battles)}
    (directory / "panel.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory


def load_panel(directory: str | Path) -> Panel:
    directory = Path(directory)
    battles = pd.read_parquet(directory / "battles.parquet")
    meta = json.loads((directory / "panel.json").read_text())
    return Panel(meta=meta, battles=battles)
