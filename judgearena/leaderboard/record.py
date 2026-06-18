"""The leaderboard result record for one scored model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass
class ResultRecord:
    """A single leaderboard row plus its per-battle judge scores."""

    model: str
    panel_version: str
    panel_hash: str
    judge_model: str
    elo_overall: float
    elo_std: float
    elo_ci: list[float]
    elo_per_lang: dict[str, float]
    winrate_overall: float
    winrate_per_lang: dict[str, float]
    n_battles: int
    n_battles_per_lang: dict[str, int]
    kappa_per_lang: dict[str, float]
    mae_vs_human: float
    scorer: dict
    generation_params: dict
    seed: int
    schema_version: str = "1"
    submitter: str | None = None
    tag: str | None = None
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    battles: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("battles", None)
        return data

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        self.battles.to_parquet(directory / "battles.parquet", index=False)
        return directory
