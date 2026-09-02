"""Preference statistics and compatibility report exports."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from judgearena.reports import BattleReport, Report


class PrefSummary(BaseModel):
    """Win/loss/tie statistics for a preference series (0=A, 0.5=tie, 1=B)."""

    num_battles: int
    winrate: float
    num_wins: int
    num_losses: int
    num_ties: int
    num_missing: int

    def to_dict(self) -> dict[str, float | int]:
        return self.model_dump()


def compute_pref_summary(prefs: pd.Series) -> PrefSummary:
    """Compute win/loss/tie stats for preference series (0=A, 0.5=tie, 1=B)."""
    prefs = pd.Series(prefs, dtype="float64")
    valid = prefs.dropna()
    num_wins = int((valid < 0.5).sum())
    num_losses = int((valid > 0.5).sum())
    num_ties = int((valid == 0.5).sum())
    num_battles = int(len(prefs))
    denom = num_wins + num_losses + num_ties
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom else float("nan")
    return PrefSummary(
        num_battles=num_battles,
        winrate=winrate,
        num_wins=num_wins,
        num_losses=num_losses,
        num_ties=num_ties,
        num_missing=int(num_battles - denom),
    )


# Compatibility exports; report models now live together in judgearena.reports.
__all__ = [
    "BattleReport",
    "PrefSummary",
    "Report",
    "compute_pref_summary",
]
