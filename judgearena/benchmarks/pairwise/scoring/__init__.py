"""Metric registry for pairwise evaluation."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from judgearena.benchmarks.pairwise.scoring.metrics import (
    collapse_pairwise_battles,
    length_controlled_winrate,
    pairwise_win_rate,
)

PAIRWISE_METRICS: dict[str, Callable[[pd.DataFrame], dict[str, object]]] = {
    "pairwise_win_rate": pairwise_win_rate,
    "length_controlled_winrate": length_controlled_winrate,
}


def _group_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def calculate_metrics(
    battles: pd.DataFrame,
    requests,
) -> dict[str, dict[str, object]]:
    """Calculate configured metrics and their requested breakdowns."""
    calculated: dict[str, dict[str, object]] = {}
    for request in requests:
        try:
            metric = PAIRWISE_METRICS[request.metric]
        except KeyError as exc:
            raise ValueError(f"Unknown metric {request.metric!r}.") from exc
        values = dict(metric(battles))
        if request.group_by:
            groups: dict[str, list[dict[str, object]]] = {}
            for field in request.group_by:
                if field not in battles:
                    raise ValueError(
                        f"Metric {request.metric!r} cannot group by missing column "
                        f"{field!r}."
                    )
                groups[field] = [
                    {"group": _group_value(key), "values": metric(group)}
                    for key, group in battles.groupby(field, sort=False, dropna=False)
                ]
            values["groups"] = groups
        calculated[request.metric] = values
    return calculated


__all__ = [
    "PAIRWISE_METRICS",
    "calculate_metrics",
    "collapse_pairwise_battles",
    "length_controlled_winrate",
    "pairwise_win_rate",
]
