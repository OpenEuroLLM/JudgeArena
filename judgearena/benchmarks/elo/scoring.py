"""Runtime scoring adapters for ELO rating tasks.

A scorer turns judged battles into the metric a benchmark reports. Each
protocol defines its own, so scorers are named components selected by task
YAML rather than logic in the runner: the runner produces preferences, the
scorer owns the metric math.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from judgearena.benchmarks.elo.rating import fit_bradley_terry

RatingFunction = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class EloScorer:
    """Rating implementation selected by an ELO task's scoring adapter."""

    fit: RatingFunction


ELO_SCORERS: dict[str, EloScorer] = {
    "bradley_terry": EloScorer(fit=fit_bradley_terry),
}


def resolve_elo_scorer(name: str) -> EloScorer:
    """Return the scorer selected by an ELO task."""
    try:
        return ELO_SCORERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ELO scorer {name!r}; available: {sorted(ELO_SCORERS)}"
        ) from exc
