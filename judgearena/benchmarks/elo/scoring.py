"""Runtime scoring adapters for ELO rating tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from judgearena.benchmarks.elo.rating import fit_bradley_terry

RatingFunction = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class EloScorer:
    """Registered rating implementation selected by an ELO task."""

    name: str
    fit: RatingFunction


_ELO_SCORERS = {
    "bradley_terry": EloScorer(
        name="bradley_terry",
        fit=fit_bradley_terry,
    )
}

ELO_SCORER_NAMES = frozenset(_ELO_SCORERS)


def resolve_elo_scorer(name: str) -> EloScorer:
    """Return the ELO scorer registered under ``name``."""
    try:
        return _ELO_SCORERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ELO scorer {name!r}; available: {sorted(ELO_SCORER_NAMES)}"
        ) from exc
