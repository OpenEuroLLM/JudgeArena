"""Persistable data model for arena battles and ELO ratings.

A :class:`Battle` is the atomic unit of an arena evaluation: two models, an
outcome, and where that outcome came from (an LLM judge or human votes). ELO
ratings are a pure function of a list of battles, so persisting the battles
(plus the bootstrap ratings) is enough to reconstruct or re-analyse a run --
completions and judge transcripts stay in the cache for now.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

# Winners accepted by compute_bradley_terry.
WINNERS = frozenset({"model_a", "model_b", "tie", "tie (bothbad)"})


@dataclass(frozen=True)
class Battle:
    """One pairwise outcome. ``source`` records its provenance."""

    model_a: str
    model_b: str
    winner: str  # one of WINNERS
    source: str  # "llm-judge" | "human"
    question_id: str | None = None  # join key back to cache / transcripts
    judge_model: str | None = None  # llm-judge battles only

    @classmethod
    def from_dict(cls, d: dict) -> Battle:
        """Build from a dict, ignoring unknown keys (forward-compatible)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def write_battles(path: str | Path, battles: list[Battle]) -> None:
    """Write battles as JSON Lines."""
    with Path(path).open("w") as f:
        for b in battles:
            f.write(json.dumps(asdict(b)) + "\n")


def read_battles(path: str | Path) -> list[Battle]:
    """Read battles from a JSON Lines file."""
    with Path(path).open() as f:
        return [Battle.from_dict(json.loads(line)) for line in f if line.strip()]


def battles_to_frame(battles: list[Battle]) -> pd.DataFrame:
    """Tabular view, suitable for ``compute_bradley_terry(..., winner_col='winner')``."""
    return pd.DataFrame(asdict(b) for b in battles)


@dataclass(frozen=True)
class RatingEntry:
    """One model's place on the leaderboard."""

    model: str
    rating: float  # mean over bootstraps
    ci_low: float
    ci_high: float
    n_battles: int
    source: str  # "evaluated" (model under test) | "human"


@dataclass
class EloReport:
    """The leaderboard plus the run metadata that produced it."""

    arena: str
    model: str
    judge_model: str
    n_bootstraps: int
    seed: int
    ratings: list[RatingEntry]

    def write(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(asdict(self), f, indent=2)


def summarize_bootstrap(
    bootstrap_ratings: list[dict[str, float]],
    battle_counts: dict[str, int],
    model_under_test: str,
    ci: tuple[float, float] = (2.5, 97.5),
) -> list[RatingEntry]:
    """Collapse per-bootstrap ratings into one :class:`RatingEntry` per model,
    sorted from highest rating to lowest."""
    models = sorted({m for r in bootstrap_ratings for m in r})
    entries = []
    for m in models:
        vals = np.array([r[m] for r in bootstrap_ratings if m in r], dtype=float)
        lo, hi = np.percentile(vals, ci)
        entries.append(
            RatingEntry(
                model=m,
                rating=float(vals.mean()),
                ci_low=float(lo),
                ci_high=float(hi),
                n_battles=int(battle_counts.get(m, 0)),
                source="evaluated" if m == model_under_test else "human",
            )
        )
    entries.sort(key=lambda e: -e.rating)
    return entries
