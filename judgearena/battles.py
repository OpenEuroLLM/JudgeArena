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

from judgearena.artifacts import to_jsonable


def preference_values(values: pd.Series) -> pd.Series:
    """Validate continuous preferences; missing judgments remain missing."""
    try:
        prefs = values.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Battle preferences must be numeric.") from exc
    if (~prefs.dropna().between(0, 1)).any():
        raise ValueError(
            "Battle preferences must be missing or finite values in [0, 1]."
        )
    return prefs


def hard_preferences(values: pd.Series) -> pd.Series:
    prefs = preference_values(values)
    return pd.Series(np.sign(prefs - 0.5) / 2 + 0.5, index=prefs.index)


def swap_comparison_slots(
    comparisons: pd.DataFrame, reverse_order: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Swap selected model/answer pairs, preserving row identity and metadata."""
    swapped = comparisons.copy()
    for prefix in ("completion", "model"):
        columns = [f"{prefix}_a", f"{prefix}_b"]
        swapped.loc[reverse_order, columns] = comparisons.loc[
            reverse_order, columns[::-1]
        ].to_numpy()
    return swapped


def pairwise_aliases(
    battles: pd.DataFrame,
    *,
    model_is_a: pd.Series,
) -> dict[str, pd.Series]:
    """Derive metric aliases from explicitly selected model slots."""
    aliases = {
        "model": battles["model_a"].where(model_is_a, battles["model_b"]),
        "baseline": battles["model_b"].where(model_is_a, battles["model_a"]),
    }
    if {"completion_a", "completion_b"} <= set(battles.columns):
        aliases.update(
            completion_model=battles["completion_a"].where(
                model_is_a, battles["completion_b"]
            ),
            completion_baseline=battles["completion_b"].where(
                model_is_a, battles["completion_a"]
            ),
        )
    return aliases


def focal_battles(battles: pd.DataFrame) -> pd.DataFrame:
    """Normalize rows carrying an evaluation model to the focal/opponent view."""
    if "evaluation_model" not in battles:
        return battles
    frame = battles.loc[battles["evaluation_model"].notna()].copy()
    required = {"model_a", "model_b", "pref"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation battles are missing columns: {missing}.")
    focal_is_a = frame["model_a"] == frame["evaluation_model"]
    focal_is_b = frame["model_b"] == frame["evaluation_model"]
    if not (focal_is_a ^ focal_is_b).all():
        raise ValueError(
            "Each evaluation battle must contain its evaluation model exactly once."
        )
    frame = frame.assign(**pairwise_aliases(frame, model_is_a=focal_is_a))
    frame["pref"] = frame["pref"].where(focal_is_a, 1 - frame["pref"])
    return frame


def judged_battles(
    comparisons: pd.DataFrame,
    judgments: pd.DataFrame,
    *,
    evaluation_model: str,
    judge: str,
    judge_temperature: float | None = None,
    judge_max_out_tokens: int | None = None,
) -> pd.DataFrame:
    """Build scoring rows in the original comparison frame, including repeats."""
    if not comparisons.index.is_unique:
        raise ValueError("Comparisons require unique row IDs.")
    ids = judgments["instruction_index"].tolist()
    frame = comparisons.loc[ids].reset_index(drop=True)
    frame["instruction_index"] = ids
    frame["pref"] = preference_values(judgments["pref"]).to_numpy()
    frame["pref_hard"] = hard_preferences(frame["pref"])
    frame["orientation"] = judgments["orientation"].to_numpy()
    frame["evaluation_model"] = evaluation_model
    frame["source"] = "llm-judge"
    frame["judge"] = judge
    frame["judge_model"] = judge
    frame["judge_prompt_preset"] = judgments["prompt_preset"].to_numpy()
    frame["judge_temperature"] = judge_temperature
    frame["judge_max_out_tokens"] = judge_max_out_tokens
    # Legacy metric/grouping aliases are derived centrally from the focal model.
    focal = focal_battles(frame)
    for column in ("model", "baseline", "completion_model", "completion_baseline"):
        frame[column] = focal[column]
    return frame


@dataclass(frozen=True)
class Battle:
    """One pairwise outcome. ``source`` records its provenance."""

    model_a: str
    model_b: str
    winner: str | None  # persisted label: model_a, model_b, tie, or missing
    source: str  # "llm-judge" | "human"
    pref: float | None = (
        None  # soft preference (0=A wins, 1=B wins); None if unparseable
    )
    question_id: str | None = None  # join key back to cache / transcripts
    judge_model: str | None = None  # llm-judge battles only

    @classmethod
    def from_dict(cls, d: dict) -> Battle:
        """Build from a dict, ignoring unknown keys (forward-compatible)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def winner_labels(preferences: pd.Series) -> pd.Series:
    """Derive legacy winner labels for exported artifacts."""
    return hard_preferences(preferences).map(
        {0.0: "model_a", 0.5: "tie", 1.0: "model_b"}
    )


def write_battles(path: str | Path, battles: pd.DataFrame) -> None:
    """Write the battle frame to Parquet, including legacy winner labels."""
    if "winner" not in battles:
        battles = battles.assign(winner=winner_labels(battles["pref"]))
    battles.to_parquet(path, index=False)


def read_battles(path: str | Path) -> list[Battle]:
    """Read a battles Parquet file into typed Battle objects.

    Missing values (e.g. ``judge_model`` on human rows, an unparsed ``pref``)
    are normalised from NaN to None so the typed view matches the declared
    ``str | None`` / ``float | None`` schema regardless of parquet null handling.
    """
    df = pd.read_parquet(path)
    df = df.astype(object).where(pd.notna(df), None)
    return [Battle.from_dict(r) for r in df.to_dict("records")]


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
class Leaderboard:
    """Per-model ratings (mean + bootstrap CI) plus the run metadata that produced them.

    Named distinctly from :class:`judgearena.reports.EloReport`,
    which is the console/``results-*.json`` run report; this is the narrower
    ``elo_ratings.json`` leaderboard artifact with per-model CIs.
    """

    arena: str
    model: str
    judge_model: str
    n_bootstraps: int
    seed: int
    ratings: list[RatingEntry]

    def write(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(to_jsonable(asdict(self)), f, indent=2, allow_nan=False)


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
