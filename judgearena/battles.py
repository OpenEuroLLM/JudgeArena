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


def require_agreement(direct: float | None, reversed: float | None) -> float | None:
    """Combine preferences already oriented to the same candidates; missing stays missing."""
    if direct is None or reversed is None:
        return None
    return direct if direct == reversed else 0.5


def agreement_outcomes(annotations: pd.DataFrame) -> pd.DataFrame:
    """Reduce canonical answer-order preferences, requiring both verdicts to agree."""
    outcomes = annotations.loc[annotations["orientation"] != "reversed"].copy()
    outcomes["inconsistent"] = False
    reverse = annotations.loc[annotations["orientation"] == "reversed"]
    if not reverse.empty:
        reverse_prefs = (
            reverse.set_index("instruction_index")
            .loc[outcomes["instruction_index"], "pref"]
            .to_numpy()
        )
        direct_prefs = outcomes["pref"]
        outcomes["inconsistent"] = (
            direct_prefs.notna()
            & pd.notna(reverse_prefs)
            & (direct_prefs != reverse_prefs)
        )
        outcomes["pref"] = pd.Series(
            [
                require_agreement(None if pd.isna(a) else a, None if pd.isna(b) else b)
                for a, b in zip(direct_prefs, reverse_prefs, strict=True)
            ],
            index=outcomes.index,
            dtype=float,
        )
    outcomes["orientation"] = "single"
    return outcomes


def hard_preferences(values: pd.Series) -> pd.Series:
    prefs = preference_values(values)
    return pd.Series(np.sign(prefs - 0.5) / 2 + 0.5, index=prefs.index)


def swap_comparison_slots(
    comparisons: pd.DataFrame, reverse_order: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Swap slots using an aligned boolean Series or a positional boolean array."""
    if isinstance(reverse_order, pd.Series) and not reverse_order.index.equals(
        comparisons.index
    ):
        raise ValueError("Swap mask index must match comparison rows in order.")
    if (
        reverse_order.ndim != 1
        or len(reverse_order) != len(comparisons)
        or not pd.api.types.is_bool_dtype(reverse_order.dtype)
        or pd.isna(reverse_order).any()
    ):
        raise ValueError("Swap mask must contain one nonmissing boolean per row.")
    swapped = comparisons.copy()
    prefixes = dict.fromkeys(c[:-2] for c in comparisons if c.endswith(("_a", "_b")))
    for prefix in prefixes:
        a, b = f"{prefix}_a", f"{prefix}_b"
        left = comparisons.get(
            a, pd.Series(None, index=comparisons.index, dtype=object)
        )
        right = comparisons.get(
            b, pd.Series(None, index=comparisons.index, dtype=object)
        )
        swapped[a] = left.where(~reverse_order, right)
        swapped[b] = right.where(~reverse_order, left)
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
    judgments: pd.DataFrame,
    *,
    evaluation_model: str,
    judge: str,
    judge_temperature: float | None = None,
    judge_max_out_tokens: int | None = None,
) -> pd.DataFrame:
    """Add shared scoring fields to complete outcome rows, including repeats."""
    frame = judgments.reset_index(drop=True).copy()
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
    return frame.assign(
        **pairwise_aliases(frame, model_is_a=frame["model_a"] == evaluation_model)
    )


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
