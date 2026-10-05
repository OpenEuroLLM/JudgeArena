"""Frozen-anchor leaderboard boundaries, scoring, and file assembly."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from judgearena.artifacts import to_jsonable
from judgearena.benchmarks.elo.rating import fit_against_frozen_ratings
from judgearena.config import RunConfig, load_config
from judgearena.log import get_logger

logger = get_logger(__name__)


class RatingSummary(BaseModel):
    """One displayed rating and its conditional bootstrap interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rating: float = Field(allow_inf_nan=False)
    ci_low: float | None = Field(default=None, allow_inf_nan=False)
    ci_high: float | None = Field(default=None, allow_inf_nan=False)
    n_battles: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_interval(self) -> RatingSummary:
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError(
                "ci_low and ci_high must either both be set or both be omitted"
            )
        if self.ci_low is not None and self.ci_low > self.ci_high:
            raise ValueError("ci_low must not exceed ci_high")
        return self


class AnchorSet(BaseModel):
    """Persisted frozen ratings and their protocol identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    name: str
    version: str = Field(default="0.01", min_length=1)
    min_anchor_battles: int = Field(default=1, ge=1)
    dataset_sources: dict[str, dict[str, object]] = Field(default_factory=dict)
    task: str
    arena: str
    baseline_model: str
    protocol_id: str
    languages: tuple[str, ...]
    ratings_by_language: dict[str, dict[str, float]]
    counts_by_language: dict[str, dict[str, int]]
    human_battles_by_language: dict[str, int]
    battles_per_language: int = Field(gt=0)
    bootstrap_seed: int = 0

    @model_validator(mode="after")
    def _validate_anchor_set(self) -> AnchorSet:
        if not self.languages or len(set(self.languages)) != len(self.languages):
            raise ValueError("languages must be non-empty and unique")
        languages = set(self.languages)
        for values in (
            self.ratings_by_language,
            self.counts_by_language,
            self.human_battles_by_language,
        ):
            if set(values) != languages:
                raise ValueError("anchor data must contain the configured languages")
        models = set(self.ratings_by_language[self.languages[0]])
        for language in self.languages:
            ratings = self.ratings_by_language[language]
            if (
                set(ratings) != models
                or set(self.counts_by_language[language]) != models
            ):
                raise ValueError("every language must contain the same anchor models")
            if not math.isclose(ratings.get(self.baseline_model, math.nan), 1000.0):
                raise ValueError("the baseline model must be rated 1000")
        return self

    @property
    def overall_ratings(self) -> dict[str, float]:
        """Return weighted multilingual ratings for all anchors."""
        models = self.ratings_by_language[self.languages[0]]
        return {
            model: sum(
                self.ratings_by_language[language][model] / len(self.languages)
                for language in self.languages
            )
            for model in models
        }

    @classmethod
    def load(cls, path: str | Path) -> AnchorSet:
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        _atomic_json(path, self.model_dump(mode="json"))
        return path


class LeaderboardEntry(BaseModel):
    """One immutable anchor or submitted-model leaderboard row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    protocol_id: str
    model: str
    source: Literal["anchor", "submission"]
    overall: RatingSummary
    by_language: dict[str, RatingSummary]

    @model_validator(mode="after")
    def _validate_entry(self) -> LeaderboardEntry:
        if not self.protocol_id or not self.model:
            raise ValueError("protocol_id and model must be non-empty")
        if not self.by_language:
            raise ValueError("by_language must be non-empty")
        return self


def comparable_config(cfg: RunConfig) -> dict[str, object]:
    """Return the frozen config after removing submission-only settings."""
    data = cfg.model_dump(
        mode="json",
        exclude={
            "run": True,
            "model": {"name"},
            "elo": {"leaderboard_dir"},
        },
    )
    if cfg.judge.prompt is not None:
        prompt = data["judge"]["prompt"]
        prompt["system_file"] = cfg.judge.prompt.system_file.read_text()
        prompt["user_file"] = cfg.judge.prompt.user_file.read_text()
    return data


def protocol_identifier(
    anchors: AnchorSet,
    panel: pd.DataFrame,
    config: dict[str, object],
) -> str:
    """Identify the complete frozen config, anchor, and panel artifacts."""
    payload = {
        "config": config,
        "anchors": anchors.model_dump(mode="json", exclude={"protocol_id"}),
        "panel": panel.to_dict(orient="records"),
    }
    encoded = json.dumps(
        payload,
        default=to_jsonable,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_frozen_files(
    directory: str | Path,
) -> tuple[AnchorSet, pd.DataFrame, RunConfig]:
    """Load and validate the immutable config, anchor, and panel files."""
    directory = Path(directory)
    anchors = AnchorSet.load(directory / "anchors.json")
    panel = pd.read_parquet(directory / "panel.parquet")
    required = {
        "panel_id",
        "question_id",
        "lang",
        "instruction",
        "opponent_model",
        "opponent_completion",
        "candidate_position",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Frozen panel is missing columns: {missing}.")
    if panel.empty or not panel["panel_id"].is_unique:
        raise ValueError("Frozen panel IDs must be non-empty and unique.")
    if not set(panel["candidate_position"]) <= {"A", "B"}:
        raise ValueError("Frozen candidate positions must be 'A' or 'B'.")
    if set(panel["lang"]) != set(anchors.languages):
        raise ValueError("Frozen panel languages do not match the anchor set.")
    known = anchors.ratings_by_language[anchors.languages[0]]
    if not panel["opponent_model"].isin(known).all():
        raise ValueError("Frozen panel contains an opponent without an anchor rating.")
    frozen_config = load_config(directory / "config.yaml")
    expected_id = protocol_identifier(anchors, panel, comparable_config(frozen_config))
    if expected_id != anchors.protocol_id:
        raise ValueError("Frozen leaderboard files do not match their protocol ID.")
    return anchors, panel.reset_index(drop=True), frozen_config


def collapse_swapped_rows(battles: pd.DataFrame, swap_mode: str) -> pd.DataFrame:
    """Skip unparseable comparisons and average complete answer-order pairs."""
    preference_columns = ["pref"]
    if "pref_hard" in battles:
        preference_columns.append("pref_hard")
    columns = ["panel_id", "lang", "model_a", "model_b", *preference_columns]
    missing = set(columns) - set(battles.columns)
    if missing:
        raise ValueError(f"battle rows are missing columns: {sorted(missing)}")
    battles = battles.copy()
    for column in preference_columns:
        values = pd.to_numeric(battles[column], errors="coerce")
        if not values.dropna().between(0, 1).all():
            raise ValueError(f"{column} values must be finite and between zero and one")
        battles[column] = values
    if swap_mode == "both":
        if "orientation" not in battles:
            raise ValueError("battle rows are missing columns: ['orientation']")
        grouped = battles.groupby("panel_id", sort=False)
        invalid_counts = grouped.size().ne(2)
        if invalid_counts.any():
            panel_id = invalid_counts.index[invalid_counts][0]
            count = int(grouped.size().loc[panel_id])
            raise ValueError(f"panel_id {panel_id!r} has {count} rows; expected 2")
        orientations = grouped["orientation"].agg(set)
        if not orientations.map(lambda value: value == {"direct", "reversed"}).all():
            raise ValueError("swap rows have invalid orientations")
        identity = grouped[["lang", "model_a", "model_b"]].nunique()
        if identity.gt(1).any(axis=None):
            raise ValueError("swap rows disagree about their battle")
    elif battles["panel_id"].duplicated().any():
        raise ValueError("frozen panel rows must be unique")

    failed_ids = battles.loc[
        battles[preference_columns].isna().any(axis=1), "panel_id"
    ].unique()
    if len(failed_ids):
        logger.warning(
            "Skipping %d/%d panel comparisons with unparseable judge scores.",
            len(failed_ids),
            battles["panel_id"].nunique(),
        )
        battles = battles.loc[~battles["panel_id"].isin(failed_ids)]
    if swap_mode != "both":
        return battles.loc[:, columns].reset_index(drop=True)

    aggregations = {
        "lang": "first",
        "model_a": "first",
        "model_b": "first",
        **{column: "mean" for column in preference_columns},
    }
    return (
        battles.groupby("panel_id", sort=False)
        .agg(aggregations)
        .reset_index()
        .loc[:, columns]
    )


def _interval(samples: list[float]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def score_frozen_submission(
    battles: pd.DataFrame,
    candidate: str,
    anchors: AnchorSet,
    *,
    soft_elo: bool,
    n_bootstraps: int,
) -> LeaderboardEntry:
    """Fit one candidate independently on every frozen language scale."""
    missing = {"panel_id", "lang"} - set(battles.columns)
    if missing:
        raise ValueError(f"frozen battles are missing columns: {sorted(missing)}")
    if battles["panel_id"].duplicated().any():
        raise ValueError("frozen scoring requires one row per panel_id")
    pref_col = "pref" if soft_elo else "pref_hard"
    if not soft_elo and "pref_hard" not in battles:
        raise ValueError("hard frozen Elo requires a pref_hard column")
    observed = set(battles["lang"])
    required = set(anchors.languages)
    if observed != required:
        raise ValueError(
            f"battle languages must be exactly {sorted(required)}; got {sorted(observed)}"
        )

    ratings: dict[str, float] = {}
    counts: dict[str, int] = {}
    by_language_samples = {language: [] for language in anchors.languages}
    rng = np.random.default_rng(anchors.bootstrap_seed)

    language_frames = {
        language: battles[battles["lang"] == language]
        .sort_values("panel_id", kind="stable")
        .reset_index(drop=True)
        for language in anchors.languages
    }
    for language, frame in language_frames.items():
        if len(frame) > anchors.battles_per_language:
            raise ValueError(
                f"language {language!r} has {len(frame)} panel rows; "
                f"expected at most {anchors.battles_per_language}"
            )
        rating = fit_against_frozen_ratings(
            frame,
            candidate,
            anchors.ratings_by_language[language],
            pref_col=pref_col,
        )
        ratings[language] = rating
        counts[language] = len(frame)

    overall_samples: list[float] = []
    for _ in range(n_bootstraps):
        sample_ratings: dict[str, float] = {}
        for language, frame in language_frames.items():
            sampled = frame.iloc[rng.integers(0, len(frame), len(frame))].copy()
            rating = fit_against_frozen_ratings(
                sampled,
                candidate,
                anchors.ratings_by_language[language],
                pref_col=pref_col,
            )
            sample_ratings[language] = rating
            by_language_samples[language].append(rating)
        overall_samples.append(float(np.mean(list(sample_ratings.values()))))

    by_language = {}
    for language in anchors.languages:
        ci_low, ci_high = _interval(by_language_samples[language])
        by_language[language] = RatingSummary(
            rating=ratings[language],
            ci_low=ci_low,
            ci_high=ci_high,
            n_battles=counts[language],
        )
    overall_rating = float(np.mean(list(ratings.values())))
    ci_low, ci_high = _interval(overall_samples)
    return LeaderboardEntry(
        protocol_id=anchors.protocol_id,
        model=candidate,
        source="submission",
        overall=RatingSummary(
            rating=overall_rating,
            ci_low=ci_low,
            ci_high=ci_high,
            n_battles=sum(counts.values()),
        ),
        by_language=by_language,
    )


def _anchor_entries(anchors: AnchorSet) -> list[LeaderboardEntry]:
    entries = []
    for model, overall_rating in anchors.overall_ratings.items():
        by_language = {
            language: RatingSummary(
                rating=anchors.ratings_by_language[language][model],
                n_battles=anchors.counts_by_language[language][model],
            )
            for language in anchors.languages
        }
        entries.append(
            LeaderboardEntry(
                protocol_id=anchors.protocol_id,
                model=model,
                source="anchor",
                overall=RatingSummary(
                    rating=overall_rating,
                    n_battles=sum(
                        summary.n_battles for summary in by_language.values()
                    ),
                ),
                by_language=by_language,
            )
        )
    return entries


def build_leaderboard(
    anchors: AnchorSet, submissions: list[LeaderboardEntry]
) -> dict[str, object]:
    """Validate, merge, and sort immutable anchor and submission rows."""
    anchor_entries = _anchor_entries(anchors)
    known_models = {entry.model for entry in anchor_entries}
    for entry in submissions:
        if entry.source != "submission":
            raise ValueError("entry files must contain submissions")
        if entry.protocol_id != anchors.protocol_id:
            raise ValueError(f"protocol mismatch for model {entry.model!r}")
        if set(entry.by_language) != set(anchors.languages):
            raise ValueError(f"language mismatch for model {entry.model!r}")
        summaries = list(entry.by_language.values())
        if any(
            not 0 < summary.n_battles <= anchors.battles_per_language
            for summary in summaries
        ):
            raise ValueError(f"incorrect battle count for model {entry.model!r}")
        if entry.overall.n_battles != sum(summary.n_battles for summary in summaries):
            raise ValueError(f"incorrect total battle count for model {entry.model!r}")
        expected_rating = float(np.mean([summary.rating for summary in summaries]))
        if not math.isclose(entry.overall.rating, expected_rating, abs_tol=1e-9):
            raise ValueError(f"incorrect overall rating for model {entry.model!r}")
        for summary in [*summaries, entry.overall]:
            if not 0 <= summary.rating <= 2000:
                raise ValueError(f"rating outside bounds for model {entry.model!r}")
            if summary.ci_low is not None and not (
                0 <= summary.ci_low <= summary.ci_high <= 2000
            ):
                raise ValueError(f"interval outside bounds for model {entry.model!r}")
        if entry.model in known_models:
            raise ValueError(f"duplicate leaderboard model: {entry.model!r}")
        known_models.add(entry.model)
    entries = sorted(
        [*anchor_entries, *submissions],
        key=lambda entry: (-entry.overall.rating, entry.model),
    )
    return {
        "schema_version": 1,
        "protocol_id": anchors.protocol_id,
        "name": anchors.name,
        "version": anchors.version,
        "entries": [entry.model_dump(mode="json") for entry in entries],
    }


def entry_filename(model: str) -> str:
    """Return the stable entry filename for a model."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "model"
    digest = hashlib.sha256(model.encode()).hexdigest()[:8]
    return f"{slug[:80]}-{digest}.json"


def write_entry(directory: str | Path, entry: LeaderboardEntry) -> Path:
    """Write one submission without replacing an existing entry."""
    if entry.source != "submission":
        raise ValueError("only submission entries can be written")
    entries = Path(directory) / "entries"
    entries.mkdir(parents=True, exist_ok=True)
    path = entries / entry_filename(entry.model)
    with path.open("x") as stream:
        json.dump(
            entry.model_dump(mode="json"),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return path


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def rebuild_leaderboard(directory: str | Path, anchors: AnchorSet) -> Path:
    """Validate all stored entries and atomically replace the derived index."""
    directory = Path(directory)
    entries_dir = directory / "entries"
    submissions = [
        LeaderboardEntry.model_validate_json(path.read_text())
        for path in sorted(entries_dir.glob("*.json"))
    ]
    leaderboard = build_leaderboard(anchors, submissions)
    path = directory / "leaderboard.json"
    _atomic_json(path, leaderboard)
    return path
