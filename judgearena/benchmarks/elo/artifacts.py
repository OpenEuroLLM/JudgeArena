"""Portable frozen leaderboard artifacts and submission validation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from judgearena.artifacts import safe_filename
from judgearena.benchmarks.elo.leaderboard import (
    AnchorSet,
    LeaderboardEntry,
    build_leaderboard,
    collapse_swapped_rows,
    entry_filename,
    load_frozen_files,
    score_frozen_submission,
)
from judgearena.config import RunConfig

FROZEN_FILES = (
    "anchors.json",
    "panel.parquet",
    "config.yaml",
    "judge-system-prompt.txt",
    "judge-user-prompt.txt",
)
BATTLE_COLUMNS = (
    "model_a",
    "model_b",
    "winner",
    "pref",
    "pref_hard",
    "source",
    "judge_model",
    "question_id",
    "panel_id",
    "lang",
    "orientation",
)


def version_path(anchors: AnchorSet) -> Path:
    """Return the export-relative path without accepting path components."""
    for value in (anchors.name, anchors.version):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError(
                "Leaderboard name and version must be safe path components."
            )
    return Path("versions") / f"{anchors.name}-v{anchors.version}"


def _reject_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if item and (
                normalized
                in {
                    "token",
                    "auth",
                    "authorization",
                    "password",
                    "secret",
                    "credentials",
                    "headers",
                    "default_headers",
                }
                or normalized.endswith(("api_key", "apikey", "_token", "secret_key"))
            ):
                raise ValueError(
                    "Frozen config contains explicit credentials; do not share it."
                )
            _reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _reject_credentials(item)


def load_frozen_artifacts(directory: Path) -> tuple[AnchorSet, pd.DataFrame, RunConfig]:
    """Load portable files, rejecting credentials and external prompt paths."""
    # Check prompt paths before config loading can read files outside the snapshot.
    data = yaml.safe_load((directory / "config.yaml").read_text())
    _reject_credentials(data)
    prompt = data.get("judge", {}).get("prompt") if isinstance(data, dict) else None
    if not isinstance(prompt, dict) or (
        prompt.get("system_file") != "judge-system-prompt.txt"
        or prompt.get("user_file") != "judge-user-prompt.txt"
    ):
        raise ValueError("Frozen config must use the saved relative prompt files.")
    anchors, panel, cfg = load_frozen_files(directory)
    version_path(anchors)
    return anchors, panel, cfg


def load_submission_entries(
    directory: Path, anchors: AnchorSet
) -> list[LeaderboardEntry]:
    """Read and validate saved entries against their frozen references."""
    entries = []
    for path in sorted((directory / "entries").glob("*.json")):
        entry = LeaderboardEntry.model_validate_json(path.read_text())
        if path.name != entry_filename(entry.model):
            raise ValueError(f"Entry filename does not match model: {path.name}")
        entries.append(entry)
    build_leaderboard(anchors, entries)
    return entries


def validate_submission(
    directory: str | Path,
    entry_path: str | Path,
    battles_path: str | Path | io.BytesIO,
) -> LeaderboardEntry:
    """Recompute a proposed entry from its complete canonical battle artifact."""
    directory = Path(directory)
    anchors, panel, cfg = load_frozen_artifacts(directory)
    entry = LeaderboardEntry.model_validate_json(Path(entry_path).read_text())
    build_leaderboard(anchors, [entry])
    battles = pd.read_parquet(battles_path)
    if set(battles.columns) != set(BATTLE_COLUMNS):
        raise ValueError("Battles must contain exactly the canonical battle columns.")
    orientations = (
        {"direct", "reversed"} if cfg.judge.swap_mode == "both" else {"single"}
    )
    expected = {
        (panel_id, order) for panel_id in panel.panel_id for order in orientations
    }
    observed = list(zip(battles.panel_id, battles.orientation, strict=True))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError(
            "Battle panel coverage or orientations do not match the frozen panel."
        )
    indexed = panel.set_index("panel_id").loc[battles.panel_id].reset_index()
    candidate_a = indexed.candidate_position.eq("A")
    expected_a = indexed.opponent_model.where(~candidate_a, entry.model)
    expected_b = indexed.opponent_model.where(candidate_a, entry.model)
    for column, values in {
        "model_a": expected_a,
        "model_b": expected_b,
        "question_id": indexed.question_id,
        "lang": indexed.lang,
        "judge_model": cfg.judge.model,
        "source": "llm-judge",
    }.items():
        if not battles[column].reset_index(drop=True).eq(values).all():
            raise ValueError(f"Battle {column} does not match the frozen panel/config.")
    for column in ("pref", "pref_hard"):
        try:
            values = pd.to_numeric(battles[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} must be numeric or missing.") from exc
        if not (values.isna() | (np.isfinite(values) & values.between(0, 1))).all():
            raise ValueError(
                f"{column} must be finite, between zero and one, or missing."
            )
        battles[column] = values
    expected_hard = (np.sign(battles.pref - 0.5) + 1) / 2
    if not (
        battles.pref_hard.eq(expected_hard)
        | (battles.pref_hard.isna() & expected_hard.isna())
    ).all():
        raise ValueError("pref_hard does not match pref.")
    expected_winner = expected_hard.map({0.0: "model_a", 0.5: "tie", 1.0: "model_b"})
    if not (
        battles.winner.eq(expected_winner)
        | (battles.winner.isna() & expected_winner.isna())
    ).all():
        raise ValueError("winner does not match pref.")
    if cfg.elo is None:
        raise ValueError("A frozen leaderboard requires an Elo configuration.")
    recomputed = score_frozen_submission(
        collapse_swapped_rows(battles, cfg.judge.swap_mode),
        entry.model,
        anchors,
        soft_elo=cfg.elo.soft_elo,
        n_bootstraps=cfg.elo.n_bootstraps,
    )
    for saved, actual in zip(
        [entry.overall, *(entry.by_language[lang] for lang in anchors.languages)],
        [
            recomputed.overall,
            *(recomputed.by_language[lang] for lang in anchors.languages),
        ],
        strict=True,
    ):
        for key, value in saved.model_dump().items():
            expected_value = getattr(actual, key)
            if value is None or expected_value is None or key == "n_battles":
                matches = value == expected_value
            else:
                matches = math.isclose(
                    value, expected_value, rel_tol=1e-10, abs_tol=1e-7
                )
            if not matches:
                raise ValueError(f"Saved entry {key} does not match recomputed score.")
    return recomputed


def export_leaderboard(
    directory: str | Path,
    output: str | Path,
    *,
    results_dir: str | Path | None = None,
) -> Path:
    """Export only frozen assets and validated submissions to a new local tree."""
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(f"Export output already exists: {output}")
    anchors, _, cfg = load_frozen_artifacts(directory)
    entries = load_submission_entries(directory, anchors)
    validated = []
    search_root = (
        Path(results_dir) if results_dir is not None else Path(cfg.run.result_folder)
    )
    for entry in entries:
        filename = entry_filename(entry.model)
        portable = directory / "submissions" / Path(filename).stem / "battles.parquet"
        if results_dir is None:
            model_digest = hashlib.sha256(entry.model.encode()).hexdigest()[:16]
            result_root = (
                search_root
                / (
                    f"elo-{safe_filename(anchors.arena)}-{safe_filename(entry.model)}-"
                    f"{safe_filename(cfg.judge.model)}"
                )
                / anchors.protocol_id
                / f"model-{model_digest}"
            )
        else:
            result_root = search_root
        candidates = (
            [portable]
            if portable.exists()
            else list(result_root.rglob("battles.parquet"))
        )
        matches = []
        for path in candidates:
            try:
                validate_submission(directory, directory / "entries" / filename, path)
            except ValueError:
                continue
            matches.append(path)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one validated battles.parquet for {entry.model!r}; found {len(matches)}. Use --results-dir to select the results."
            )
        validated.append((filename, matches[0]))
    target = output / version_path(anchors)
    target.mkdir(parents=True)
    for name in FROZEN_FILES:
        shutil.copyfile(directory / name, target / name)
    # Runtime paths and logging settings do not form part of the frozen protocol.
    config = yaml.safe_load((target / "config.yaml").read_text())
    config.pop("run", None)
    config["elo"]["leaderboard_dir"] = None
    (target / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (target / "entries").mkdir()
    for filename, battles in validated:
        shutil.copyfile(directory / "entries" / filename, target / "entries" / filename)
        destination = target / "submissions" / Path(filename).stem / "battles.parquet"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(battles, destination)
    (target / "leaderboard.json").write_text(
        json.dumps(build_leaderboard(anchors, []), indent=2, allow_nan=False) + "\n"
    )
    load_frozen_artifacts(target)
    return target
