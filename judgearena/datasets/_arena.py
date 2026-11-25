"""Source downloads and single-turn preference rows shared by the arena adapters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd
from fast_langdetect import detect_language
from huggingface_hub import snapshot_download

from judgearena.datasets.data import PreferenceData, normalize_preferences
from judgearena.log import get_logger
from judgearena.tasks.schema import (
    HuggingFaceDatasetSource,
)

# Labels emitted by the supported arena source loaders.
_ARENA_OUTCOME_PREFERENCES = {
    "model_a": 0.0,
    "model_b": 1.0,
    "tie": 0.5,
    "tie (bothbad)": 0.5,
}


logger = get_logger(__name__)


def _download_arena_dataset(
    *,
    repo_id: str,
    default_allow_patterns: str | tuple[str, ...] | None,
    dataset_sources: Mapping[str, HuggingFaceDatasetSource],
) -> str:
    """Download one arena source at the revision pinned by its task definition."""
    try:
        source = dataset_sources[repo_id]
    except KeyError as exc:
        raise ValueError(
            f"Arena task does not declare required dataset source {repo_id!r}."
        ) from exc
    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=source.allow_patterns or default_allow_patterns,
        force_download=False,
        revision=source.revision,
    )


def extract_turn_text(turn: dict) -> str:
    """Extract plain text from a conversation turn.

    Handles both the 100k schema (content is a plain string) and the 140k
    schema (content is an array of {type, text, ...} objects). Moderated or
    empty turns ship ``content: None`` and yield an empty string.
    """
    content = turn.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text") or ""
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def single_turn_battles(df: pd.DataFrame, arena: str) -> pd.DataFrame:
    def user_text(conversation):
        if not isinstance(conversation, (list, tuple, np.ndarray)):
            return ""
        try:
            return next(
                (
                    extract_turn_text(turn)
                    for turn in conversation
                    if isinstance(turn, dict) and turn.get("role") == "user"
                ),
                "",
            )
        except (TypeError, ValueError):
            return ""  # The adapter counts and drops malformed source rows.

    df["lang"] = (
        df["conversation_a"]
        .map(user_text)
        .map(lambda text: detect_language(text).lower() if text else "")
    )

    cols = [
        "question_id",
        "tstamp",
        "model_a",
        "model_b",
        "winner",
        "conversation_a",
        "conversation_b",
        "benchmark",
        "lang",
    ]
    df = df.loc[:, cols]

    # keep only one turn conversation for now as they are easier to evaluate
    df["turns"] = df["conversation_a"].map(
        lambda conversation: (
            len(conversation)
            - int(
                isinstance(conversation[0], dict)
                and conversation[0].get("role") == "system"
            )
            - 1
            if isinstance(conversation, (list, tuple, np.ndarray)) and len(conversation)
            else 0
        )
    )
    n_before = len(df)
    df = df.loc[df.turns <= 1]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "[%s] Dropped %d/%d multi-turn battles (keeping single-turn only).",
            arena,
            n_dropped,
            n_before,
        )

    return df


def _conversation_prefix(conversation) -> list[dict[str, str]]:
    """Normalize the first answer, retaining its optional system context."""
    if not isinstance(conversation, (list, tuple, np.ndarray)) or not len(conversation):
        raise ValueError("missing user/assistant turns")
    system = (
        isinstance(conversation[0], dict) and conversation[0].get("role") == "system"
    )
    prefix = list(conversation[: 3 if system else 2])
    roles = ["system", "user", "assistant"] if system else ["user", "assistant"]
    if len(prefix) != len(roles) or any(
        not isinstance(turn, dict) or turn.get("role") != role
        for turn, role in zip(prefix, roles, strict=True)
    ):
        raise ValueError("invalid user/assistant turns")
    try:
        return [
            {"role": role, "content": extract_turn_text(turn)}
            for turn, role in zip(prefix, roles, strict=True)
        ]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid turn content") from exc


def normalize_battles(
    frame: pd.DataFrame,
    arena: str,
    *,
    outcome_preferences: dict[str, float] | None = None,
) -> PreferenceData:
    """Expose the arena's existing single-turn view to any compatible runner."""
    positions, normalized = [], []
    dropped = Counter()
    for position, (_, battle) in enumerate(frame.iterrows()):
        try:
            a = _conversation_prefix(battle.get("conversation_a"))
            b = _conversation_prefix(battle.get("conversation_b"))
            if a[-2]["content"] != b[-2]["content"]:
                raise ValueError("different user prompts")
        except ValueError as exc:
            dropped[str(exc)] += 1
            continue
        item = {
            "instruction": a[-2]["content"],
            "completion_a": a[-1]["content"],
            "completion_b": b[-1]["content"],
        }
        positions.append(position)
        normalized.append(item)
    if dropped:
        logger.warning(
            "[%s] Dropped %d/%d invalid comparisons: %s",
            arena,
            sum(dropped.values()),
            len(frame),
            dict(dropped),
        )
    if not positions:
        raise ValueError(f"No comparable user/assistant pairs remain in {arena!r}.")
    rows = frame.iloc[positions].reset_index(drop=True).copy()
    for column, values in pd.DataFrame(normalized).items():
        rows[column] = values
    rows["instruction_id"] = arena + ":" + rows["question_id"].astype(str)
    if "battle_id" not in rows:
        ids = rows["instruction_id"]
        rows["battle_id"] = ids.where(
            ~ids.duplicated(keep=False),
            ids + ":row-" + pd.RangeIndex(len(rows)).astype(str),
        )
    rows["source"] = "human"
    return PreferenceData.from_battles(
        normalize_preferences(
            rows,
            outcome_preferences=outcome_preferences or _ARENA_OUTCOME_PREFERENCES,
        )
    )


def arena_sources(task):
    if not task.spec.dataset.sources:
        raise ValueError("Arena adapter requires dataset.sources.")
    sources: dict[str, HuggingFaceDatasetSource] = {}
    for source in task.spec.dataset.sources.values():
        if not isinstance(source, HuggingFaceDatasetSource):
            raise ValueError("Arena sources must be Hugging Face datasets.")
        if source.repo_id in sources:
            raise ValueError(f"Duplicate arena source {source.repo_id!r}.")
        sources[source.repo_id] = source
    return sources


def download_arena_sources(sources):
    for repo_id in sources:
        _download_arena_dataset(
            repo_id=repo_id, default_allow_patterns=None, dataset_sources=sources
        )
