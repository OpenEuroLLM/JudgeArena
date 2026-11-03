"""Dataset adapter for human preference battles used by arena-backed tasks."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.arenas_utils import (
    KNOWN_ARENAS,
    extract_turn_text,
    load_arena_dataframe,
)
from judgearena.datasets.data import PreferenceData, normalize_preferences
from judgearena.datasets.registry import DatasetAdapter, register_dataset
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


@register_dataset("arena_battles")
class ArenaBattlesAdapter(DatasetAdapter):
    def _task_sources(self) -> tuple[str, dict[str, HuggingFaceDatasetSource]]:
        arena = self.task.spec.dataset.format or getattr(
            self.task.spec.protocol, "arena", None
        )
        if arena not in {*KNOWN_ARENAS, "LMArena"}:
            raise ValueError(f"Unsupported arena format {arena!r}; set dataset.format.")

        if not self.task.spec.dataset.sources:
            raise ValueError("Arena adapter requires dataset.sources.")
        sources: dict[str, HuggingFaceDatasetSource] = {}
        for source in self.task.spec.dataset.sources.values():
            if not isinstance(source, HuggingFaceDatasetSource):
                raise ValueError("Arena sources must be Hugging Face datasets.")
            if source.repo_id in sources:
                raise ValueError(f"Duplicate arena source {source.repo_id!r}.")
            sources[source.repo_id] = source
        return arena, sources

    def download(self) -> None:
        """Download every pinned human-battle source declared by the task."""
        _, sources = self._task_sources()
        for source in sources.values():
            snapshot_download(
                repo_id=source.repo_id,
                repo_type="dataset",
                revision=source.revision,
                allow_patterns=source.allow_patterns or None,
                force_download=False,
            )

    def load_battles(self) -> pd.DataFrame:
        """Load and normalize the task's pinned human preference battles."""
        arena, sources = self._task_sources()
        return load_arena_dataframe(arena, dataset_sources=sources)

    def load_data(self) -> PreferenceData:
        arena, _ = self._task_sources()
        return normalize_battles(
            self.load_battles(),
            arena,
            outcome_preferences=self.task.spec.dataset.outcome_preferences,
        )
