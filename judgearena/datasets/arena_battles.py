"""Dataset adapter for human preference battles used by arena-backed tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.arenas_utils import (
    KNOWN_ARENAS,
    extract_turn_text,
    load_arena_dataframe,
)
from judgearena.datasets.data import PreferenceData
from judgearena.datasets.registry import DatasetAdapter, register_dataset
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


def _turn_text(turns, position: int) -> str | None:
    # Keep incomplete rows available for filtering; runners validate selected text.
    try:
        return extract_turn_text(turns[position])
    except (IndexError, KeyError, TypeError, AttributeError, ValueError):
        return None


def validate_conversations(df: pd.DataFrame) -> None:
    """Validate selected arena conversations before using their normalized text."""
    for _, battle in df.iterrows():
        battle_id = battle.get("battle_id", battle.get("question_id", "unknown"))
        conversations = []
        for column in ("conversation_a", "conversation_b"):
            conversation = battle.get(column)
            if (
                not isinstance(conversation, (list, tuple, np.ndarray))
                or len(conversation) < 2
            ):
                raise ValueError(
                    f"Battle {battle_id!r} requires user and assistant turns in "
                    f"{column}."
                )
            user_turn, assistant_turn = conversation[:2]
            if (
                not isinstance(user_turn, dict)
                or not isinstance(assistant_turn, dict)
                or user_turn.get("role") != "user"
                or assistant_turn.get("role") != "assistant"
            ):
                raise ValueError(
                    f"Battle {battle_id!r} requires user and assistant turns in "
                    f"{column}."
                )
            try:
                conversations.append(
                    (extract_turn_text(user_turn), extract_turn_text(assistant_turn))
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Battle {battle_id!r} has invalid turn content in {column}."
                ) from exc
        if conversations[0][0] != conversations[1][0]:
            raise ValueError(
                f"Battle {battle_id!r} has different user prompts across conversations."
            )


def normalize_battles(
    frame: pd.DataFrame,
    arena: str,
    *,
    outcome_preferences: dict[str, float] | None = None,
) -> PreferenceData:
    """Expose the arena's existing single-turn view to any compatible runner."""
    rows = frame.copy()
    rows["instruction"] = [_turn_text(turns, 0) for turns in rows["conversation_a"]]
    rows["completion_a"] = [_turn_text(turns, 1) for turns in rows["conversation_a"]]
    rows["completion_b"] = [_turn_text(turns, 1) for turns in rows["conversation_b"]]
    rows["instruction_id"] = arena + ":" + rows["question_id"].astype(str)
    rows["source"] = "human"
    return PreferenceData.from_battles(
        rows,
        outcome_preferences=outcome_preferences or _ARENA_OUTCOME_PREFERENCES,
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
