"""Validated dataset capabilities shared by evaluation pipelines."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Self

import pandas as pd

from judgearena.battles import (
    hard_preferences,
    preference_values,
)


def validate_instruction_ids(ids: pd.Series, known: pd.Index, context: str) -> None:
    """Require references to existing instruction IDs without coercing their types."""
    unknown = ids.loc[~ids.isin(known)]
    if not unknown.empty:
        examples = [
            f"{value!r} ({type(value).__name__})" for value in unknown.head(3).tolist()
        ]
        expected = [
            f"{value!r} ({type(value).__name__})" for value in known[:3].tolist()
        ]
        raise ValueError(
            f"{context} reference unknown instruction IDs: {', '.join(examples)}. "
            f"Known ID examples: {', '.join(expected)}. ID values and types must match."
        )


@dataclass(frozen=True)
class CompletionData:
    """Adapter output contract for instructions and optional model completions.

    Attributes:
        instructions: DataFrame with an ``instruction`` column containing prompt
            text and a unique, nonmissing row index identifying each instruction.
            Use stable IDs (e.g. strings or integers), with matching types in
            output rows. Extra columns such as ``lang`` or ``category`` are
            preserved and can be used by ``select()`` for metadata filtering.
        model_outputs: Optional DataFrame with ``instruction_index`` (an ID from
            the instructions index), ``model`` (a nonmissing model name), and
            ``completion`` (answer text). It may cover only some models or rows;
            unavailable completions are left for the runner to generate. Extra
            output columns are preserved. Defaults to ``None``.
        model_output_loader: Optional callable accepting a model name and returning
            a DataFrame with the same output columns, or ``None`` if unavailable.
            Called when no eager outputs exist for that model; it does not fill
            gaps in a partially populated model. Returned rows are filtered to
            the requested model and current instruction IDs, so a loader may
            return answers for the full dataset after ``select()``.

    Construction validates column names, instruction IDs, eager output references,
    and model identities; it does not coerce text values or enforce text dtypes.
    Column names must be unique. Missing output text and repeated output rows are
    accepted. Answer lookups preserve null text and require at most one saved
    distinct row per requested instruction/model. Identical repeated rows are reused;
    adapters or callers select among conflicting answers.

    Adapters normalize source fields before construction. ``from_frames()``
    accepts canonical columns, sets ``instruction_index`` as the index, and sorts
    instructions by ID. Other columns remain metadata.

    Pipelines check the returned data for their additional requirements. For
    example, conversation generation needs ``user_turns``: a nonempty list of
    user prompt strings per instruction. Saved conversation outputs carry
    ``messages``: a list of role/content dictionaries alternating user and
    assistant, optionally preceded by a system message, ending in an assistant
    answer. ``completion`` provides a text view for text-only consumers. Metadata and
    each model's complete history are preserved. These optional fields are
    validated on construction and when lazy outputs are loaded. Existing comparisons
    use ``PreferenceData`` with optional reference labels.
    """

    instructions: pd.DataFrame
    model_outputs: pd.DataFrame | None = None
    model_output_loader: Callable[[str], pd.DataFrame | None] | None = field(
        default=None, kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _require_columns(self.instructions, {"instruction"}, "Instructions")
        if (
            self.instructions.index.has_duplicates
            or self.instructions.index.isna().any()
        ):
            raise ValueError("Instructions require unique, nonmissing row IDs.")
        if "user_turns" in self.instructions:
            for turns in self.instructions["user_turns"]:
                validate_user_turns(turns)
        if "messages" in self.instructions:
            for messages in self.instructions["messages"].dropna():
                validate_messages(messages, ending="user")
        if self.model_outputs is None:
            return
        _validate_model_outputs(self.model_outputs)
        validate_instruction_ids(
            self.model_outputs["instruction_index"],
            self.instructions.index,
            "Model outputs",
        )

    @classmethod
    def from_frames(
        cls,
        instructions: pd.DataFrame,
        model_outputs: pd.DataFrame | None = None,
        *,
        model_output_loader: Callable[[str], pd.DataFrame | None] | None = None,
    ) -> CompletionData:
        """Build from canonical frames; source normalization belongs to the adapter."""
        instructions = instructions.copy()
        if "instruction_index" in instructions:
            instructions = instructions.set_index("instruction_index")
        elif instructions.index.name != "instruction_index":
            raise ValueError("Instructions must provide 'instruction_index'.")
        return cls(
            instructions.sort_index(),
            None if model_outputs is None else model_outputs.copy(),
            model_output_loader=model_output_loader,
        )

    def select(self, filters: Mapping[str, Collection[object]]) -> Self:
        """Select metadata values, keeping instructions, outputs and battles aligned."""
        if not filters:
            return self
        frame = self.battles if isinstance(self, PreferenceData) else self.instructions
        if isinstance(self, PreferenceData):
            instruction_columns = [
                column
                for column in filters
                if column not in frame and column in self.instructions
            ]
            frame = frame.join(
                self.instructions[instruction_columns], on="instruction_index"
            )
        _require_columns(frame, set(filters), "Dataset selection")
        selected = pd.Series(True, index=frame.index)
        for column, values in filters.items():
            selected &= frame[column].isin(values)
        frame = frame.loc[selected].copy()
        if frame.empty:
            raise ValueError(f"No dataset rows match filters: {dict(filters)}.")
        ids = (
            frame["instruction_index"]
            if isinstance(self, PreferenceData)
            else frame.index
        )
        updates = {
            "instructions": self.instructions.loc[
                self.instructions.index.isin(ids)
            ].copy(),
            "model_outputs": (
                self.model_outputs.loc[
                    self.model_outputs["instruction_index"].isin(ids)
                ].copy()
                if self.model_outputs is not None
                else None
            ),
        }
        if isinstance(self, PreferenceData):
            updates["battles"] = self.battles.loc[selected].copy()
        return replace(self, **updates)

    def model_outputs_for(self, model: str) -> pd.DataFrame | None:
        """Load a model's outputs through its adapter, retaining output metadata."""
        outputs = self.model_outputs
        if outputs is not None:
            outputs = outputs.loc[outputs["model"] == model]
        if (outputs is None or outputs.empty) and self.model_output_loader is not None:
            outputs = self.model_output_loader(model)
            if outputs is not None:
                _validate_model_outputs(outputs)
                outputs = outputs.loc[outputs["model"] == model]
        if outputs is None or outputs.empty:
            return None
        return outputs.loc[
            outputs["instruction_index"].isin(self.instructions.index)
        ].copy()

    def model_output_rows(
        self,
        model: str,
        *,
        instruction_ids: pd.Index | None = None,
    ) -> pd.DataFrame | None:
        """Return complete answer rows aligned to inputs; absent rows remain missing."""
        outputs = self.model_outputs_for(model)
        if outputs is None:
            return None
        ids = self.instructions.index if instruction_ids is None else instruction_ids
        outputs = outputs.loc[outputs["instruction_index"].isin(ids)]
        outputs = outputs.set_index("instruction_index")
        unique = outputs.loc[~outputs.index.duplicated()]
        if not unique.reindex(outputs.index).equals(outputs):
            raise ValueError(
                f"Model {model!r} has multiple saved answers for a requested "
                "instruction; select one answer per instruction before lookup."
            )
        return unique.reindex(ids)

    def model_completion(
        self,
        model: str,
        *,
        instruction_ids: pd.Index | None = None,
    ) -> pd.Series | None:
        """Return aligned text completions for text-only consumers."""
        outputs = self.model_output_rows(model, instruction_ids=instruction_ids)
        return None if outputs is None else outputs["completion"]


def validate_user_turns(turns: object) -> None:
    """Validate the planned user prompts needed for sequential generation."""
    if (
        not isinstance(turns, list)
        or not turns
        or any(not isinstance(turn, str) for turn in turns)
    ):
        raise ValueError(
            "Generation requires a nonempty list of text user_turns per row."
        )


def validate_messages(
    messages: object, *, ending: Literal["user", "assistant"] = "assistant"
) -> None:
    """Validate text chat history for generation or a completed answer."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Conversation messages must be a nonempty list.")
    expected = "user"
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Conversation messages require role and text content.")
        role = message.get("role")
        if i == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError(f"Conversation expected {expected!r}, got {role!r}.")
        expected = "assistant" if role == "user" else "user"
    if messages[-1]["role"] != ending:
        raise ValueError(f"Conversation must end with a {ending} message.")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{label} has duplicate column names after mapping.")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}.")


def _validate_model_outputs(outputs: pd.DataFrame) -> None:
    _require_columns(
        outputs, {"instruction_index", "model", "completion"}, "Model outputs"
    )
    if "messages" in outputs:
        for messages in outputs["messages"].dropna():
            validate_messages(messages)
    if outputs["model"].isna().any():
        raise ValueError("Model outputs require model identities.")


@dataclass(frozen=True)
class PreferenceData(CompletionData):
    """Completion data with existing comparisons and optional reference preferences.

    ``battles`` is a DataFrame with these canonical columns:

    - ``battle_id``: unique, nonmissing comparison ID. When omitted, row positions
      become string IDs; adapters should supply IDs stable across source reloads.
    - ``instruction_index``: an ID present in ``instructions.index``.
    - ``instruction``, ``completion_a``, ``completion_b``: prompt and answer text.
    - ``model_a``, ``model_b``: nonmissing identities for the two answer slots.
    - ``reference_pref`` (optional): numeric preference in [0, 1]; 0 favors A,
      0.5 is a tie, and 1 favors B. Intermediate values express soft preferences. Missing outcomes
      are allowed; values are normalized to floats.
    - ``source`` (optional): provenance metadata; it does not determine whether
      a comparison can be judged or used as a reference.

    Newly judged preferences use ``pref`` in scoring results. Reference labels
    remain attached to their original answer pairs.

    The ``CompletionData`` contract also applies. Extra battle columns remain
    metadata. Pipelines validate text content. Use ``from_battles()`` to derive
    instruction and output views together.
    """

    battles: pd.DataFrame = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_columns(
            self.battles,
            {
                "instruction_index",
                "instruction",
                "completion_a",
                "completion_b",
                "model_a",
                "model_b",
            },
            "Battles",
        )
        validate_instruction_ids(
            self.battles["instruction_index"], self.instructions.index, "Battles"
        )
        if self.battles[["model_a", "model_b"]].isna().any().any():
            raise ValueError("Battles require model identities.")
        battles = self.battles.copy()
        if "battle_id" not in battles:
            battles["battle_id"] = pd.RangeIndex(len(battles)).astype(str)
        ids = battles["battle_id"]
        if ids.isna().any() or ids.astype(str).duplicated().any():
            raise ValueError("Comparisons require unique, nonmissing battle IDs.")
        if "reference_pref" in battles:
            battles["reference_pref"] = preference_values(battles["reference_pref"])
        object.__setattr__(self, "battles", battles)

    @classmethod
    def from_battles(cls, frame: pd.DataFrame) -> PreferenceData:
        """Derive aligned views from canonical battles, retaining slot metadata.

        Columns ending in _a/_b belong to that candidate. Other columns describe
        the comparison. Missing instruction_index values are assigned by row.
        Shared IDs require matching instruction text and generation inputs.
        Metadata consistent across each instruction's comparisons is also exposed
        in instructions; varying metadata stays in battles. All comparison and
        output rows are retained.
        """
        battles = frame.reset_index(drop=True).copy()
        _require_columns(
            battles,
            {
                "instruction",
                "model_a",
                "model_b",
                "completion_a",
                "completion_b",
            },
            "Battles",
        )
        if "instruction_index" not in battles:
            battles["instruction_index"] = range(len(battles))
        indexed = battles.set_index("instruction_index", drop=False)
        slot_columns = [c for c in battles if c.endswith(("_a", "_b"))]
        instructions = indexed.drop(
            columns=[
                "instruction_index",
                "battle_id",
                "reference_pref",
                "pref",
                "source",
                "winner",
                *slot_columns,
            ],
            errors="ignore",
        )
        unique_instructions = instructions.loc[~instructions.index.duplicated()]
        aligned = unique_instructions.reindex(instructions.index)
        varying = [c for c in instructions if not aligned[c].equals(instructions[c])]
        if {"instruction", "instruction_id", "user_turns", "messages"}.intersection(
            varying
        ):
            raise ValueError(
                "Shared instruction IDs have conflicting instruction inputs."
            )
        instructions = unique_instructions.drop(columns=varying)
        outputs = pd.concat(
            [
                battles[
                    [
                        "instruction_index",
                        *[c for c in slot_columns if c.endswith(f"_{slot}")],
                    ]
                ].rename(
                    columns={c: c[:-2] for c in slot_columns if c.endswith(f"_{slot}")}
                )
                for slot in ("a", "b")
            ],
            ignore_index=True,
        ).dropna(subset=["completion"])
        return cls(instructions, outputs, battles=battles)


def normalize_preferences(
    frame: pd.DataFrame, outcome_preferences: dict[str, float] | None = None
) -> pd.DataFrame:
    """Normalize source labels in an adapter before constructing PreferenceData."""
    battles = frame.copy()
    if "winner" in battles and ("reference_pref" not in battles or outcome_preferences):
        if not outcome_preferences:
            raise ValueError(
                "Source winner labels require an explicit outcome_preferences mapping."
            )
        unknown = (
            battles["winner"]
            .dropna()
            .loc[lambda values: ~values.isin(outcome_preferences)]
        )
        if not unknown.empty:
            raise ValueError(f"Unknown battle outcomes: {unknown.unique().tolist()}")
        mapped = preference_values(battles["winner"].map(outcome_preferences))
        if "reference_pref" not in battles:
            battles["reference_pref"] = mapped
        else:
            observed = hard_preferences(mapped)
            supplied = hard_preferences(battles["reference_pref"])
            consistent = observed.eq(supplied) | (observed.isna() & supplied.isna())
            if not consistent.all():
                raise ValueError(
                    "Source winner labels conflict with battle preferences."
                )
    return battles
