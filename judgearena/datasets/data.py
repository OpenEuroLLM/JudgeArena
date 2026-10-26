"""Validated dataset capabilities shared by evaluation pipelines."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import Self

import pandas as pd

from judgearena.battles import (
    hard_preferences,
    preference_values,
)
from judgearena.log import get_logger

logger = get_logger(__name__)


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
            ``output`` (completion text). It may cover only some models or rows;
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
    accepted: ``model_completion()`` fills null text with an empty string and
    keeps the last row for each instruction/model pair.

    Prefer ``from_frames()`` in adapters: it maps source column names, makes
    ``instruction_index`` the index, and sorts instructions by ID. For example::

        return CompletionData.from_frames(
            questions,
            answers,  # Optional; omit when the dataset has no saved completions.
            instruction_columns={"question_id": "instruction_index", "prompt": "instruction"},
            output_columns={"question_id": "instruction_index", "answer": "output"},
        )

    Pipelines check the returned data for their additional requirements. For
    example, conversation generation needs ``user_turns``: a nonempty list of
    user prompt strings per instruction. Saved conversation outputs carry
    ``messages``: a list of role/content dictionaries alternating user and
    assistant, optionally preceded by a system message, ending in an assistant
    answer. ``output`` provides a text view for text-only consumers. Metadata and
    each model's complete history are preserved. These optional fields are
    validated on construction and when lazy outputs are loaded. Human preference
    evaluation uses ``PreferenceData`` with observed human labels.
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
        if self.model_outputs is None:
            return
        _validate_model_outputs(self.model_outputs)
        if (
            not self.model_outputs["instruction_index"]
            .isin(self.instructions.index)
            .all()
        ):
            raise ValueError("Model outputs reference unknown instruction IDs.")

    @classmethod
    def from_frames(
        cls,
        instructions: pd.DataFrame,
        model_outputs: pd.DataFrame | None = None,
        *,
        instruction_columns: dict[str, str] | None = None,
        output_columns: dict[str, str] | None = None,
        model_output_loader: Callable[[str], pd.DataFrame | None] | None = None,
    ) -> CompletionData:
        """Copy and normalize adapter frames using source-name -> canonical-name maps.

        After mapping, instructions must have an ``instruction_index`` column or
        an index with that name. Output IDs remain a column; values are not cast.
        Unmapped metadata columns and the optional loader are retained.
        """
        instructions = instructions.rename(columns=instruction_columns or {}).copy()
        if "instruction_index" in instructions:
            instructions = instructions.set_index("instruction_index")
        elif instructions.index.name != "instruction_index":
            raise ValueError("Instructions must provide 'instruction_index'.")
        outputs = (
            None
            if model_outputs is None
            else model_outputs.rename(columns=output_columns or {}).copy()
        )
        return cls(
            instructions.sort_index(), outputs, model_output_loader=model_output_loader
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

    def model_completion(
        self,
        model: str,
        *,
        instruction_ids: pd.Index | None = None,
    ) -> pd.Series | None:
        """Return aligned completions, with missing rows left for generation."""
        outputs = self.model_outputs_for(model)
        if outputs is None:
            return None

        outputs["output"] = outputs["output"].fillna("")
        completions = (
            outputs.drop_duplicates("instruction_index", keep="last")
            .set_index("instruction_index")["output"]
            .sort_index()
        )
        target_index = (
            self.instructions.index
            if instruction_ids is None
            else pd.Index(instruction_ids, name=self.instructions.index.name)
        )
        logger.info("Found pre-existing completions for model %r.", model)
        return completions.reindex(target_index).rename("completion")


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


def validate_messages(messages: object) -> None:
    """Validate a completed text conversation, optionally starting with a system message."""
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
    if messages[-1]["role"] != "assistant":
        raise ValueError("Completed conversations must end with an assistant answer.")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{label} has duplicate column names after mapping.")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}.")


def _validate_model_outputs(outputs: pd.DataFrame) -> None:
    _require_columns(outputs, {"instruction_index", "model", "output"}, "Model outputs")
    if "messages" in outputs:
        for messages in outputs["messages"]:
            validate_messages(messages)
    if outputs["model"].isna().any():
        raise ValueError("Model outputs require model identities.")


@dataclass(frozen=True)
class PreferenceData(CompletionData):
    """Completion data with observed pairwise outcomes for evaluation.

    ``battles`` is a DataFrame with these canonical columns:

    - ``instruction_index``: an ID present in ``instructions.index``.
    - ``instruction``, ``completion_a``, ``completion_b``: prompt and answer text.
    - ``model_a``, ``model_b``: nonmissing identities for the two answer slots.
    - ``pref``: numeric preference in [0, 1]; 0 favors A, 0.5 is a tie, and
      1 favors B. Intermediate values express soft preferences. Missing outcomes
      are allowed; values are normalized to floats.
    - ``source``: nonmissing label provenance. Use ``"human"`` for human labels;
      ``human_battles()`` selects those rows with observed preferences and raises
      if none exist.

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
                "pref",
                "source",
            },
            "Battles",
        )
        if not self.battles["instruction_index"].isin(self.instructions.index).all():
            raise ValueError("Battles reference unknown instruction IDs.")
        if self.battles[["model_a", "model_b", "source"]].isna().any().any():
            raise ValueError("Battles require model identities and label provenance.")
        battles = self.battles.assign(pref=preference_values(self.battles["pref"]))
        object.__setattr__(self, "battles", battles)

    @classmethod
    def from_battles(
        cls,
        frame: pd.DataFrame,
        *,
        columns: dict[str, str] | None = None,
        outcome_preferences: dict[str, float] | None = None,
    ) -> PreferenceData:
        """Build aligned instruction, completion, and battle views from source rows.

        ``columns`` maps source names to canonical names, e.g.
        ``{"prompt": "instruction", "answer_left": "completion_a"}``.
        All battle columns listed on ``PreferenceData`` must be present after
        mapping, except ``instruction_index`` can be omitted to assign row IDs.
        Supplied IDs must be unique and nonmissing here: each battle becomes an
        instruction row. Keep repeated source question IDs in a metadata column.

        Supply either ``pref`` or a ``winner`` column with an explicit
        ``outcome_preferences`` mapping, e.g. ``{"left": 0, "draw": 0.5, "right": 1}``.
        Unmapped nonmissing labels raise. When both preferences and a mapping
        are supplied, their A/tie/B directions and missingness must agree;
        without a mapping, an existing ``pref`` leaves ``winner`` as metadata.
        Null answer text is omitted from the derived model outputs.
        """
        battles = frame.rename(columns=columns or {}).reset_index(drop=True).copy()
        _require_columns(
            battles,
            {
                "instruction",
                "model_a",
                "model_b",
                "completion_a",
                "completion_b",
                "source",
            },
            "Battles",
        )
        if "winner" in battles and ("pref" not in battles or outcome_preferences):
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
                raise ValueError(
                    f"Unknown battle outcomes: {unknown.unique().tolist()}"
                )
            mapped = preference_values(battles["winner"].map(outcome_preferences))
            if "pref" not in battles:
                battles["pref"] = mapped
            else:
                observed = hard_preferences(mapped)
                supplied = hard_preferences(battles["pref"])
                consistent = observed.eq(supplied) | (observed.isna() & supplied.isna())
                if not consistent.all():
                    raise ValueError(
                        "Source winner labels conflict with battle preferences."
                    )
        _require_columns(battles, {"pref"}, "Battles")
        if "instruction_index" not in battles:
            battles["instruction_index"] = range(len(battles))
        indexed = battles.set_index("instruction_index", drop=False)
        instructions = indexed.drop(
            columns=[
                "instruction_index",
                "model_a",
                "model_b",
                "completion_a",
                "completion_b",
                "pref",
                "source",
                "winner",
            ],
            errors="ignore",
        )
        outputs = pd.concat(
            [
                battles[
                    ["instruction_index", f"model_{slot}", f"completion_{slot}"]
                ].rename(
                    columns={f"model_{slot}": "model", f"completion_{slot}": "output"}
                )
                for slot in ("a", "b")
            ],
            ignore_index=True,
        ).dropna(subset=["output"])
        return cls(instructions, outputs, battles=battles)

    def human_battles(self) -> pd.DataFrame:
        """Require observed human outcomes for an anchored evaluation."""
        human = self.battles.loc[
            (self.battles["source"] == "human") & self.battles["pref"].notna()
        ].copy()
        if human.empty:
            raise ValueError(
                "Human-anchored evaluation requires human preference labels."
            )
        return human
