"""Prepare text or conversation comparisons from loaded and generated outputs."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import pandas as pd

from judgearena.benchmarks.execution import (
    load_or_generate_completions,
)
from judgearena.benchmarks.pairwise.baselines import BaselinePlan
from judgearena.datasets.data import CompletionData, _require_columns, validate_messages
from judgearena.generate import (
    generate_base,
    generate_instructions,
    generate_messages,
)

if TYPE_CHECKING:
    from judgearena.config import RunConfig


def conversation_comparisons(
    frame: pd.DataFrame,
    turns: Literal["all", "last"] | tuple[int, ...] | None = None,
    *,
    source_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """Expand each candidate pair into selected assistant-ending prefixes.

    `last` compares complete conversations, including histories of unequal length.
    Numbered turns compare matching assistant positions. Human labels attached to
    a whole conversation are only retained on its complete comparison.
    """
    if turns is None:
        return frame
    _require_columns(frame, {"messages_a", "messages_b"}, "Turn selection")
    rows = []
    for identity, row in frame.iterrows():
        a, b = row["messages_a"], row["messages_b"]
        validate_messages(a)
        validate_messages(b)
        ends_a = [
            i + 1 for i, message in enumerate(a) if message["role"] == "assistant"
        ]
        ends_b = [
            i + 1 for i, message in enumerate(b) if message["role"] == "assistant"
        ]
        if turns == "last":
            selected = [max(len(ends_a), len(ends_b))]
        else:
            if len(ends_a) != len(ends_b):
                raise ValueError(
                    "Turn-wise comparisons require equally many answers; use turns='last' for complete conversations."
                )
            selected = (
                range(1, len(ends_a) + 1)
                if turns == "all"
                else sorted(set(turns) & set(range(1, len(ends_a) + 1)))
            )
        for turn in selected:
            prefix_a = a if turns == "last" else a[: ends_a[turn - 1]]
            prefix_b = b if turns == "last" else b[: ends_b[turn - 1]]
            item = {
                **row.to_dict(),
                "comparison_id": f"{identity}:turn-{turn}",
                "source_instruction_index": identity
                if source_ids is None
                else source_ids.loc[identity],
                "question_id": row.get("question_id", identity),
                "turn": turn,
                "messages_a": prefix_a,
                "messages_b": prefix_b,
                "completion_a": prefix_a[-1]["content"],
                "completion_b": prefix_b[-1]["content"],
            }
            if "instruction_id" in row:
                item["instruction_id"] = f"{row['instruction_id']}:turn-{turn}"
            if "reference_pref" in item and (
                len(prefix_a) != len(a) or len(prefix_b) != len(b)
            ):
                item["reference_pref"] = float("nan")
            rows.append(item)
    if not rows:
        raise ValueError("No conversation turns match the judging selection.")
    return pd.DataFrame(rows).set_index("comparison_id")


def select_reference_battles(battles: pd.DataFrame) -> pd.DataFrame:
    """Select observed reference labels for reference-based sampling or scoring."""
    if "reference_pref" not in battles:
        raise ValueError("This evaluation requires reference preferences.")
    selected = battles.loc[battles["reference_pref"].notna()].copy()
    if selected.empty:
        raise ValueError("This evaluation requires observed reference preferences.")
    return selected


def preload_model_outputs(data: CompletionData, models: list[str]) -> CompletionData:
    """Load requested saved outputs once, before prompt validation or generation."""
    if data.model_output_loader is None:
        return data
    frames = [
        outputs
        for model in dict.fromkeys(models)
        if (outputs := data.model_outputs_for(model)) is not None
    ]
    return replace(
        data,
        model_outputs=pd.concat(frames, ignore_index=True) if frames else None,
        model_output_loader=None,
    )


def prepare_answers(
    cfg,
    data: CompletionData,
    selected: pd.DataFrame,
    model: str,
    *,
    role: Literal["A", "B"],
) -> pd.DataFrame:
    """Load saved answers and generate missing ones using the task's input format."""
    generation = cfg.resolved_task.spec.protocol.generation
    answers = data.model_output_rows(model, instruction_ids=selected.index)
    if answers is None or answers["completion"].isna().any():
        inputs = selected["instruction"]
        if generation.mode == "provided_messages":
            _require_columns(selected, {"messages"}, "Provided-history generation")
            inputs = selected["messages"]
        inputs = inputs.set_axis(selected.get("instruction_id", selected.index))
        answers = load_or_generate_completions(
            cfg,
            model,
            inputs,
            answers,
            role=role,
            generate={
                "base_completion": generate_base,
                "single_turn_chat": generate_instructions,
                "provided_messages": generate_messages,
            }[generation.mode],
            use_tqdm=cfg.run.use_tqdm,
        ).set_axis(selected.index)
    if generation.mode == "provided_messages" and "messages" in selected:
        messages = answers.get(
            "messages", pd.Series(None, index=selected.index, dtype=object)
        )
        answers["messages"] = [
            saved
            if isinstance(saved, list)
            else history + [{"role": "assistant", "content": completion}]
            for saved, history, completion in zip(
                messages, selected["messages"], answers["completion"], strict=True
            )
        ]
    return answers


def assemble_comparisons(
    instructions: pd.DataFrame, answers_a: pd.DataFrame, answers_b: pd.DataFrame
) -> pd.DataFrame:
    """Align selected answer rows by input ID, retaining metadata and conversations.

    Inputs and answers have unique input IDs as their index. Reference labels
    belong to the original answer pair and are not inherited by new pairs.
    """
    metadata = instructions.drop(
        columns=["reference_pref", "pref", "pref_hard"]
        + [column for column in instructions if column.endswith(("_a", "_b"))],
        errors="ignore",
    )
    return metadata.join(answers_a.add_suffix("_a"), validate="one_to_one").join(
        answers_b.add_suffix("_b"), validate="one_to_one"
    )


def prepare_comparisons(
    cfg: RunConfig,
    data: CompletionData,
    instructions: pd.DataFrame,
    baseline_plan: BaselinePlan,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return comparisons and their explicit source-instruction routing IDs."""
    generation = cfg.resolved_task.spec.protocol.generation
    provided_messages = generation.mode == "provided_messages"

    baseline = baseline_plan.aligned_to(instructions.index)
    a = prepare_answers(cfg, data, instructions, cfg.model.name, role="A")
    frames = []
    for model in baseline_plan.unique_models:
        ids = baseline.index[baseline == model]
        b = prepare_answers(cfg, data, instructions.loc[ids], model, role="B")
        frames.append(assemble_comparisons(instructions.loc[ids], a.loc[ids], b))
    frame = pd.concat(frames)
    if baseline.index.is_unique:
        frame = frame.loc[instructions.index]
        source_ids = pd.Series(frame.index, index=frame.index)
    else:
        frame["source_instruction_index"] = frame.index
        frame.index = pd.Index(
            [
                json.dumps([str(identity), model])
                for identity, model in zip(frame.index, frame["model_b"], strict=True)
            ],
            name="comparison_id",
        )
        source_ids = frame["source_instruction_index"].copy()
    # Supplied histories may carry stable turn IDs for generation and judging.
    internal = (
        ["cache_orientation"]
        if provided_messages
        else ["instruction_id", "cache_orientation"]
    )
    frame = frame.drop(columns=internal, errors="ignore")
    if cfg.resolved_task.spec.protocol.judge.turns is not None:
        frame = conversation_comparisons(
            frame, cfg.resolved_task.spec.protocol.judge.turns, source_ids=source_ids
        )
        source_ids = frame["source_instruction_index"].copy()
    return frame, source_ids
