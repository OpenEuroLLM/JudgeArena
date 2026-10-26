"""Prepare text or conversation comparisons from loaded and generated outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from judgearena.benchmarks.execution import (
    build_completion_cache,
    build_generation_kwargs,
    complete_missing,
    load_or_generate_completions,
)
from judgearena.benchmarks.pairwise.baselines import BaselinePlan
from judgearena.datasets.data import CompletionData, _require_columns, validate_messages
from judgearena.generate import (
    generate_base,
    generate_instructions,
    generate_messages,
    generate_multiturn,
)

if TYPE_CHECKING:
    from judgearena.config import RunConfig


def load_conversations(
    cfg: RunConfig,
    data: CompletionData,
    questions: pd.DataFrame,
    model: str,
    *,
    role: Literal["A", "B"],
) -> pd.Series:
    """Reuse complete saved conversations and generate missing rows by ID."""
    loaded = data.model_outputs_for(model)
    existing = None
    if loaded is not None:
        if "messages" not in loaded:
            raise ValueError("Saved conversation outputs require messages.")
        existing = loaded.drop_duplicates("instruction_index", keep="last").set_index(
            "instruction_index"
        )["messages"]

    def generate(pending):
        kwargs = build_generation_kwargs(cfg, model, role=role)
        temperatures = (
            None
            if "temperature" in kwargs
            else dict(cfg.resolved_task.spec.protocol.generation.category_temperatures)
        )
        return generate_multiturn(
            questions=pending,
            model=model,
            truncate_input_chars=cfg.generation.truncate_all_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            temperature_config=temperatures,
            strip_thinking_before_history=cfg.judge.strip_thinking_before_judging,
            inference_cache=build_completion_cache(cfg),
            **kwargs,
        ).set_index("instruction_index")["messages"]

    return complete_missing(questions, existing, generate)


def conversation_comparisons(
    frame: pd.DataFrame,
    turns: Literal["all", "last"] | tuple[int, ...] = "all",
) -> pd.DataFrame:
    """Expand each candidate pair into selected assistant-ending prefixes.

    `last` compares complete conversations, including histories of unequal length.
    Numbered turns compare matching assistant positions. Human labels attached to
    a whole conversation are only retained on its complete comparison.
    """
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
                "source_instruction_index": identity,
                "question_id": row.get("question_id", identity),
                "turn": turn,
                "messages_a": prefix_a,
                "messages_b": prefix_b,
                "completion_a": prefix_a[-1]["content"],
                "completion_b": prefix_b[-1]["content"],
            }
            if "reference_pref" in item and (
                len(prefix_a) != len(a) or len(prefix_b) != len(b)
            ):
                item["reference_pref"] = float("nan")
            rows.append(item)
    if not rows:
        raise ValueError("No conversation turns match the judging selection.")
    return pd.DataFrame(rows).set_index("comparison_id")


def prepare_comparisons(
    cfg: RunConfig,
    data: CompletionData,
    instructions: pd.DataFrame,
    baseline_plan: BaselinePlan,
) -> pd.DataFrame:
    """Resolve candidate outputs and keep dataset metadata on comparison rows."""
    generation = cfg.resolved_task.spec.protocol.generation
    conversations = generation.mode == "multi_turn_chat"
    provided_messages = generation.mode == "provided_messages"
    if provided_messages:
        _require_columns(instructions, {"messages"}, "Provided-history generation")

    def load(model, role, ids):
        selected = instructions.loc[ids]
        if conversations:
            return load_conversations(cfg, data, selected, model, role=role)
        inputs = selected["instruction"]
        if provided_messages:
            inputs = selected["messages"].set_axis(
                selected.get("instruction_id", selected.index)
            )
        return load_or_generate_completions(
            cfg,
            model,
            inputs,
            data.model_completion(model, instruction_ids=ids),
            role=role,
            generate={
                "base_completion": generate_base,
                "single_turn_chat": generate_instructions,
                "provided_messages": generate_messages,
            }[generation.mode],
            use_tqdm=cfg.run.use_tqdm,
        ).set_axis(ids)

    baseline = baseline_plan.aligned_to(instructions.index)
    a = load(cfg.model.name, "A", instructions.index)
    b = pd.concat(
        [
            load(model, "B", baseline.index[baseline == model])
            for model in baseline_plan.unique_models
        ]
    ).reindex(instructions.index)
    column = "messages" if conversations else "completion"
    # Histories belong to the selected model outputs, not the source battle.
    frame = instructions.drop(
        columns=["messages_a", "messages_b"], errors="ignore"
    ).assign(
        **{
            f"{column}_a": a,
            f"{column}_b": b,
            "model_a": cfg.model.name,
            "model_b": baseline,
        }
    )
    # Supplied histories may carry stable turn IDs for generation and judging.
    internal = (
        ["cache_orientation"]
        if provided_messages
        else ["instruction_id", "cache_orientation"]
    )
    frame = frame.drop(columns=internal, errors="ignore")
    if conversations:
        frame = conversation_comparisons(
            frame, cfg.resolved_task.spec.protocol.judge.turns
        )
    return frame
