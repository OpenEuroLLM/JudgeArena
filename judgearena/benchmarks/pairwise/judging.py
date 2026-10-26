"""Pairwise judge orchestration, retaining row identity across answer orders."""

from __future__ import annotations

import random
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal

import pandas as pd

from judgearena.battles import agreement_outcomes, swap_comparison_slots
from judgearena.benchmarks.pairwise.scoring.metrics import score_outcomes
from judgearena.datasets.data import _require_columns, validate_messages
from judgearena.evaluate import render_comparison_requests
from judgearena.judging import (
    execute_pairwise,
    judge_single_answers,
    pairwise_annotation_frame,
)
from judgearena.prompts.registry import ResolvedJudgePrompt, prompt_metadata
from judgearena.utils import strip_thinking_tags

if TYPE_CHECKING:
    from judgearena.config import RunConfig


def comparison_text(
    comparisons: pd.DataFrame, *, strip_thinking: bool = False
) -> pd.DataFrame:
    """Prepare text prompt fields from indexed text pairs or complete conversations.

    Conversation rows require messages_a/b and model_a/b. Each message has role
    and content; candidate histories may differ. Other columns remain metadata.
    """
    _require_columns(comparisons, {"model_a", "model_b"}, "Comparisons")
    if not comparisons.index.is_unique or comparisons.index.isna().any():
        raise ValueError("Comparisons require unique, nonmissing row IDs.")
    if comparisons[["model_a", "model_b"]].isna().any().any():
        raise ValueError("Comparisons require model identities.")
    frame = comparisons.copy()
    if {"messages_a", "messages_b"} & set(frame.columns):
        _require_columns(
            frame, {"messages_a", "messages_b"}, "Conversation comparisons"
        )
        for slot in ("a", "b"):
            rendered = []
            for messages in frame[f"messages_{slot}"]:
                validate_messages(messages)
                rendered.append(
                    "\n".join(
                        f"{message['role']}: "
                        + (
                            strip_thinking_tags(message["content"])
                            if strip_thinking and message["role"] == "assistant"
                            else message["content"]
                        )
                        for message in messages
                    )
                )
            frame[f"completion_{slot}"] = rendered
        if "instruction" not in frame:
            frame["instruction"] = "Compare the following conversations."
    _require_columns(
        frame, {"instruction", "completion_a", "completion_b"}, "Comparisons"
    )
    if frame[["instruction", "completion_a", "completion_b"]].isna().any().any():
        raise ValueError(
            "Cannot judge comparisons with missing instruction/completion text."
        )
    return frame


def _render_standard_requests(
    comparisons, *, prompt, truncate_input_chars, strip_thinking
):
    judged = comparison_text(comparisons, strip_thinking=strip_thinking)
    if strip_thinking and "messages_a" not in judged:
        for column in ("completion_a", "completion_b"):
            judged[column] = judged[column].map(strip_thinking_tags)
    return render_comparison_requests(judged, prompt, truncate_input_chars)


def _standard_prompt_batches(groups, *, strip_thinking, truncate_input_chars):
    return [
        (
            index,
            [prompt] * len(index),
            partial(
                _render_standard_requests,
                prompt=prompt,
                truncate_input_chars=truncate_input_chars,
                strip_thinking=strip_thinking,
            ),
        )
        for prompt, index in groups
    ]


def judge_pairwise_batches(
    judge_chat_model,
    comparisons: pd.DataFrame,
    batches: list[tuple[pd.Index, list[ResolvedJudgePrompt], Callable]],
    *,
    swap_mode: str,
    use_tqdm: bool,
    aggregation: Literal["individual", "agreement"] = "individual",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Execute prepared renderers and assemble annotations and scoring outcomes.

    Each batch supplies comparison IDs, one prompt per row, and a renderer.
    Renderers may group requests internally. Batch order controls scheduling;
    returned direct annotations precede reversed annotations across all batches.
    """
    reverse_order = pd.Series(False, index=comparisons.index)
    if swap_mode == "random":
        # Preserve the historical seed and mask: False puts the baseline first.
        instructions = comparisons.get(
            "instruction",
            pd.Series("Compare the following conversations.", index=comparisons.index),
        )
        reverse_order = instructions.map(
            lambda instruction: (
                not random.Random(f"is_switched_outputs{instruction}0").choices(
                    [False, True], k=1
                )[0]
            )
        ).astype(bool)
    judged = swap_comparison_slots(comparisons, reverse_order)
    cache_orientation = judged.get(
        "cache_orientation", pd.Series("direct", index=judged.index)
    )
    judged = judged.assign(
        cache_orientation=cache_orientation.where(
            ~reverse_order,
            cache_orientation.map({"direct": "reversed", "reversed": "direct"}),
        )
    )
    direct_frames, reversed_frames = [], []
    for index, prompts, render in batches:
        direct, reverse = execute_pairwise(
            judge_chat_model,
            judged.loc[index],
            render,
            swap_mode="both" if swap_mode == "both" else "fixed",
            use_tqdm=use_tqdm,
        )
        direct_frames.append(
            pairwise_annotation_frame(
                comparisons.loc[index],
                direct,
                prompts,
                orientation="direct" if reverse is not None else "single",
                swapped=reverse_order.loc[index].tolist(),
            )
        )
        if reverse is not None:
            reversed_frames.append(
                pairwise_annotation_frame(
                    comparisons.loc[index],
                    reverse,
                    prompts,
                    orientation="reversed",
                    swapped=True,
                )
            )
    annotations = pd.concat(direct_frames + reversed_frames, ignore_index=True)
    outcomes = (
        agreement_outcomes(annotations) if aggregation == "agreement" else annotations
    )
    metadata = {
        "judge_prompts": prompt_metadata(
            prompt for _, prompts, _ in batches for prompt in prompts
        ),
    }
    if aggregation == "agreement":
        metadata["num_inconsistent"] = int(outcomes["inconsistent"].sum())
    return annotations, outcomes, metadata


def judge_comparisons(
    judge_chat_model,
    comparisons: pd.DataFrame,
    prompt_groups: list[tuple[ResolvedJudgePrompt, pd.Index]],
    *,
    swap_mode: str,
    strip_thinking_before_judging: bool,
    truncate_input_chars: int | None,
    use_tqdm: bool,
) -> pd.DataFrame:
    """Judge text or conversation comparisons using standard prompt fields.

    Returned model/answer columns and preferences use the original A/B slots.
    ``swapped`` records actual judge input order; ``orientation`` identifies
    single, direct, or reversed passes. Optional ``instruction_id`` and
    ``cache_orientation`` override cache identity and initial orientation.
    """
    batches = _standard_prompt_batches(
        prompt_groups,
        strip_thinking=strip_thinking_before_judging,
        truncate_input_chars=truncate_input_chars,
    )
    annotations, _, _ = judge_pairwise_batches(
        judge_chat_model,
        comparisons,
        batches,
        swap_mode=swap_mode,
        use_tqdm=use_tqdm,
    )
    return annotations


def _build_judge_batches(
    cfg: RunConfig,
    instructions_df: pd.DataFrame,
    resolved_prompt: ResolvedJudgePrompt,
) -> list[tuple[ResolvedJudgePrompt, pd.Index]]:
    """Group evaluation rows that share the same resolved judge prompt.

    Tasks may map categories to prompt presets (e.g. Arena-Hard v2.0 judges
    creative writing with a different system prompt). Runtime prompt overrides
    (an explicit preset or prompt files) disable the per-category mapping.
    """
    category_prompts = getattr(
        cfg.resolved_task.spec.protocol.judge, "category_prompts", {}
    )
    if (
        not category_prompts
        or (
            cfg.judge.prompt_preset is not None
            and cfg.judge.prompt_preset
            != cfg.resolved_task.spec.protocol.judge.default_prompt_preset
        )
        or resolved_prompt.source != "preset"
        or "category" not in instructions_df.columns
    ):
        return [(resolved_prompt, instructions_df.index)]

    from judgearena.prompts.registry import resolve_judge_prompt

    categories = instructions_df["category"]
    groups: list[tuple[ResolvedJudgePrompt, pd.Index]] = []
    for category, rows in categories.groupby(
        categories, sort=False, dropna=False, observed=True
    ):
        preset = None if pd.isna(category) else category_prompts.get(category)
        prompt = (
            resolved_prompt if preset is None else resolve_judge_prompt(preset=preset)
        )
        groups.append((prompt, rows.index))
    return groups


def prepare_standard_prompts(
    cfg: RunConfig,
    resolved_prompt: ResolvedJudgePrompt,
    instructions: pd.DataFrame,
) -> tuple[Callable, float | None]:
    """Resolve standard prompt groups before generation and bind their renderers."""
    groups = _build_judge_batches(cfg, instructions, resolved_prompt)
    for prompt in [resolved_prompt, *(prompt for prompt, _ in groups)]:
        if prompt.delegated:
            raise ValueError(
                f"delegated prompt preset {prompt.preset_name!r} cannot be used for generic pairwise judging."
            )

    def prepare(comparisons):
        selected = groups
        if cfg.resolved_task.spec.protocol.generation.mode == "multi_turn_chat":
            selected = [
                (
                    prompt,
                    comparisons.index[
                        comparisons["source_instruction_index"].isin(ids)
                    ],
                )
                for prompt, ids in groups
            ]
        selected = [(prompt, ids) for prompt, ids in selected if len(ids)]
        return _standard_prompt_batches(
            selected,
            strip_thinking=cfg.judge.strip_thinking_before_judging,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        )

    return prepare, None


def prepare_pairwise_judging(
    cfg: RunConfig,
    resolved_prompt: ResolvedJudgePrompt,
    instructions: pd.DataFrame,
) -> tuple[Callable, float | None]:
    """Bind prompt preparation and outcome policy to the shared judging loop."""
    spec = cfg.resolved_task.spec.protocol.judge
    if spec.format == "independent_scores":
        if resolved_prompt.score_renderer is None:
            raise ValueError(
                "Independent scoring requires a prompt with a score renderer."
            )
        _require_columns(instructions, set(spec.minimum_score_by), "Score grouping")
        render = partial(
            resolved_prompt.score_renderer,
            prompt_preset=resolved_prompt.preset_name,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        )
        return partial(
            judge_independent_comparisons,
            render=render,
            prompt=resolved_prompt,
            minimum_by=spec.minimum_score_by,
            use_tqdm=cfg.run.use_tqdm,
        ), None
    prepare, default_temperature = (
        cfg.resolved_task.spec.protocol.judge.prepare_prompts(
            cfg, resolved_prompt, instructions
        )
    )

    def judge(model, comparisons):
        return judge_pairwise_batches(
            model,
            comparisons,
            prepare(comparisons),
            swap_mode=cfg.judge.swap_mode,
            use_tqdm=cfg.run.use_tqdm,
            aggregation=resolved_prompt.aggregation,
        )

    return judge, default_temperature


def judge_independent_comparisons(
    model,
    comparisons: pd.DataFrame,
    *,
    render: Callable,
    prompt: ResolvedJudgePrompt,
    minimum_by: tuple[str, ...] = (),
    use_tqdm: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Grade all A answers, then all B answers, and compare their scalar scores."""
    scored = [
        judge_single_answers(
            model,
            comparisons.assign(
                model=comparisons[f"model_{slot}"],
                completion=comparisons[f"completion_{slot}"],
            ),
            render,
            use_tqdm=use_tqdm,
        )
        for slot in ("a", "b")
    ]
    frames = [frame for frame, _ in scored]
    turns = score_outcomes(
        comparisons.reset_index(drop=True).assign(
            instruction_index=comparisons.index,
            score_A=frames[0]["score"],
            score_B=frames[1]["score"],
        )
    )
    annotations = pd.concat(frames, ignore_index=True).merge(
        turns[["instruction_index", "score_A", "score_B", "pref"]].rename(
            columns={"pref": "turn_preference"}
        ),
        on="instruction_index",
        how="left",
        validate="many_to_one",
    )
    outcomes = score_outcomes(turns, minimum_by=minimum_by).assign(
        orientation="single",
        prompt_preset=prompt.preset_name,
    )
    return (
        annotations,
        outcomes,
        {
            "judge_prompts": prompt_metadata(
                prompt for _, prompts in scored for prompt in prompts
            ),
        },
    )
