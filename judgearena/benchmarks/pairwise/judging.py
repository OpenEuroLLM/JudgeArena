"""Pairwise judge orchestration, retaining row identity across answer orders."""

import random
from dataclasses import fields
from typing import Literal

import pandas as pd

from judgearena.battles import swap_comparison_slots
from judgearena.evaluate import JudgeAnnotation, judge_and_parse_prefs
from judgearena.prompts.registry import ResolvedJudgePrompt


def _annotation_frame(
    annotations: list[JudgeAnnotation],
    comparisons: pd.DataFrame,
    preferences: pd.Series,
    *,
    orientation: Literal["single", "direct", "reversed"],
) -> pd.DataFrame:
    """Attach row identity and actual judge slots without reorienting preferences."""
    slot_a, slot_b = ("b", "a") if orientation == "reversed" else ("a", "b")
    return pd.DataFrame(
        annotations, columns=[field.name for field in fields(JudgeAnnotation)]
    ).assign(
        instruction_index=comparisons.index,
        model_A=comparisons[f"model_{slot_a}"].to_numpy(),
        model_B=comparisons[f"model_{slot_b}"].to_numpy(),
        pref=preferences.to_numpy(dtype="float64"),
        orientation=orientation,
    )


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
    """Judge indexed instruction/completion_a/b/model_a/b rows.

    Returned annotations keep the existing export columns and actual judge slots;
    ``pref`` is always in the original A/B frame. For ``both``, all direct rows
    precede reversed rows.
    ``orientation`` describes scoring repetitions: single, direct, or reversed.
    Fixed/both preserve the supplied initial order. Optional ``instruction_id``
    and ``cache_orientation`` columns override cache identity and initial
    orientation; the unique table index remains the returned row identity.
    """
    judged = comparisons
    reverse_order = pd.Series(False, index=judged.index)
    if swap_mode == "random":
        # Preserve the historical seed and mask: False puts the baseline first.
        reverse_order = pd.Series(
            [
                not random.Random(f"is_switched_outputs{instruction}0").choices(
                    [False, True], k=1
                )[0]
                for instruction in judged["instruction"]
            ],
            index=judged.index,
        )
        judged = swap_comparison_slots(comparisons, reverse_order)

    cache_orientation = judged.get(
        "cache_orientation", pd.Series("direct", index=judged.index)
    )
    cache_orientation = cache_orientation.where(
        ~reverse_order,
        cache_orientation.map({"direct": "reversed", "reversed": "direct"}),
    )

    direct_frames, reversed_frames = [], []
    for prompt, index in prompt_groups:
        group = judged.loc[index]
        if group[["instruction", "completion_a", "completion_b"]].isna().any().any():
            raise ValueError(
                "Cannot judge comparisons with missing instruction/completion text."
            )
        annotations, reversed_annotations, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=group["instruction"].tolist(),
            completions_A=group["completion_a"].tolist(),
            completions_B=group["completion_b"].tolist(),
            swap_mode=swap_mode,
            strip_thinking_before_judging=strip_thinking_before_judging,
            prompt=prompt,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
            cache_row_metadata=[
                {
                    "instruction_id": str(getattr(row, "instruction_id", row.Index)),
                    "model_a": row.model_a,
                    "model_b": row.model_b,
                    "orientation": cache_orientation.loc[row.Index],
                }
                for row in group.itertuples()
            ],
        )
        frame = _annotation_frame(
            annotations,
            group,
            prefs.iloc[: len(group)],
            orientation="direct" if swap_mode == "both" else "single",
        )
        frame["pref"] = frame["pref"].where(
            ~reverse_order.loc[index].to_numpy(), 1 - frame["pref"]
        )
        direct_frames.append(frame)

        if reversed_annotations is not None:
            # judge_and_parse_prefs already reorients reversed preferences.
            reversed_frames.append(
                _annotation_frame(
                    reversed_annotations,
                    group,
                    prefs.iloc[len(group) :],
                    orientation="reversed",
                )
            )

    return pd.concat(direct_frames + reversed_frames, ignore_index=True)
