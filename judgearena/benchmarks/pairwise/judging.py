"""Pairwise judge orchestration, retaining row identity across answer orders."""

import random
from dataclasses import fields

import pandas as pd

from judgearena.evaluate import JudgeAnnotation, judge_and_parse_prefs
from judgearena.prompts.registry import ResolvedJudgePrompt


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
    """Judge indexed instruction/completion_A/B/model_A/B rows.

    Returned annotations retain the actual judge slots; ``pref`` is always in
    the original A/B frame. For ``both``, all direct rows precede reversed rows.
    ``orientation`` describes scoring repetitions: single, direct, or reversed.
    Fixed/both preserve the supplied initial order. Optional ``instruction_id``
    and ``cache_orientation`` columns override cache identity and initial
    orientation; the unique table index remains the returned row identity.
    """
    judged = comparisons.copy()
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
        for prefix in ("completion", "model"):
            judged.loc[reverse_order, [f"{prefix}_A", f"{prefix}_B"]] = judged.loc[
                reverse_order, [f"{prefix}_B", f"{prefix}_A"]
            ].to_numpy()

    cache_orientation = judged.get(
        "cache_orientation", pd.Series("direct", index=judged.index)
    )
    cache_orientation = cache_orientation.where(
        ~reverse_order,
        cache_orientation.map({"direct": "reversed", "reversed": "direct"}),
    )

    annotation_columns = [field.name for field in fields(JudgeAnnotation)]
    direct_frames, reversed_frames = [], []
    for prompt, index in prompt_groups:
        group = judged.loc[index]
        annotations, reversed_annotations, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=group["instruction"].tolist(),
            completions_A=group["completion_A"].tolist(),
            completions_B=group["completion_B"].tolist(),
            swap_mode=swap_mode,
            strip_thinking_before_judging=strip_thinking_before_judging,
            system_prompt=prompt.system_prompt,
            user_prompt_template=prompt.user_prompt_template,
            prompt_preset=prompt.preset_name,
            parse=prompt.parser,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
            cache_row_metadata=[
                {
                    "instruction_id": str(getattr(row, "instruction_id", row.Index)),
                    "model_a": row.model_A,
                    "model_b": row.model_B,
                    "orientation": cache_orientation.loc[row.Index],
                }
                for row in group.itertuples()
            ],
        )
        frame = pd.DataFrame(annotations, columns=annotation_columns)
        frame["instruction_index"] = index
        frame["model_A"] = group["model_A"].to_numpy()
        frame["model_B"] = group["model_B"].to_numpy()
        frame["pref"] = prefs.iloc[: len(group)].astype("float64").to_numpy()
        frame["pref"] = frame["pref"].where(
            ~reverse_order.loc[index].to_numpy(), 1 - frame["pref"]
        )
        frame["orientation"] = "direct" if swap_mode == "both" else "single"
        direct_frames.append(frame)

        if reversed_annotations is not None:
            frame = pd.DataFrame(reversed_annotations, columns=annotation_columns)
            frame["instruction_index"] = index
            frame["model_A"] = group["model_B"].to_numpy()
            frame["model_B"] = group["model_A"].to_numpy()
            # judge_and_parse_prefs already reorients reversed preferences.
            frame["pref"] = prefs.iloc[len(group) :].astype("float64").to_numpy()
            frame["orientation"] = "reversed"
            reversed_frames.append(frame)

    return pd.concat(direct_frames + reversed_frames, ignore_index=True)
