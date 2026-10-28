"""Pairwise judge orchestration, retaining row identity across answer orders."""

from __future__ import annotations

import random
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Literal

import pandas as pd

from judgearena.battles import agreement_outcomes, swap_comparison_slots
from judgearena.datasets.data import _require_columns, validate_messages
from judgearena.evaluate import (
    render_comparison_requests,
    validate_battle_prompt_fields,
)
from judgearena.judging import (
    answer_text,
    execute_pairwise,
    pairwise_annotation_frame,
)
from judgearena.prompts.registry import (
    ResolvedJudgePrompt,
    group_judge_prompts,
    prompt_metadata,
)

if TYPE_CHECKING:
    from judgearena.config import RunConfig


def validate_comparisons(comparisons: pd.DataFrame) -> None:
    """Require indexed model pairs with text answers or completed conversations.

    Text pairs provide instruction/completion_a/completion_b. Conversation pairs
    provide messages_a/messages_b; flattened answers and instruction are optional.
    Request IDs must remain unique as strings; source/cache IDs may repeat.
    """
    _require_columns(comparisons, {"model_a", "model_b"}, "Comparisons")
    if (
        comparisons.index.isna().any()
        or not comparisons.index.is_unique
        or not comparisons.index.map(str).is_unique
    ):
        raise ValueError("Comparisons require unique, nonmissing request IDs.")
    if comparisons[["model_a", "model_b"]].isna().any().any():
        raise ValueError("Comparisons require model identities.")
    for slot in ("a", "b"):
        messages = comparisons.get(
            f"messages_{slot}", pd.Series(None, index=comparisons.index, dtype=object)
        )
        for history in messages.dropna():
            validate_messages(history)
        text_rows = comparisons.loc[messages.isna()]
        if not text_rows.empty:
            fields = {"instruction", f"completion_{slot}"}
            _require_columns(text_rows, fields, "Comparisons")
            if text_rows[list(fields)].isna().any().any():
                raise ValueError(
                    "Cannot judge comparisons with missing instruction/completion text."
                )


def comparison_text(
    comparisons: pd.DataFrame, *, strip_thinking: bool = False
) -> pd.DataFrame:
    """Render each candidate's saved conversation, falling back to completion text."""
    frame = comparisons.copy()
    for slot in ("a", "b"):
        raw = frame.get(
            f"completion_{slot}", pd.Series(None, index=frame.index, dtype=object)
        ).copy()
        missing = raw.isna()
        if missing.any():
            raw.loc[missing] = frame.loc[missing, f"messages_{slot}"].map(
                lambda messages: messages[-1]["content"]
            )
        frame[f"completion_{slot}_text"] = raw.map(
            lambda text: answer_text(text, strip_thinking=strip_thinking)
        )
        frame[f"completion_{slot}"] = [
            answer_text(
                row.get(f"completion_{slot}"),
                row.get(f"messages_{slot}"),
                strip_thinking=strip_thinking,
            )
            for row in frame.to_dict("records")
        ]
    if "instruction" not in frame:
        frame["instruction"] = "Compare the following conversations."
    return frame


def _render_standard_requests(
    comparisons, *, prompt, truncate_input_chars, strip_thinking
):
    judged = comparison_text(comparisons, strip_thinking=strip_thinking)
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
    for index, _, _ in batches:
        validate_comparisons(comparisons.loc[index])
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
        for prompt in prompts:
            prompt.validate_for("pairwise")
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


def prepare_standard_prompts(
    cfg: RunConfig,
    resolved_prompt: ResolvedJudgePrompt,
    instructions: pd.DataFrame,
    *,
    available_columns=None,
) -> tuple[Callable, float | None]:
    """Resolve standard prompt groups before generation and bind their renderers."""
    groups = group_judge_prompts(cfg, instructions, resolved_prompt)
    for prompt in [resolved_prompt, *(prompt for prompt, _ in groups)]:
        prompt.validate_for("pairwise", inference_kwargs=cfg.judge_model_kwargs())
        if prompt.delegated:
            raise ValueError(
                f"delegated prompt preset {prompt.preset_name!r} cannot be used for generic pairwise judging."
            )

        validate_battle_prompt_fields(
            prompt,
            None
            if available_columns is None
            else set(available_columns)
            | {"instruction", "model_a", "model_b", "completion_a", "completion_b"},
        )

    def prepare(comparisons, *, source_ids=None):
        for prompt, _ in groups:
            validate_battle_prompt_fields(prompt, comparisons.columns)
        routing = (
            comparisons.index
            if source_ids is None
            else source_ids.loc[comparisons.index]
        )
        selected = [
            (prompt, comparisons.index[routing.isin(ids)]) for prompt, ids in groups
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
    *,
    swap_mode: str | None = None,
    use_tqdm: bool | None = None,
    available_columns=None,
) -> tuple[Callable, float | None]:
    """Bind prompt preparation and outcome policy to the shared judging loop."""
    swap_mode = cfg.judge.swap_mode if swap_mode is None else swap_mode
    use_tqdm = cfg.run.use_tqdm if use_tqdm is None else use_tqdm
    spec = cfg.resolved_task.spec.protocol.judge
    if spec.minimum_score_by:
        raise ValueError("minimum_score_by requires a single-answer scoring prompt.")
    if resolved_prompt.prepare_prompts is None:
        prepare, default_temperature = prepare_standard_prompts(
            cfg, resolved_prompt, instructions, available_columns=available_columns
        )
    else:
        prepare, default_temperature = resolved_prompt.prepare_prompts(
            cfg, resolved_prompt, instructions
        )

    def judge(model, comparisons, *, source_ids=None):
        return judge_pairwise_batches(
            model,
            comparisons,
            prepare(comparisons, source_ids=source_ids),
            swap_mode=swap_mode,
            use_tqdm=use_tqdm,
            aggregation=resolved_prompt.aggregation,
        )

    return judge, default_temperature
