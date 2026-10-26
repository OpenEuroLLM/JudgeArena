"""Judge sampled arena battles and produce canonical physical-battle rows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.benchmarks.pairwise.judging import judge_comparisons
from judgearena.datasets.arena_battles import validate_conversations

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.prompts.registry import ResolvedJudgePrompt


def serialize_judge_input(judge_input: object) -> str:
    if judge_input is None:
        return ""
    to_string = getattr(judge_input, "to_string", None)
    if callable(to_string):
        return to_string()
    return str(judge_input)


def _serialize_mapping(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _annotation_frame(annotations: pd.DataFrame) -> pd.DataFrame:
    """Keep the meta-evaluation artifact schema over shared judge annotations."""
    rows = []
    for annotation in annotations.itertuples():
        parsed = annotation.parsed or {}
        rows.append(
            {
                "battle_id": annotation.instruction_index,
                "orientation": annotation.orientation,
                "pref": annotation.pref,
                "judge_input": serialize_judge_input(annotation.judge_input),
                "judge_completion": annotation.judge_completion,
                "judge_top_logprobs_json": _serialize_mapping(
                    annotation.judge_top_logprobs
                ),
                "parsed_label": parsed.get("label"),
                "parsed_scores_json": _serialize_mapping(parsed.get("scores")),
                "parsed_details_json": _serialize_mapping(parsed.get("details")),
            }
        )
    return pd.DataFrame(rows)


def aggregate_battle_preferences(
    annotations: pd.DataFrame, *, swap_mode: str
) -> pd.DataFrame:
    """Combine canonical judge passes into one row per physical battle."""
    expected_orientations = (
        {"direct", "reversed"} if swap_mode == "both" else {"single"}
    )
    rows = []
    for battle_id, passes in annotations.groupby("battle_id", sort=False):
        orientations = set(passes["orientation"])
        if (
            len(passes) != len(expected_orientations)
            or orientations != expected_orientations
        ):
            raise ValueError(
                f"Battle {battle_id!r} has {len(passes)} passes and orientations "
                f"{sorted(orientations)}; expected {sorted(expected_orientations)}."
            )
        preference = (
            float(passes["pref"].mean())
            if passes["pref"].notna().all()
            else float("nan")
        )
        rows.append({"battle_id": battle_id, "pref": preference})
    return pd.DataFrame(rows, columns=["battle_id", "pref"])


def annotate_sample(
    df_sample: pd.DataFrame,
    cfg: RunConfig,
    *,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
) -> pd.DataFrame:
    assert resolved_prompt.parser is not None
    if {"conversation_a", "conversation_b"} & set(df_sample.columns):
        validate_conversations(df_sample)
    comparisons = df_sample.set_index("battle_id").assign(
        instruction_id=df_sample["battle_id"].to_numpy(), cache_orientation="direct"
    )
    annotations = judge_comparisons(
        judge_chat_model,
        comparisons,
        [(resolved_prompt, comparisons.index)],
        swap_mode=cfg.judge.swap_mode,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
    )
    return _annotation_frame(annotations)
