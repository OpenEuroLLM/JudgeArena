"""Judge sampled arena battles and produce canonical physical-battle rows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.evaluate import JudgeAnnotation, judge_and_parse_prefs

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


def _battle_texts(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Validate each conversation pair and return the text sent to the judge."""
    instructions, completions_a, completions_b = [], [], []
    for _, battle in df.iterrows():
        battle_id = battle.get("battle_id", battle.get("question_id", "unknown"))
        conversations = []
        for column in ("conversation_a", "conversation_b"):
            conversation = battle[column]
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
        instructions.append(conversations[0][0])
        completions_a.append(conversations[0][1])
        completions_b.append(conversations[1][1])
    return instructions, completions_a, completions_b


def _serialize_mapping(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _annotation_frame(
    original: pd.DataFrame,
    annotations: list[JudgeAnnotation],
    preferences: pd.Series,
    *,
    orientation: str,
) -> pd.DataFrame:
    """Store one judge pass with preferences in the arena's A/B orientation."""
    rows = []
    for annotation, battle_id, preference in zip(
        annotations, original["battle_id"], preferences, strict=True
    ):
        parsed = annotation.parsed
        rows.append(
            {
                "battle_id": battle_id,
                "orientation": orientation,
                "pref": preference,
                "judge_input": serialize_judge_input(annotation.judge_input),
                "judge_completion": annotation.judge_completion,
                "judge_top_logprobs_json": _serialize_mapping(
                    annotation.judge_top_logprobs
                ),
                "parsed_label": None if parsed is None else parsed.label,
                "parsed_scores_json": (
                    None if parsed is None else _serialize_mapping(parsed.scores)
                ),
                "parsed_details_json": (
                    None if parsed is None else _serialize_mapping(parsed.details)
                ),
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
    parser = resolved_prompt.parser
    assert parser is not None
    instructions, completions_a, completions_b = _battle_texts(df_sample)
    annotations, reversed_annotations, preferences = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_a,
        completions_B=completions_b,
        swap_mode=cfg.judge.swap_mode,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        parse=parser,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
    )

    n_battles = len(df_sample)
    parts = [
        _annotation_frame(
            df_sample,
            annotations,
            preferences.iloc[:n_battles],
            orientation="direct" if reversed_annotations is not None else "single",
        )
    ]
    if reversed_annotations is not None:
        parts.append(
            _annotation_frame(
                df_sample,
                reversed_annotations,
                preferences.iloc[n_battles:],
                orientation="reversed",
            )
        )
    return pd.concat(parts, ignore_index=True)
