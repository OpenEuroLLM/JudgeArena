"""Judge a sampled arena battle set, optionally in both A/B orders."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.evaluate import JudgeAnnotation, judge_and_parse_prefs

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.prompts.registry import ResolvedJudgePrompt

TIE_EPSILON = 0.01


def serialize_judge_input(judge_input: object) -> str:
    if judge_input is None:
        return ""
    to_string = getattr(judge_input, "to_string", None)
    if callable(to_string):
        return to_string()
    return str(judge_input)


def invert_winner(winner: str) -> str:
    if winner == "model_a":
        return "model_b"
    if winner == "model_b":
        return "model_a"
    return winner


def preference_to_winner(preference: float | None) -> str | None:
    """Apply the tie boundary, preserving missing parser outputs as missing."""
    if preference is None or pd.isna(preference):
        return None
    if abs(float(preference) - 0.5) <= TIE_EPSILON:
        return "tie"
    return "model_b" if preference > 0.5 else "model_a"


def validate_battle_conversations(df: pd.DataFrame) -> None:
    """Validate sampled conversation shapes before constructing a paid judge."""
    for _, battle in df.iterrows():
        battle_id = battle.get("battle_id", battle.get("question_id", "unknown"))
        for column in ("conversation_a", "conversation_b"):
            conversation = battle[column]
            if not isinstance(conversation, (list, tuple)) or not conversation:
                raise ValueError(
                    f"Battle {battle_id!r} has an empty or invalid {column}."
                )
            turns = conversation[:2]
            if any(not isinstance(turn, dict) for turn in turns):
                raise ValueError(
                    f"Battle {battle_id!r} has non-object turns in {column}."
                )
            for turn in turns:
                try:
                    extract_turn_text(turn)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Battle {battle_id!r} has invalid turn content in {column}."
                    ) from exc


def _battle_texts(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    instructions = [extract_turn_text(conv[0]) for conv in df["conversation_a"]]
    completions_a = [
        extract_turn_text(conv[1]) if len(conv) > 1 else ""
        for conv in df["conversation_a"]
    ]
    completions_b = [
        extract_turn_text(conv[1]) if len(conv) > 1 else ""
        for conv in df["conversation_b"]
    ]
    return instructions, completions_a, completions_b


def _annotation_frame(
    original: pd.DataFrame,
    annotations: list[JudgeAnnotation],
    normalized_preferences: pd.Series,
    *,
    orientation: str,
    parser_name: str,
) -> pd.DataFrame:
    """Attach shared judge outputs to the arena's stored A/B identity."""
    rows = []
    for annotation, preference, (_, battle) in zip(
        annotations,
        normalized_preferences,
        original.iterrows(),
        strict=True,
    ):
        parse_ok = preference is not None and not pd.isna(preference)
        normalized_preference = float(preference) if parse_ok else float("nan")
        swapped = orientation == "swapped"
        presented_preference = (
            1.0 - normalized_preference
            if swapped and parse_ok
            else normalized_preference
        )
        rows.append(
            {
                "battle_id": battle["battle_id"],
                "question_id": battle["question_id"],
                "model_a": battle["model_a"],
                "model_b": battle["model_b"],
                "winner": battle["winner"],
                "lang": battle["lang"],
                "instruction": annotation.instruction,
                "completion_a": (
                    annotation.completion_B if swapped else annotation.completion_A
                ),
                "completion_b": (
                    annotation.completion_A if swapped else annotation.completion_B
                ),
                "judge_input": serialize_judge_input(annotation.judge_input),
                "judge_completion": annotation.judge_completion,
                "judge_parser": parser_name,
                "judge_top_logprobs_json": (
                    json.dumps(
                        annotation.judge_top_logprobs,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if annotation.judge_top_logprobs is not None
                    else None
                ),
                "parse_ok": parse_ok,
                "pref_presented": presented_preference,
                "winner_llm": preference_to_winner(normalized_preference),
                "pref_llm": normalized_preference,
                "orientation": orientation,
                "presented_model_a": (
                    battle["model_b"] if swapped else battle["model_a"]
                ),
                "presented_model_b": (
                    battle["model_a"] if swapped else battle["model_b"]
                ),
                "presented_completion_a": annotation.completion_A,
                "presented_completion_b": annotation.completion_B,
            }
        )
    return pd.DataFrame(rows)


def aggregate_battle_preferences(
    annotations: pd.DataFrame, *, swap_mode: str
) -> pd.DataFrame:
    """Combine normalized judge passes into one row per physical battle."""
    expected_passes = 2 if swap_mode == "both" else 1
    expected_orientations = (
        {"forward", "swapped"} if swap_mode == "both" else {"forward"}
    )
    rows = []
    for battle_id, passes in annotations.groupby("battle_id", sort=False):
        orientations = set(passes["orientation"])
        if len(passes) != expected_passes or orientations != expected_orientations:
            raise ValueError(
                f"Battle {battle_id!r} has {len(passes)} passes and orientations "
                f"{sorted(orientations)}; expected {sorted(expected_orientations)}."
            )
        valid = passes[passes["parse_ok"] & passes["pref_llm"].notna()]
        parsed = len(valid)
        preference = float(valid["pref_llm"].mean()) if parsed else float("nan")
        first = passes.iloc[0]
        rows.append(
            {
                "battle_id": battle_id,
                "question_id": first["question_id"],
                "model_a": first["model_a"],
                "model_b": first["model_b"],
                "winner": first["winner"],
                "lang": first["lang"],
                "parse_ok": parsed > 0,
                "pref_llm": preference,
                "winner_llm": preference_to_winner(preference),
                "n_passes_expected": expected_passes,
                "n_passes_parsed": parsed,
                "parse_status": (
                    "complete"
                    if parsed == expected_passes
                    else "partial"
                    if parsed
                    else "missing"
                ),
            }
        )
    aggregated = pd.DataFrame(rows)
    if aggregated["battle_id"].duplicated().any():
        raise ValueError(
            "Aggregated meta-eval battles must have unique battle_id values."
        )
    return aggregated


def annotate_sample(
    df_sample: pd.DataFrame,
    cfg: RunConfig,
    *,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
) -> pd.DataFrame:
    parser = resolved_prompt.parser
    if parser is None:
        raise ValueError(
            f"Prompt preset {resolved_prompt.preset_name!r} has no judge parser."
        )

    df_sample = df_sample.copy()
    if "battle_id" not in df_sample:
        raise ValueError("annotate_sample requires stable battle_id values.")
    if df_sample["battle_id"].isna().any() or df_sample["battle_id"].duplicated().any():
        raise ValueError("annotate_sample requires unique, non-null battle_id values.")
    validate_battle_conversations(df_sample)

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
            orientation="forward",
            parser_name=parser.name,
        )
    ]
    if reversed_annotations is not None:
        parts.append(
            _annotation_frame(
                df_sample,
                reversed_annotations,
                preferences.iloc[n_battles:],
                orientation="swapped",
                parser_name=parser.name,
            )
        )
    return pd.concat(parts, ignore_index=True)
