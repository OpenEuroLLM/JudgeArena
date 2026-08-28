"""Judge a sampled arena battle set, optionally in both A/B orders."""

from __future__ import annotations

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


def preference_to_winner(preference: float | None) -> str:
    """Apply the tie boundary used by Erlis's meta-eval implementation."""
    if preference is None or pd.isna(preference):
        return "tie"
    if abs(float(preference) - 0.5) <= TIE_EPSILON:
        return "tie"
    return "model_b" if preference > 0.5 else "model_a"


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
) -> pd.DataFrame:
    """Attach shared judge outputs to the arena's stored A/B identity."""
    rows = []
    for annotation, preference, (_, battle) in zip(
        annotations,
        normalized_preferences,
        original.iterrows(),
        strict=True,
    ):
        # Preserve the meta-eval methodology's existing missing-parse policy.
        if preference is None or pd.isna(preference):
            preference = 0.5
        swapped = orientation == "swapped"
        rows.append(
            {
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
                "winner_llm": preference_to_winner(preference),
                "pref_llm": float(preference),
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
        )
    ]
    if reversed_annotations is not None:
        parts.append(
            _annotation_frame(
                df_sample,
                reversed_annotations,
                preferences.iloc[n_battles:],
                orientation="swapped",
            )
        )
    return pd.concat(parts, ignore_index=True)
