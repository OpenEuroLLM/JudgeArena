"""Judge a sampled arena battle set, optionally in both A/B orders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.evaluate import PairScore, annotate_battles

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.prompts.registry import ResolvedJudgePrompt

TIE_EPSILON = 0.01
# Paper PairScore temperature for meta-eval; generate+judge keeps 0.3.
META_EVAL_PAIRSCORE_TEMPERATURE = 0.5


def serialize_judge_input(judge_input: object) -> str:
    if judge_input is None:
        return ""
    to_string = getattr(judge_input, "to_string", None)
    if callable(to_string):
        return to_string()
    return str(judge_input)


def parse_pairscore_pref(
    judge_completion: str, *, temperature: float = META_EVAL_PAIRSCORE_TEMPERATURE
) -> float:
    score = PairScore(temperature=temperature).parse_model_raw(judge_completion)
    if score is None or np.isnan(score):
        return 0.5
    return float(score)


def parse_pairscore_winner(
    judge_completion: str,
    *,
    temperature: float = META_EVAL_PAIRSCORE_TEMPERATURE,
    eps: float = TIE_EPSILON,
) -> str:
    score = parse_pairscore_pref(judge_completion, temperature=temperature)
    if abs(score - 0.5) < eps:
        return "tie"
    return "model_b" if score > 0.5 + eps else "model_a"


def invert_winner(winner: str) -> str:
    if winner == "model_a":
        return "model_b"
    if winner == "model_b":
        return "model_a"
    return winner


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


def _swap_batch(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    swapped["conversation_a"] = df["conversation_b"]
    swapped["conversation_b"] = df["conversation_a"]
    swapped["model_a"] = df["model_b"]
    swapped["model_b"] = df["model_a"]
    return swapped


def _judge_pass(
    df_batch: pd.DataFrame,
    cfg: RunConfig,
    *,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
) -> pd.DataFrame:
    instructions, completions_a, completions_b = _battle_texts(df_batch)
    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_a,
        completions_B=completions_b,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        provide_explanation=cfg.judge.provide_explanation,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        use_tqdm=cfg.run.use_tqdm,
    )
    rows = []
    for annotation, (_, battle) in zip(annotations, df_batch.iterrows(), strict=True):
        completion = annotation.judge_completion
        rows.append(
            {
                "question_id": battle["question_id"],
                "model_a": battle["model_a"],
                "model_b": battle["model_b"],
                "winner": battle["winner"],
                "lang": battle["lang"],
                "instruction": annotation.instruction,
                "completion_a": annotation.completion_A,
                "completion_b": annotation.completion_B,
                "judge_input": serialize_judge_input(annotation.judge_input),
                "judge_completion": completion,
                "winner_llm": parse_pairscore_winner(completion),
                "pref_llm": parse_pairscore_pref(completion),
            }
        )
    return pd.DataFrame(rows)


def _normalize_pass(
    pass_frame: pd.DataFrame, original: pd.DataFrame, *, orientation: str
) -> pd.DataFrame:
    """Map a judged pass back onto the arena's stored A/B identity."""
    normalized = pass_frame.copy()
    normalized["orientation"] = orientation
    normalized["presented_model_a"] = pass_frame["model_a"].tolist()
    normalized["presented_model_b"] = pass_frame["model_b"].tolist()
    normalized["presented_completion_a"] = pass_frame["completion_a"].tolist()
    normalized["presented_completion_b"] = pass_frame["completion_b"].tolist()
    normalized["model_a"] = original["model_a"].tolist()
    normalized["model_b"] = original["model_b"].tolist()
    normalized["winner"] = original["winner"].tolist()
    if orientation == "swapped":
        normalized["completion_a"] = pass_frame["completion_b"].tolist()
        normalized["completion_b"] = pass_frame["completion_a"].tolist()
        normalized["winner_llm"] = [
            invert_winner(winner) for winner in pass_frame["winner_llm"]
        ]
        normalized["pref_llm"] = 1.0 - pass_frame["pref_llm"]
    return normalized


def annotate_sample(
    df_sample: pd.DataFrame,
    cfg: RunConfig,
    *,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
) -> pd.DataFrame:
    parts = [
        _normalize_pass(
            _judge_pass(
                df_sample,
                cfg,
                judge_chat_model=judge_chat_model,
                resolved_prompt=resolved_prompt,
            ),
            df_sample,
            orientation="forward",
        )
    ]
    if cfg.judge.swap_mode == "both":
        parts.append(
            _normalize_pass(
                _judge_pass(
                    _swap_batch(df_sample),
                    cfg,
                    judge_chat_model=judge_chat_model,
                    resolved_prompt=resolved_prompt,
                ),
                df_sample,
                orientation="swapped",
            )
        )
    return pd.concat(parts, ignore_index=True)
