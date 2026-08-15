"""Judge a sampled arena battle set in the stored A/B order."""

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
    if abs(score - 0.5) <= eps:
        return "tie"
    return "model_b" if score > 0.5 else "model_a"


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


def annotate_sample(
    df_sample: pd.DataFrame,
    cfg: RunConfig,
    *,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
) -> pd.DataFrame:
    instructions, completions_a, completions_b = _battle_texts(df_sample)
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
    for annotation, (_, battle) in zip(annotations, df_sample.iterrows(), strict=True):
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
