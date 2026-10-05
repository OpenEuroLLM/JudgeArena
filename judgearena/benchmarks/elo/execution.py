"""Candidate judging and battle conversion shared by Elo flows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from judgearena.benchmarks.elo.rating import prefs_to_battle_results
from judgearena.benchmarks.execution import build_judge
from judgearena.evaluate import PairScore, combine_swapped_prefs, judge_and_parse_prefs

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.evaluate import JudgeAnnotation, JudgeParser
    from judgearena.prompts.registry import ResolvedJudgePrompt


def judge_candidate_battles(
    cfg: RunConfig,
    panel: pd.DataFrame,
    completions: pd.Series,
    resolved_prompt: ResolvedJudgePrompt,
) -> tuple[list[JudgeAnnotation], list[JudgeAnnotation] | None, pd.Series]:
    """Judge prepared opponents in panel order, with instruction IDs in its index."""
    our_completions = completions.tolist()
    opponent_completions = panel["opponent_completion"].tolist()
    opponent_models = panel["opponent_model"].tolist()
    our_model_is_position_a = panel["candidate_position"].eq("A").to_numpy()
    n = len(panel)
    completions_A = [
        our_completions[i] if our_model_is_position_a[i] else opponent_completions[i]
        for i in range(n)
    ]
    completions_B = [
        opponent_completions[i] if our_model_is_position_a[i] else our_completions[i]
        for i in range(n)
    ]
    return judge_and_parse_prefs(
        judge_chat_model=build_judge(cfg),
        instructions=panel["instruction"].tolist(),
        completions_A=completions_A,
        completions_B=completions_B,
        swap_mode=cfg.judge.swap_mode,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        parse=resolved_prompt.parser,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=False,
        cache_row_metadata=[
            {
                "instruction_id": str(panel.index[index]),
                "model_a": (
                    cfg.model.name
                    if our_model_is_position_a[index]
                    else opponent_models[index]
                ),
                "model_b": (
                    opponent_models[index]
                    if our_model_is_position_a[index]
                    else cfg.model.name
                ),
                "orientation": (
                    "direct" if our_model_is_position_a[index] else "reversed"
                ),
            }
            for index in range(n)
        ],
    )


def build_candidate_battles(
    cfg: RunConfig,
    panel: pd.DataFrame,
    completions: pd.Series,
    judged: tuple[list[JudgeAnnotation], list[JudgeAnnotation] | None, pd.Series],
    *,
    parser: JudgeParser,
    effective_temperature: float,
) -> pd.DataFrame:
    """Build canonical per-pass battles, retaining original text for focal metrics."""
    annotations, annotations_reversed, prefs = judged
    row_annotations = list(annotations)
    passes = 1
    if annotations_reversed is not None:
        row_annotations += annotations_reversed
        passes = 2

    # Reparse at this run's temperature, keeping the selected parser's validation.
    if cfg.elo.soft_elo and isinstance(parser, PairScore):
        reparsed_prefs: list[float] = []
        for annotation in row_annotations:
            parsed = parser.parse_result(
                annotation.judge_completion, temperature=effective_temperature
            )
            reparsed_prefs.append(float("nan") if parsed is None else parsed.preference)
        new_prefs_ab = pd.Series(reparsed_prefs, dtype=float)
        if cfg.judge.swap_mode == "both":
            n_half = len(row_annotations) // 2
            prefs = combine_swapped_prefs(new_prefs_ab[:n_half], new_prefs_ab[n_half:])
        else:
            prefs = new_prefs_ab

    # Both passes use the original A/B orientation; preferences are canonical.
    our_model_is_position_a = panel["candidate_position"].eq("A").tolist() * passes
    opponent_models = panel["opponent_model"].tolist() * passes
    question_ids = panel["question_id"].tolist() * passes
    df_llm_judge = prefs_to_battle_results(
        prefs.tolist(),
        our_model_is_position_a,
        opponent_models,
        cfg.model.name,
        judge_model=cfg.judge.model,
        question_ids=question_ids,
    )

    row_our_completions = completions.tolist() * passes
    row_opponent_completions = panel["opponent_completion"].tolist() * passes
    focal_is_a = pd.Series(our_model_is_position_a, dtype="bool")
    df_llm_judge["evaluation_model"] = cfg.model.name
    df_llm_judge["completion_a"] = pd.Series(row_our_completions).where(
        focal_is_a, row_opponent_completions
    )
    df_llm_judge["completion_b"] = pd.Series(row_opponent_completions).where(
        focal_is_a, row_our_completions
    )
    df_llm_judge["instruction_index"] = question_ids
    if cfg.judge.swap_mode == "both":
        half = len(df_llm_judge) // 2
        df_llm_judge["orientation"] = ["direct"] * half + ["reversed"] * half
    else:
        df_llm_judge["orientation"] = "single"
    return df_llm_judge
