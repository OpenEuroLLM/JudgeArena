"""Score a model's completions against a frozen panel.

Pairs with ``leaderboard/curate.py``, which produces the frozen panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from judgearena.config import JudgeArgs
from judgearena.estimate_elo_ratings import (
    _prefs_to_battle_results,
    fit_bradley_terry,
)
from judgearena.evaluate import (
    PairScore,
    judge_and_parse_prefs,
    resolve_run_judge_prompt,
)
from judgearena.generate import generate_instructions
from judgearena.leaderboard.anchors import human_elo_from_battles
from judgearena.leaderboard.assemble import CI_PERCENTILES, RNG_SEED_MAX
from judgearena.leaderboard.panel import Panel
from judgearena.leaderboard.record import ResultRecord
from judgearena.models import make_model
from judgearena.utils import compute_pref_summary


def generate_panel_completions(
    panel: Panel,
    model: str,
    *,
    instructions: pd.Series | None = None,
    max_out_tokens: int = 32768,
    truncate_all_input_chars: int = 8192,
    **engine_kwargs,
) -> list[str]:
    """Generate the model's answer to each panel instruction, in row order.

    When *instructions* is given, generate for those (in order) instead of
    ``panel.battles["instruction"]``.
    """
    if instructions is None:
        instructions = panel.battles["instruction"]
    instructions = pd.Series(list(instructions), name="instruction")
    df = generate_instructions(
        instructions=instructions,
        model=model,
        truncate_input_chars=truncate_all_input_chars,
        max_tokens=max_out_tokens,
        use_tqdm=True,
        **engine_kwargs,
    ).set_index("instruction_index")
    return df.loc[:, "completion"].tolist()


def _orient_challenger(
    battles: pd.DataFrame,
    completions: list[str],
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Return (chal_pos_a, opponent_models, completions_A, completions_B)."""
    chal_pos_a = (battles["challenger_position"] == "A").to_numpy()
    opponent_models = battles["challenger_opponent"].tolist()
    opponent_completions = [
        row["completion_a"] if row["challenger_opponent"] == row["model_a"]
        else row["completion_b"]
        for _, row in battles.iterrows()
    ]
    completions_A = [
        completions[i] if chal_pos_a[i] else opponent_completions[i]
        for i in range(len(battles))
    ]
    completions_B = [
        opponent_completions[i] if chal_pos_a[i] else completions[i]
        for i in range(len(battles))
    ]
    return chal_pos_a, opponent_models, completions_A, completions_B


def _fit_challenger_ratings(
    df_results: pd.DataFrame,
    model: str,
    pref_col: str,
    baseline_model: str | None,
    n_bootstraps: int,
    rng: np.random.Generator,
) -> tuple[float, float, list[float]]:
    """Pooled bootstrap BT → (elo_overall, elo_std, elo_ci)."""
    model_ratings: list[float] = []
    for _ in range(n_bootstraps):
        sample = df_results.sample(
            n=len(df_results), replace=True, random_state=int(rng.integers(0, RNG_SEED_MAX))
        )
        ratings = fit_bradley_terry(sample, pref_col=pref_col, baseline_model=baseline_model)
        if model in ratings:
            model_ratings.append(ratings[model])
    elo_overall = float(np.mean(model_ratings)) if model_ratings else float("nan")
    elo_std = float(np.std(model_ratings)) if model_ratings else float("nan")
    elo_ci = (
        [
            float(np.percentile(model_ratings, CI_PERCENTILES[0])),
            float(np.percentile(model_ratings, CI_PERCENTILES[1])),
        ]
        if model_ratings else [float("nan"), float("nan")]
    )
    return elo_overall, elo_std, elo_ci


def _per_language_elo(
    df_results: pd.DataFrame,
    battles: pd.DataFrame,
    model: str,
    pref_col: str,
    baseline_model: str | None,
) -> dict[str, float]:
    """Single BT fit per language."""
    elo_per_lang: dict[str, float] = {}
    for lang in sorted(battles["lang"].unique()):
        ratings = fit_bradley_terry(
            df_results[df_results["lang"] == lang],
            pref_col=pref_col,
            baseline_model=baseline_model,
        )
        elo_per_lang[lang] = float(ratings.get(model, float("nan")))
    return elo_per_lang


def _winrates(
    prefs: pd.Series,
    chal_pos_a: np.ndarray,
    battles: pd.DataFrame,
) -> tuple[float, dict[str, float], dict[str, int]]:
    """Overall + per-language win rate and battle counts for the challenger."""
    oriented = pd.Series(
        [p if (p is None or chal_pos_a[i]) else (1 - p) for i, p in enumerate(prefs)]
    )
    winrate_overall = compute_pref_summary(oriented).winrate
    winrate_per_lang: dict[str, float] = {}
    n_battles_per_lang: dict[str, int] = {}
    for lang in sorted(battles["lang"].unique()):
        idx = battles.index[battles["lang"] == lang]
        winrate_per_lang[lang] = compute_pref_summary(oriented.iloc[idx]).winrate
        n_battles_per_lang[lang] = int(len(idx))
    return winrate_overall, winrate_per_lang, n_battles_per_lang


def _mae_vs_human(
    df_results: pd.DataFrame,
    battles: pd.DataFrame,
    model: str,
    pref_col: str,
    baseline_model: str | None,
) -> float:
    """MAE between pooled judge-ELO and human-ELO for anchor models."""
    human_elo = human_elo_from_battles(battles, baseline_model)
    pooled = fit_bradley_terry(df_results, pref_col=pref_col, baseline_model=baseline_model)
    overlap = [m for m in pooled if m in human_elo and m != model]
    return (
        float(np.mean([abs(pooled[m] - human_elo[m]) for m in overlap]))
        if overlap else float("nan")
    )


def score_against_panel(
    panel: Panel,
    completions: list[str],
    *,
    model: str,
    judge_cfg: JudgeArgs,
    n_bootstraps: int = 100,
    seed: int = 0,
    generation_params: dict | None = None,
    scorer: PairScore | None = None,
) -> ResultRecord:
    """Judge the challenger battles, fit pooled + per-language BT, build a record."""
    rng = np.random.default_rng(seed)
    battles = panel.battles.reset_index(drop=True)
    scorer_meta = dict(panel.meta.get("scorer", {}))
    temperature = scorer_meta.get("temperature", 0.3)
    method = scorer_meta.get("method", "soft")
    pref_col = "pref_hard" if method == "hard" else "pref"
    baseline_model = panel.meta.get("baseline_model")
    if scorer is None:
        scorer = PairScore(temperature=temperature)

    # Orient challenger vs its frozen opponent at the frozen position.
    chal_pos_a, opponent_models, completions_A, completions_B = _orient_challenger(
        battles, completions
    )

    resolved = resolve_run_judge_prompt(panel.meta.get("arena"), judge_cfg)
    judge_model = make_model(model=judge_cfg.model)
    annotations, _, prefs = judge_and_parse_prefs(
        judge_chat_model=judge_model,
        instructions=battles["instruction"].tolist(),
        completions_A=completions_A,
        completions_B=completions_B,
        swap_mode=judge_cfg.swap_mode,
        provide_explanation=judge_cfg.provide_explanation,
        system_prompt=resolved.system_prompt,
        user_prompt_template=resolved.user_prompt_template,
        prompt_preset=resolved.preset_name,
        score_parser=scorer,
        use_tqdm=True,
    )
    prefs = pd.Series(prefs).reset_index(drop=True)

    # Challenger battle rows (model-name level), with language attached.
    df_challenger = _prefs_to_battle_results(prefs, chal_pos_a, opponent_models, model)
    df_challenger["lang"] = battles["lang"].to_numpy()

    # Frozen anchor battles carry their stored judge prefs.
    df_anchor = pd.DataFrame(
        {
            "model_a": battles["model_a"],
            "model_b": battles["model_b"],
            "pref": battles["judge_pref"],
            "pref_hard": battles["judge_pref_hard"],
            "lang": battles["lang"],
        }
    )
    df_results = pd.concat([df_challenger, df_anchor], ignore_index=True)

    # Pooled BT with bootstrap CI on the challenger's rating.
    elo_overall, elo_std, elo_ci = _fit_challenger_ratings(
        df_results, model, pref_col, baseline_model, n_bootstraps, rng
    )

    # Per-language ELO (single fit per language).
    elo_per_lang = _per_language_elo(df_results, battles, model, pref_col, baseline_model)

    # Win-rate from challenger prefs oriented so 0 = challenger wins.
    winrate_overall, winrate_per_lang, n_battles_per_lang = _winrates(
        prefs, chal_pos_a, battles
    )

    # MAE vs human-ELO (diagnostic) from the panel's human labels.
    mae_vs_human = _mae_vs_human(df_results, battles, model, pref_col, baseline_model)

    # Per-battle artifact: raw judge scores for this model.
    battles_out = pd.DataFrame(
        {
            "battle_id": battles["battle_id"],
            "lang": battles["lang"],
            "opponent": opponent_models,
            "position": battles["challenger_position"],
            "challenger_completion": completions,
            "judge_completion": [a.judge_completion for a in annotations],
            "judge_pref": prefs.to_numpy(),
            "winner": df_challenger["winner"].to_numpy(),
        }
    )

    return ResultRecord(
        model=model,
        panel_version=panel.meta.get("panel_version", ""),
        panel_hash=panel.meta.get("panel_hash", ""),
        judge_model=judge_cfg.model,
        elo_overall=elo_overall,
        elo_std=elo_std,
        elo_ci=elo_ci,
        elo_per_lang=elo_per_lang,
        winrate_overall=winrate_overall,
        winrate_per_lang=winrate_per_lang,
        n_battles=int(len(battles)),
        n_battles_per_lang=n_battles_per_lang,
        kappa_per_lang=dict(panel.meta.get("kappa_per_language", {})),
        mae_vs_human=mae_vs_human,
        scorer=scorer_meta or {"method": method, "temperature": temperature},
        generation_params=generation_params or {},
        seed=seed,
        battles=battles_out,
    )
