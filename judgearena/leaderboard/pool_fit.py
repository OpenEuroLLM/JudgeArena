"""Extend the reference pool with a new model (re-fit) or place a model against it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from judgearena.leaderboard.panel import PANEL_BATTLE_COLUMNS, Panel
from judgearena.leaderboard.record import ResultRecord
from judgearena.leaderboard.score import (
    _fit_challenger_ratings,
    _mae_vs_human,
    _per_language_elo,
    _winrates,
)


def extend_pool(
    panel: Panel,
    completions: pd.DataFrame,
    new_model: str,
    new_completions: pd.DataFrame,
    new_battles: pd.DataFrame,
    *,
    bump_version: str,
) -> tuple[Panel, pd.DataFrame]:
    """Append a judge-only model's battles+completions, bump the version, carry trust meta."""
    # Conform new battles to the panel battle schema (denormalized; null human/anchor-only cols).
    rows = pd.DataFrame({col: pd.NA for col in PANEL_BATTLE_COLUMNS}, index=new_battles.index)
    rows["battle_id"] = [f"{new_model}|{r.model_b}|{r.instruction}" for r in new_battles.itertuples()]
    for col in ("lang", "model_a", "model_b", "instruction", "judge_pref", "judge_pref_hard"):
        rows[col] = new_battles[col].to_numpy()
    merged_battles = pd.concat([panel.battles, rows[PANEL_BATTLE_COLUMNS]], ignore_index=True)

    merged_completions = pd.concat([completions, new_completions], ignore_index=True)

    meta = dict(panel.meta)
    meta["panel_version"] = bump_version
    meta["pool_models"] = [*meta.get("pool_models", []), new_model]
    # anchor_models, scorer (temperature), kappa_per_language carried over unchanged.
    return Panel(meta=meta, battles=merged_battles), merged_completions


def place_against_pool(
    panel: Panel,
    new_model: str,
    new_battles: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
) -> ResultRecord:
    """Fit new_model against the frozen pool (pool unchanged); return a submission record."""
    method = panel.meta.get("scorer", {}).get("method", "soft")
    pref_col = "pref_hard" if method == "hard" else "pref"
    baseline_model = panel.meta.get("baseline_model")
    rng = np.random.default_rng(seed)

    # Pool battles mapped to pref/pref_hard columns expected by the BT helpers.
    pool_results = pd.DataFrame({
        "model_a": panel.battles["model_a"],
        "model_b": panel.battles["model_b"],
        "pref": panel.battles["judge_pref"],
        "pref_hard": panel.battles["judge_pref_hard"],
        "lang": panel.battles["lang"],
    })
    # New model battles mapped to the same schema.
    new_results = pd.DataFrame({
        "model_a": new_battles["model_a"],
        "model_b": new_battles["model_b"],
        "pref": new_battles["judge_pref"],
        "pref_hard": new_battles["judge_pref_hard"],
        "lang": new_battles["lang"],
    })
    df_results = pd.concat([new_results, pool_results], ignore_index=True)

    # Pooled bootstrap BT → (elo_overall, elo_std, elo_ci).
    elo_overall, elo_std, elo_ci = _fit_challenger_ratings(
        df_results, new_model, pref_col, baseline_model, n_bootstraps, rng
    )

    # Per-language ELO (single fit per language over combined battles).
    elo_per_lang = _per_language_elo(df_results, df_results, new_model, pref_col, baseline_model)

    # Win-rate from the new model's own battles (model_a = new_model always).
    nb = new_battles.reset_index(drop=True)
    prefs = nb["judge_pref"]
    chal_pos_a = np.ones(len(nb), dtype=bool)  # new_model is always model_a
    winrate_overall, winrate_per_lang, n_battles_per_lang = _winrates(prefs, chal_pos_a, nb)

    # MAE vs human-ELO using the pool's human labels.
    mae_vs_human = _mae_vs_human(df_results, panel.battles, new_model, pref_col, baseline_model)

    battles_out = new_battles.reset_index(drop=True)

    return ResultRecord(
        model=new_model,
        panel_version=panel.meta.get("panel_version", ""),
        panel_hash=panel.meta.get("panel_hash", ""),
        judge_model=panel.meta.get("judge_model", ""),
        elo_overall=elo_overall,
        elo_std=elo_std,
        elo_ci=elo_ci,
        elo_per_lang=elo_per_lang,
        winrate_overall=winrate_overall,
        winrate_per_lang=winrate_per_lang,
        n_battles=int(len(new_battles)),
        n_battles_per_lang=n_battles_per_lang,
        kappa_per_lang=dict(panel.meta.get("kappa_per_language", {})),
        mae_vs_human=mae_vs_human,
        scorer=dict(panel.meta.get("scorer", {})),
        generation_params={},
        seed=seed,
        battles=battles_out,
    )
