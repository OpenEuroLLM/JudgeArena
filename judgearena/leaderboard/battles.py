"""Sample and judge battles between a model and the reference pool."""

from __future__ import annotations

import numpy as np
import pandas as pd

from judgearena.evaluate import judge_and_parse_prefs, resolve_run_judge_prompt
from judgearena.leaderboard.assemble import pref_to_win_a
from judgearena.leaderboard.pool import pool_completion, pool_models
from judgearena.models import make_model


def _quantize(pref) -> float:
    win_a = pref_to_win_a(pref)
    return float("nan") if win_a is None else 1.0 - win_a  # hardened pref (A-preferred = 0.0)


def sample_pool_battles(
    new_model: str,
    new_completions: pd.DataFrame,
    panel,
    completions: pd.DataFrame,
    *,
    n_per_pair: int,
    seed: int,
    opponents: list[str] | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    opponents = opponents or [m for m in pool_models(panel) if m != new_model]
    new_by_instr = dict(zip(new_completions["instruction"], new_completions["completion"], strict=True))
    new_lang = dict(zip(new_completions["instruction"], new_completions["lang"], strict=True))
    rows = []
    for opp in opponents:
        opp_instr = completions[completions["model"] == opp]["instruction"].tolist()
        shared = [i for i in opp_instr if i in new_by_instr]
        if not shared:
            continue
        if len(shared) > n_per_pair:
            shared = list(rng.choice(shared, size=n_per_pair, replace=False))
        for instr in shared:
            rows.append({
                "instruction": instr, "lang": new_lang[instr], "opponent": opp,
                "new_completion": new_by_instr[instr],
                "opp_completion": pool_completion(completions, opp, instr),
                "challenger_position": str(rng.choice(["A", "B"])),
            })
    if not rows:
        raise ValueError("disconnected: no shared instructions with the pool")
    return pd.DataFrame(rows)


def judge_pool_battles(specs: pd.DataFrame, new_model: str, *, judge_cfg, scorer, arena=None) -> pd.DataFrame:
    resolved = resolve_run_judge_prompt(arena, judge_cfg)
    judge_model = make_model(model=judge_cfg.model)

    pos = specs["challenger_position"].to_numpy()
    new_in_a = pos == "A"
    new_completion = specs["new_completion"].tolist()
    opp_completion = specs["opp_completion"].tolist()
    completions_A = [new_completion[i] if new_in_a[i] else opp_completion[i] for i in range(len(specs))]
    completions_B = [opp_completion[i] if new_in_a[i] else new_completion[i] for i in range(len(specs))]

    _, _, prefs = judge_and_parse_prefs(
        judge_chat_model=judge_model,
        instructions=specs["instruction"].tolist(),
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

    # Un-flip prefs back to the new=model_a frame (mirror score.py:_winrates).
    # prefs = P(judge-side-B wins); when new is in B, flip to P(new wins) = 1 - p.
    pref_mframe = [p if (p is None or new_in_a[i]) else (1.0 - p) for i, p in enumerate(prefs)]

    pref_series = pd.Series(pref_mframe).reset_index(drop=True)
    return pd.DataFrame({
        "model_a": new_model,
        "model_b": specs["opponent"].to_numpy(),
        "instruction": specs["instruction"].to_numpy(),
        "lang": specs["lang"].to_numpy(),
        "completion_a": specs["new_completion"].to_numpy(),
        "completion_b": specs["opp_completion"].to_numpy(),
        "judge_pref": pref_series.to_numpy(),
        "judge_pref_hard": pref_series.map(_quantize).to_numpy(),
        # Stored frame is canonical (new = model_a = position A); prefs were
        # un-flipped into it above, so "position" stays "A". The randomized judge
        # orientation lives only in specs' challenger_position and isn't persisted.
        # This keeps assemble's submission head-to-head consistent
        # (sub_win_a = pref_to_win_a(judge_pref) with the new model at A).
        "position": "A",
        "opponent": specs["opponent"].to_numpy(),
    })
