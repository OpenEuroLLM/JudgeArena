"""Apply one attempted-battle horizon to completed tuning searches."""

from __future__ import annotations

import pandas as pd


def clip_trials_to_common_budget(
    runs: dict[str, pd.DataFrame], panel_sizes: dict[int, int]
) -> tuple[dict[str, pd.DataFrame], dict]:
    ordered = {}
    totals = {}
    for name, source in runs.items():
        trials = source.copy()
        if "time_sampled" not in trials:
            raise ValueError(f"{name}: trials lack NePS time_sampled ordering metadata")
        trials["_sample_order"] = pd.to_numeric(trials["time_sampled"], errors="coerce")
        if trials["_sample_order"].isna().any():
            raise ValueError(f"{name}: invalid NePS time_sampled metadata")
        trials = trials.sort_values("_sample_order", kind="stable")
        trials["logical_battles"] = trials["battles_per_model"].map(panel_sizes)
        if trials["logical_battles"].isna().any():
            raise ValueError(f"{name}: unknown tuning fidelity in battles_per_model")
        trials["logical_evaluations"] = trials["logical_battles"].cumsum()
        totals[name] = int(trials["logical_battles"].sum())
        ordered[name] = trials

    horizon = min(totals.values())
    clipped = {}
    excluded = {}
    for name, trials in ordered.items():
        outside = trials["logical_evaluations"] > horizon
        excluded[name] = trials.loc[outside, "neps_trial_id"].astype(str).tolist()
        kept = trials.loc[~outside].copy()
        clipped[name] = kept.drop(columns="_sample_order")
    return clipped, {
        "logical_battle_horizon": horizon,
        "attempted_logical_battles_by_run": totals,
        "excluded_trial_ids_by_run": excluded,
    }
