"""Prepare reference comparisons and align judge predictions with reference labels."""

from __future__ import annotations

import pandas as pd

from judgearena.benchmarks.pairwise.comparisons import conversation_comparisons


def prepare_reference_comparisons(sample: pd.DataFrame, *, turns=None) -> pd.DataFrame:
    """Use physical battle IDs for judging normalized reference comparisons."""
    comparisons = sample.set_index("battle_id", drop=False).assign(
        instruction_id=sample["battle_id"].to_numpy(), cache_orientation="direct"
    )
    return conversation_comparisons(comparisons, turns)


def reference_scoring_battles(
    pool: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    """Keep judged evidence and metadata alongside unjudged reference comparisons."""
    if (
        "battle_id" not in outcomes
        or not outcomes["battle_id"].isin(pool["battle_id"]).all()
    ):
        raise ValueError(
            "Reference scoring requires outcomes tied to their original battle IDs."
        )
    # Earlier conversation prefixes cannot inherit the full conversation's label.
    labeled = outcomes.loc[outcomes["reference_pref"].notna(), "battle_id"]
    unjudged = pool.loc[~pool["battle_id"].isin(labeled)].assign(
        pref=float("nan"),
        sampled=False,
    )
    return pd.concat(
        [outcomes.assign(sampled=True), unjudged], ignore_index=True
    ).sort_values(
        "battle_id",
        kind="stable",
        ignore_index=True,
    )
