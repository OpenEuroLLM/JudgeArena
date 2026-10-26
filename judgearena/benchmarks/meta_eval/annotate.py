"""Prepare reference comparisons and align judge predictions with human labels."""

from __future__ import annotations

import pandas as pd

from judgearena.datasets.arena_battles import validate_conversations


def aggregate_battle_preferences(
    annotations: pd.DataFrame, *, swap_mode: str
) -> pd.DataFrame:
    """Combine canonical judge passes into one row per physical battle."""
    expected_orientations = (
        {"direct", "reversed"} if swap_mode == "both" else {"single"}
    )
    rows = []
    for battle_id, passes in annotations.groupby("battle_id", sort=False):
        orientations = set(passes["orientation"])
        if (
            len(passes) != len(expected_orientations)
            or orientations != expected_orientations
        ):
            raise ValueError(
                f"Battle {battle_id!r} has {len(passes)} passes and orientations "
                f"{sorted(orientations)}; expected {sorted(expected_orientations)}."
            )
        preference = (
            float(passes["pref"].mean())
            if passes["pref"].notna().all()
            else float("nan")
        )
        rows.append({"battle_id": battle_id, "pref": preference})
    return pd.DataFrame(rows, columns=["battle_id", "pref"])


def prepare_reference_comparisons(sample: pd.DataFrame) -> pd.DataFrame:
    """Validate source conversations and use physical battle IDs for judging."""
    if {"conversation_a", "conversation_b"} & set(sample.columns):
        validate_conversations(sample)
    return sample.set_index("battle_id", drop=False).assign(
        instruction_id=sample["battle_id"].to_numpy(), cache_orientation="direct"
    )


def reference_scoring_battles(
    pool: pd.DataFrame, annotations: pd.DataFrame, *, swap_mode: str
) -> pd.DataFrame:
    """Join predictions to the full reference pool, retaining unsampled battles."""
    predictions = aggregate_battle_preferences(annotations, swap_mode=swap_mode)
    battles = pool[["battle_id", "model_a", "model_b", "reference_pref"]].copy()
    battles["sampled"] = battles["battle_id"].isin(predictions["battle_id"])
    return battles.merge(predictions, on="battle_id", how="left").sort_values(
        "battle_id", kind="stable", ignore_index=True
    )
