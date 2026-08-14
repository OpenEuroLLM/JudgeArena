"""Deterministic arena battle sampling for judge meta-evaluation."""

from __future__ import annotations

import pandas as pd

from judgearena.log import get_logger

logger = get_logger(__name__)


class MetaEvalSamplingError(ValueError):
    """Raised when filtering or sampling yields an unusable battle subset."""


def normalize_human_winner(winner: object) -> str:
    """Collapse the arena's tie labels ("tie", "tie (bothbad)") into "tie"."""
    text = str(winner)
    return "tie" if "tie" in text else text


def count_battles_per_model(df: pd.DataFrame) -> dict[str, int]:
    return (
        pd.concat([df["model_a"], df["model_b"]], ignore_index=True)
        .value_counts()
        .to_dict()
    )


def select_top_models(
    df: pd.DataFrame, *, top_models: int
) -> tuple[list[str], pd.DataFrame]:
    """Pick the most-battled models and keep only battles fought among them.

    Restricting to intra-group battles keeps every judged comparison inside a
    single connected pool, which is what makes the resulting annotations usable
    for a ranking fit later on.
    """
    battle_counts = count_battles_per_model(df)
    if not battle_counts:
        raise MetaEvalSamplingError("Cannot select top models from an empty dataframe.")

    top = sorted(battle_counts, key=battle_counts.__getitem__, reverse=True)[
        :top_models
    ]
    top_set = set(top)
    df_top = df[df["model_a"].isin(top_set) & df["model_b"].isin(top_set)].copy()
    if df_top.empty:
        raise MetaEvalSamplingError(
            f"No battles remain among the top {top_models} models."
        )
    return top, df_top


def sample_battles_per_model(
    df_top: pd.DataFrame,
    top_models: list[str],
    *,
    battles_per_model: int,
    seed: int,
) -> pd.DataFrame:
    """Draw up to ``battles_per_model`` battles for each selected model.

    Sampling per model rather than globally keeps thin models represented. A
    battle between two selected models can be drawn for either of them, so the
    result may repeat a battle.
    """
    per_model_samples: list[pd.DataFrame] = []
    for model_index, model in enumerate(top_models):
        df_model = df_top[(df_top["model_a"] == model) | (df_top["model_b"] == model)]
        if df_model.empty:
            logger.warning("Model %s has no battles among top models; skipping.", model)
            continue
        per_model_samples.append(
            df_model.sample(
                n=min(battles_per_model, len(df_model)),
                replace=False,
                random_state=seed + model_index,
            )
        )

    if not per_model_samples:
        raise MetaEvalSamplingError(
            "Sampling produced no battles; reduce top_models or battles_per_model."
        )
    return pd.concat(per_model_samples, ignore_index=True)
