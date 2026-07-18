"""Deterministic arena battle sampling for meta-evaluation."""

from __future__ import annotations

import pandas as pd

from judgearena.arenas_utils import KNOWN_ARENAS, load_arena_dataframe
from judgearena.log import get_logger

logger = get_logger(__name__)


class MetaEvalSamplingError(ValueError):
    """Raised when filtering or sampling yields an unusable subset."""


def normalize_human_winner(winner: object) -> str:
    text = str(winner)
    if "tie" in text:
        return "tie"
    return text


def count_battles_per_model(df: pd.DataFrame) -> dict[str, int]:
    return (
        pd.concat([df["model_a"], df["model_b"]], ignore_index=True)
        .value_counts()
        .to_dict()
    )


def load_reference_arena_battles(
    reference_arena: str,
    *,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    if reference_arena not in KNOWN_ARENAS:
        raise MetaEvalSamplingError(
            f"Unsupported reference arena {reference_arena!r}; "
            f"expected one of {KNOWN_ARENAS}."
        )

    df = load_arena_dataframe(arena=reference_arena).copy()
    df["winner"] = df["winner"].map(normalize_human_winner)

    if languages:
        df = df[df["lang"].isin(languages)].copy()
        if df.empty:
            langs = ", ".join(languages)
            raise MetaEvalSamplingError(
                f"No battles remain after filtering to languages: {langs}."
            )

    if df.empty:
        raise MetaEvalSamplingError(
            f"No battles found for reference arena {reference_arena!r}."
        )
    return df


def select_top_models(
    df: pd.DataFrame,
    *,
    top_models: int,
) -> tuple[list[str], pd.DataFrame]:
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
    per_model_samples: list[pd.DataFrame] = []
    for model_index, model in enumerate(top_models):
        model_mask = (df_top["model_a"] == model) | (df_top["model_b"] == model)
        df_model = df_top[model_mask]
        if df_model.empty:
            logger.warning("Model %s has no battles among top models; skipping.", model)
            continue
        sample_size = min(battles_per_model, len(df_model))
        sampled = df_model.sample(
            n=sample_size,
            replace=False,
            random_state=seed + model_index,
        )
        per_model_samples.append(sampled)

    if not per_model_samples:
        raise MetaEvalSamplingError(
            "Sampling produced no battles; reduce top_models or battles_per_model."
        )

    df_sample = pd.concat(per_model_samples, ignore_index=True)
    if df_sample.empty:
        raise MetaEvalSamplingError("Sampled battle set is empty.")
    return df_sample
