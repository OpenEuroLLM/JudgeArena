"""Deterministic arena battle sampling for judge meta-evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd


class MetaEvalSamplingError(ValueError):
    """Raised when filtering or sampling yields an unusable battle subset."""


def count_battles_per_model(df: pd.DataFrame) -> dict[str, int]:
    """Count unique physical rows incident to each model."""
    model_a_counts = df["model_a"].value_counts()
    model_b_counts = df["model_b"].value_counts()
    return model_a_counts.add(model_b_counts, fill_value=0).astype(int).to_dict()


def comparison_components(
    df: pd.DataFrame, models: Iterable[str] | None = None
) -> list[frozenset[str]]:
    """Return connected components of the model comparison graph."""
    model_set = set(models or ())
    model_set.update(df["model_a"].dropna().astype(str))
    model_set.update(df["model_b"].dropna().astype(str))
    adjacency = {model: set() for model in model_set}
    for model_a, model_b in df[["model_a", "model_b"]].itertuples(
        index=False, name=None
    ):
        model_a, model_b = str(model_a), str(model_b)
        adjacency[model_a].add(model_b)
        adjacency[model_b].add(model_a)

    components: list[frozenset[str]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        stack = [start]
        component = set()
        while stack:
            model = stack.pop()
            if model in component:
                continue
            component.add(model)
            stack.extend(adjacency[model] - component)
        unseen -= component
        components.append(frozenset(component))
    return sorted(components, key=lambda component: sorted(component))


def require_connected_pool(
    df: pd.DataFrame, models: Iterable[str], *, context: str
) -> None:
    """Reject comparison pools whose Bradley-Terry offsets are unidentified."""
    components = comparison_components(df, models)
    if len(components) != 1:
        rendered = [sorted(component) for component in components]
        raise MetaEvalSamplingError(
            f"{context} comparison graph is disconnected: {rendered}."
        )


def select_top_models(
    df: pd.DataFrame, *, top_models: int
) -> tuple[list[str], pd.DataFrame]:
    """Pick the most-battled models and require one connected induced pool."""
    if (df["model_a"] == df["model_b"]).any():
        raise MetaEvalSamplingError("Meta-evaluation does not allow self-comparisons.")
    battle_counts = count_battles_per_model(df)
    if not battle_counts:
        raise MetaEvalSamplingError("Cannot select top models from an empty dataframe.")

    top = sorted(battle_counts, key=lambda model: (-battle_counts[model], str(model)))[
        :top_models
    ]
    if len(top) != top_models:
        raise MetaEvalSamplingError(
            f"Requested {top_models} top models, but only {len(top)} are available."
        )
    top_set = set(top)
    df_top = df[df["model_a"].isin(top_set) & df["model_b"].isin(top_set)].copy()
    require_connected_pool(df_top, top, context="Top-model")
    return top, df_top


def _stable_priority(battle_id: object, *, seed: int) -> str:
    payload = f"{seed}\0{battle_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _spanning_battle_ids(df: pd.DataFrame, top_models: list[str]) -> set[object]:
    """Choose a stable-priority spanning tree before filling quotas."""
    parent = {model: model for model in top_models}

    def find(model: str) -> str:
        while parent[model] != model:
            parent[model] = parent[parent[model]]
            model = parent[model]
        return model

    def union(model_a: str, model_b: str) -> bool:
        root_a, root_b = find(model_a), find(model_b)
        if root_a == root_b:
            return False
        parent[root_b] = root_a
        return True

    selected: set[object] = set()
    for row in df.itertuples(index=False):
        if union(str(row.model_a), str(row.model_b)):
            selected.add(row.battle_id)
            if len(selected) == len(top_models) - 1:
                break
    return selected


def sample_battles_per_model(
    df_top: pd.DataFrame,
    top_models: list[str],
    *,
    battles_per_model: int,
    seed: int,
) -> pd.DataFrame:
    """Build one connected panel with no repeated physical battles.

    A spanning tree guarantees connectivity. Already-selected battles count
    toward both endpoint quotas; remaining unseen battles fill each model's
    target deterministically. Connectivity can make some models exceed the
    target, which is preferable to an unidentified ranking graph.
    """
    working = df_top.copy()
    working["_sample_priority"] = working["battle_id"].map(
        lambda battle_id: _stable_priority(battle_id, seed=seed)
    )
    working = working.sort_values(
        ["_sample_priority", "battle_id"], kind="stable"
    ).reset_index(drop=True)
    available_counts = count_battles_per_model(working)
    shortfalls = {
        model: available_counts.get(model, 0)
        for model in top_models
        if available_counts.get(model, 0) < battles_per_model
    }
    if shortfalls:
        raise MetaEvalSamplingError(
            "Insufficient unique battles for the requested per-model quota: "
            f"{shortfalls}."
        )
    selected_ids = _spanning_battle_ids(working, top_models)

    for model in top_models:
        incident = (working["model_a"] == model) | (working["model_b"] == model)
        already_selected = working["battle_id"].isin(selected_ids)
        current = int((incident & already_selected).sum())
        needed = max(0, battles_per_model - current)
        candidates = working[incident & ~already_selected]
        if needed:
            selected_ids.update(candidates.head(needed)["battle_id"].tolist())

    sample = working[working["battle_id"].isin(selected_ids)].copy()

    return sample.drop(columns="_sample_priority").reset_index(drop=True)
