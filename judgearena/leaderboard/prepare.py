"""Selection helpers shared by release curation and judging."""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from judgearena.battles import has_support
from judgearena.benchmarks.pairwise.judging import (
    judge_configured_comparisons as judge_comparisons,  # noqa: F401
)


def comparison_groups(frame):
    """Keep sessions and identical prompt texts together, including transitive links."""
    parents = list(range(len(frame)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for column in ("session", "instruction"):
        if column not in frame:
            continue
        seen = {}
        for i, value in enumerate(frame[column]):
            if pd.isna(value):
                if column == "session":
                    continue
                raise ValueError(f"Missing {column} for split grouping")
            if value in seen:
                parents[root(i)] = root(seen[value])
            else:
                seen[value] = i
    return pd.Series([root(i) for i in range(len(frame))], index=frame.index)


def _largest_component(frame):
    """Keep comparisons among the largest strongly connected set of models."""
    names = sorted(set(frame.model_a) | set(frame.model_b))
    lookup = {name: i for i, name in enumerate(names)}
    a, b = frame.model_a.map(lookup), frame.model_b.map(lookup)
    y = frame.reference_pref
    src = np.r_[a[y < 1], b[y > 0]]
    dst = np.r_[b[y < 1], a[y > 0]]
    graph = coo_matrix(
        (np.ones(len(src)), (src, dst)), shape=(len(names), len(names))
    ).tocsr()
    _, labels = connected_components(graph, directed=True, connection="strong")
    largest = np.bincount(labels).argmax()
    keep = {name for name, label in zip(names, labels, strict=True) if label == largest}
    return frame[frame.model_a.isin(keep) & frame.model_b.isin(keep)]


def reference_pool(frame, minimum, *, connected=True):
    """Prune low-support models and, by default, disconnected ones until stable."""
    while not frame.empty:
        before = len(frame)
        frame = frame[has_support(frame, minimum)]
        if connected and not frame.empty:
            frame = _largest_component(frame)
        if len(frame) == before:
            return frame.copy()
    raise ValueError(
        "No supported connected reference graph remains"
        if connected
        else "no models meet the minimum comparison count"
    )


def sample_battles(
    battles, settings, *, seed, baseline_models, languages=None, require_connected=True
):
    """Select a supported graph and fixed panel without inspecting judge labels."""
    sampling = settings.sampling
    valid = battles.model_a.ne(battles.model_b)
    for column in ("instruction", "model_a", "model_b", "completion_a", "completion_b"):
        valid &= battles[column].map(lambda x: isinstance(x, str) and bool(x.strip()))
    if settings.labels == "human":
        if "reference_pref" not in battles:
            raise ValueError("Human releases require reference_pref")
        valid &= battles.reference_pref.isin([0, 0.5, 1])
    battles = battles.loc[valid].sort_values("battle_id").reset_index(drop=True).copy()
    battles["split_group"] = comparison_groups(battles)
    split_rng, cap_rng, panel_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    groups = np.sort(battles.split_group.unique())
    heldout = split_rng.choice(
        groups, size=round(len(groups) * sampling.validation_fraction), replace=False
    )
    languages = languages if languages is not None else sorted(battles.lang.unique())
    validation = battles[battles.split_group.isin(heldout)].copy()
    training = battles[~battles.split_group.isin(heldout)].copy()
    references, panels, eligible = [], [], []
    for lang in languages:
        available = training[training.lang == lang]
        if sampling.max_reference_comparisons is not None:
            clusters = list(
                available.groupby("split_group", sort=True).indices.values()
            )
            positions, count = [], 0
            for i in cap_rng.permutation(len(clusters)):
                group = clusters[i]
                if count + len(group) <= sampling.max_reference_comparisons:
                    positions.extend(group)
                    count += len(group)
            available = available.iloc[positions]
        # Before judging, only topology is known. Do not select using human votes.
        graph = (
            available
            if settings.labels == "human"
            else available.assign(reference_pref=0.5)
        )
        try:
            pool = reference_pool(
                graph, sampling.minimum_model_comparisons, connected=require_connected
            )
        except ValueError as error:
            raise ValueError(f"{lang}: {error}") from error
        reference = available.loc[pool.index].copy()
        models = set(reference.model_a) | set(reference.model_b)
        missing = {*(sampling.opponents or []), *baseline_models} - models
        if missing:
            raise ValueError(
                f"{lang}: models absent after sampling/support checks: {sorted(missing)}"
            )
        references.append(reference)
        eligible.append(models)
    opponents = sampling.opponents
    if opponents is None:
        opponents = sorted(set.intersection(*eligible)) if eligible else []
    if not opponents:
        raise ValueError("No common eligible opponents across the selected languages")
    for lang, reference in zip(languages, references, strict=True):
        used = set()
        for opponent in opponents:
            candidates = reference[
                (reference.model_a == opponent) | (reference.model_b == opponent)
            ]
            candidates = candidates.sample(
                frac=1, random_state=int(panel_rng.integers(2**31))
            )
            selected = (
                candidates[~candidates.split_group.isin(used)]
                .drop_duplicates("split_group")
                .head(sampling.comparisons_per_opponent)
                .copy()
            )
            if len(selected) != sampling.comparisons_per_opponent:
                raise ValueError(f"{lang}: insufficient distinct groups for {opponent}")
            selected["opponent_model"] = opponent
            selected["opponent_slot"] = np.where(selected.model_a == opponent, "a", "b")
            used.update(selected.split_group)
            panels.append(selected)
    return pd.concat(references), pd.concat(panels), validation


def reference_intervals(scoring, *, baseline, bootstraps, seed):
    """Resample whole prompt groups, rejecting unidentified bootstrap fits."""
    from judgearena.benchmarks.elo.rating import fit_bradley_terry, resample_clusters

    models = set(scoring.model_a) | set(scoring.model_b)
    scoring = scoring[["model_a", "model_b", "reference_pref", "split_group"]]
    clusters = list(scoring.groupby("split_group", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    samples = []
    for positions in resample_clusters(clusters, bootstraps, rng):
        sample = scoring.iloc[positions]
        try:
            connected = reference_pool(sample, 1)
        except ValueError:
            continue
        if (
            len(connected) != len(sample)
            or (set(sample.model_a) | set(sample.model_b)) != models
        ):
            continue
        samples.append(
            fit_bradley_terry(
                sample, pref_col="reference_pref", baseline_model=baseline
            )
        )
    if len(samples) < bootstraps * 0.9:
        raise ValueError(
            "Too many disconnected reference bootstrap replicates; increase graph support"
        )
    return pd.DataFrame(samples)
