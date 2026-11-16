"""Select graph comparisons and pair candidate answers with saved opponents."""

import hashlib
import json

import numpy as np
import pandas as pd

from judgearena.battles import swap_comparison_slots


def validate_graph_task(protocol):
    """Graph preparation supplies opponents rather than a task baseline."""
    if not hasattr(protocol, "generation") or not hasattr(protocol.judge, "turns"):
        raise ValueError(
            "Graph releases require a task that can generate and compare answers"
        )
    if protocol.baseline.strategy != "none":
        raise ValueError(
            "Graph preparation requires protocol.baseline.strategy: none; the panel supplies opponents"
        )


def select_graph_battles(
    df_battles, *, seed, random_battles=None, group_by=None, per_group=None, limit=None
):
    """Select comparison rows without choosing a metric or fitting ratings."""
    random_sampling = random_battles is not None
    sampling_metadata: dict[str, object] = {"sampling_mode": "head"}
    if random_sampling:
        if limit is not None or per_group is not None:
            raise ValueError(
                "generation.n_instructions and graph.per_group cannot be combined "
                "with graph.random_battles."
            )
        df_battles, sampling_metadata = select_seeded_random_battles(
            df_battles,
            n_battles=random_battles,
            seed=seed,
        )
    else:
        if per_group is not None:
            df_battles = (
                df_battles.groupby(group_by).head(per_group).reset_index(drop=True)
            )

        # Keep at most n_instructions total (subset used for LLM-judge evaluation)
        if limit is not None:
            df_battles = df_battles.head(limit)

    df_battles = df_battles.reset_index(drop=True)

    return df_battles, sampling_metadata


def prepare_candidate_comparisons(battles, answers, *, rng, turns=None):
    """Return graph comparisons and routing IDs, matching baseline preparation."""
    from judgearena.benchmarks.pairwise.comparisons import (
        assemble_comparisons,
        conversation_comparisons,
    )

    answers = answers.reindex(battles["instruction_index"]).set_axis(battles.index)

    use_model_a = rng.choice([True, False], size=len(battles))
    candidate_is_a = rng.choice([True, False], size=len(battles))
    context = swap_comparison_slots(battles, use_model_a)
    opponents = context.filter(regex="_b$").rename(columns=lambda c: c[:-2])
    comparisons = assemble_comparisons(context, answers, opponents)
    comparisons = comparisons.assign(
        instruction_id=battles.get(
            "instruction_id", battles["instruction_index"].astype(str)
        ),
        cache_orientation=np.where(candidate_is_a, "direct", "reversed"),
        question_id=battles.get("question_id", battles["instruction_index"]),
        source="llm-judge",
    )
    comparisons = conversation_comparisons(
        swap_comparison_slots(comparisons, ~candidate_is_a), turns
    )
    source_ids = comparisons["source_instruction_index"] if turns is not None else None
    return comparisons, source_ids


def _sample_fingerprint(sampled: pd.DataFrame) -> str:
    rows = [
        {
            "index": int(index) if isinstance(index, int | np.integer) else str(index),
            "question_id": str(row.get("question_id", row.get("instruction_index"))),
            "model_a": str(row["model_a"]),
            "model_b": str(row["model_b"]),
        }
        for index, row in sampled.iterrows()
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def select_seeded_random_battles(
    df_battles: pd.DataFrame,
    *,
    n_battles: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select a reproducible comparison panel and record its sampling identities."""
    sampled = df_battles.sample(
        n=min(n_battles, len(df_battles)), random_state=seed, replace=False
    )
    metadata: dict[str, object] = {
        "sampling_mode": "seeded_random",
        "random_seed": seed,
        "requested_rows": n_battles,
        "sampled_rows": len(sampled),
        "sampled_original_indices": [
            int(index) if isinstance(index, int | np.integer) else str(index)
            for index in sampled.index
        ],
        "sampled_question_ids": [
            str(value)
            for value in sampled[
                "question_id" if "question_id" in sampled else "instruction_index"
            ].tolist()
        ],
        "sample_fingerprint": _sample_fingerprint(sampled),
    }
    return sampled.reset_index(drop=True), metadata


def prepare_graph(cfg, data):
    """Select candidate comparisons and retain available reference labels."""
    from judgearena.datasets.data import PreferenceData

    validate_graph_task(cfg.resolved_task.spec.protocol)
    if not isinstance(data, PreferenceData):
        raise ValueError("Graph preparation requires PreferenceData")
    from judgearena.datasets import resolve_dataset_filters

    sampling = cfg.graph
    battles, metadata = select_graph_battles(
        data.select(
            resolve_dataset_filters(cfg.resolved_task, cfg.run.filters)
        ).battles,
        seed=cfg.run.seed,
        random_battles=sampling.random_battles,
        group_by=sampling.group_by,
        per_group=sampling.per_group,
        limit=cfg.generation.n_instructions,
    )
    references = data.battles.drop(
        columns=["pref", "pref_hard"], errors="ignore"
    ).copy()
    if "reference_pref" not in references:
        references["reference_pref"] = float("nan")
    references = references.loc[references.reference_pref.notna()].copy()
    references["evaluation_model"] = None
    return battles, references, metadata
