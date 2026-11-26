"""Select graph comparisons and pair candidate answers with saved opponents."""

import hashlib
import json

import numpy as np
import pandas as pd

from judgearena.battles import swap_comparison_slots
from judgearena.benchmarks.pairwise.comparisons import (
    assemble_comparisons,
    conversation_comparisons,
    select_rows,
)
from judgearena.tasks.schema import NoBaseline, RecordedOpponents


def validate_graph_task(protocol):
    """Graph preparation supplies opponents rather than a task baseline."""
    if protocol.runner != "pairwise":
        raise ValueError(
            "Graph releases require a task that can generate and compare answers"
        )
    if protocol.baseline is not None and not isinstance(
        protocol.baseline, (RecordedOpponents, NoBaseline)
    ):
        raise ValueError(
            "A supplied opponent panel requires baseline.strategy recorded_opponents or none"
        )


def prepare_candidate_comparisons(panel, answers, *, rng, turns=None):
    """Return graph comparisons and routing IDs, matching baseline preparation."""
    answers = answers.reindex(panel["instruction_index"]).set_axis(panel.index)
    opponents = panel.filter(regex="_b$").rename(columns=lambda c: c[:-2])
    comparisons = assemble_comparisons(panel, answers, opponents)
    comparisons = comparisons.assign(
        instruction_id=panel.get(
            "instruction_id", panel["instruction_index"].astype(str)
        ),
        cache_orientation="direct",
        question_id=panel.get("question_id", panel["instruction_index"]),
        source="llm-judge",
    )
    if rng is not None:
        candidate_is_a = rng.choice([True, False], size=len(panel))
        comparisons["cache_orientation"] = np.where(
            candidate_is_a, "direct", "reversed"
        )
        comparisons = swap_comparison_slots(comparisons, ~candidate_is_a)
    comparisons = conversation_comparisons(comparisons, turns)
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


def _sampling_metadata(sampled, sampling, seed):
    if sampling.method == "first":
        return {"sampling_mode": "head"}
    return {
        "sampling_mode": "seeded_random",
        "random_seed": seed,
        "requested_rows": sampling.limit,
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


def prepare_recorded_opponents(cfg, data, *, rng):
    """Select one recorded opponent for each sampled battle."""
    from judgearena.datasets import resolve_dataset_filters
    from judgearena.datasets.data import PreferenceData

    if not isinstance(data, PreferenceData):
        raise ValueError("Recorded opponents require PreferenceData")
    filtered = data.select(resolve_dataset_filters(cfg.resolved_task, cfg.run.filters))
    battles = select_rows(
        filtered.battles,
        cfg.run.sampling,
        seed=cfg.run.seed,
        limit_by=cfg.resolved_task.spec.protocol.limit_by,
    )
    metadata = _sampling_metadata(battles, cfg.run.sampling, cfg.run.seed)
    battles = battles.reset_index(drop=True)
    # Keep the selected opponent in slot B until the candidate answer is available.
    panel = swap_comparison_slots(battles, rng.choice([True, False], size=len(battles)))
    panel = panel.drop(columns=panel.filter(regex="_a$").columns)
    return panel, metadata


def recorded_references(data):
    """Keep labels on the original pairs, independently of the candidate panel."""
    references = data.battles.drop(
        columns=["pref", "pref_hard"], errors="ignore"
    ).copy()
    if "reference_pref" not in references:
        references["reference_pref"] = float("nan")
    references = references.loc[references.reference_pref.notna()].copy()
    references["evaluation_model"] = None
    return references
