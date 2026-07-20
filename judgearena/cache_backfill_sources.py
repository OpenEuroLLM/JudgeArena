"""Extract backfill rows from saved GAE, MT-Bench, and meta-eval run folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from judgearena.cache_backfill_common import (
    chat_prompt_value,
    increment,
    is_backfillable_provider,
    mt_swapped,
    prompt_text,
    source_run_id,
)
from judgearena.cache_backfill_config import (
    build_gae_judge_model,
    build_meta_judge_model,
    build_mt_judge_model,
    load_gae_run_config,
    load_meta_args,
)
from judgearena.config import RunConfig, meta_eval_cache_task
from judgearena.evaluate import render_judge_inputs, resolve_run_judge_prompt
from judgearena.meta_eval.prompts import resolve_prompt_mode
from judgearena.model_adapters import PreparedModel


@dataclass(frozen=True)
class BackfillRow:
    task: str
    model_spec: str
    descriptor: dict[str, Any]
    canonical_input: str
    output_text: str
    row_metadata: dict[str, Any]
    producer_metadata: dict[str, Any]


@dataclass
class SourceExtraction:
    rows: list[BackfillRow]
    skipped: dict[str, int]
    source_kind: str


def _annotations_path(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("*-annotations.csv"))
    if len(paths) != 1:
        names = ", ".join(path.name for path in paths) or "none"
        raise ValueError(
            f"Expected exactly one *-annotations.csv in {run_dir.name}; found {names}."
        )
    return paths[0]


def _maybe_descriptor(model: PreparedModel) -> dict[str, Any] | None:
    return model.cache_descriptor()


def _infer_gae_orientation(row: pd.Series, *, cfg: RunConfig) -> str | None:
    model_a = str(row.get("model_A", ""))
    model_b = str(row.get("model_B", ""))
    focal = cfg.model.name
    a_is_focal = model_a == focal
    b_is_focal = model_b == focal
    if a_is_focal and not b_is_focal:
        return "direct"
    if b_is_focal and not a_is_focal:
        return "reversed"
    return None


def _meta_eval_verify_completions(row: pd.Series) -> tuple[str, str]:
    presented_a = prompt_text(row.get("presented_completion_a"))
    presented_b = prompt_text(row.get("presented_completion_b"))
    if presented_a is not None and presented_b is not None:
        return presented_a, presented_b
    completion_a = str(row.get("completion_a", ""))
    completion_b = str(row.get("completion_b", ""))
    if str(row.get("orientation", "forward")) == "swapped":
        return completion_b, completion_a
    return completion_a, completion_b


def extract_gae_rows(run_dir: Path) -> SourceExtraction:
    cfg = load_gae_run_config(run_dir)
    annotations_path = _annotations_path(run_dir)
    df = pd.read_csv(annotations_path, keep_default_na=False)
    skipped: dict[str, int] = {}
    rows: list[BackfillRow] = []
    run_id = source_run_id(run_dir)

    if not is_backfillable_provider(cfg.judge.model):
        increment(skipped, "local_engine_unsupported", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="gae")

    judge_model = build_gae_judge_model(cfg)
    descriptor = _maybe_descriptor(judge_model)
    if descriptor is None:
        increment(skipped, "local_engine_unsupported", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="gae")

    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
    output_column = (
        "judge_completion" if "judge_completion" in df.columns else "judge_output"
    )
    if output_column not in df.columns:
        increment(skipped, "judge_input_unverifiable", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="gae")

    if "judge_input" not in df.columns:
        increment(skipped, "judge_input_unverifiable", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="gae")

    rendered_inputs = render_judge_inputs(
        df["instruction"].astype(str).tolist(),
        df["completion_A"].astype(str).tolist(),
        df["completion_B"].astype(str).tolist(),
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        provide_explanation=cfg.judge.provide_explanation,
        prompt_preset=resolved_prompt.preset_name,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        task=cfg.task,
        system_file=cfg.judge.system_prompt_file,
        user_file=cfg.judge.user_prompt_file,
    )

    producer = judge_model.producer_metadata()

    for idx, (_, row) in enumerate(df.iterrows()):
        stored_input = row.get("judge_input")
        if pd.isna(stored_input) or stored_input is None:
            increment(skipped, "judge_input_unverifiable")
            continue
        rendered = rendered_inputs[idx].to_string()
        if str(stored_input) != rendered:
            increment(skipped, "judge_input_mismatch")
            continue
        output_text = prompt_text(row.get(output_column))
        if output_text is None:
            increment(skipped, "judge_output_missing")
            continue

        orientation = _infer_gae_orientation(row, cfg=cfg)
        if orientation is None:
            increment(skipped, "battle_orientation_unverifiable")
            continue
        prompt_input = rendered_inputs[idx]
        rows.append(
            BackfillRow(
                task=cfg.task,
                model_spec=judge_model.model_spec,
                descriptor=descriptor,
                canonical_input=judge_model.canonicalize_input(prompt_input),
                output_text=output_text,
                row_metadata={
                    "task": cfg.task,
                    "instruction_index": str(row.get("instruction_index", idx)),
                    "presented_model_a": str(row.get("model_A", cfg.model.name)),
                    "presented_model_b": str(row.get("model_B", "")),
                    "orientation": orientation,
                    "source_run_id": run_id,
                },
                producer_metadata=producer,
            )
        )

    return SourceExtraction(rows=rows, skipped=skipped, source_kind="gae")


def _mt_is_fastchat(df: pd.DataFrame) -> bool:
    return "g1_user_prompt" in df.columns


def extract_mt_bench_rows(run_dir: Path) -> SourceExtraction:
    cfg = load_gae_run_config(run_dir)
    annotations_path = _annotations_path(run_dir)
    df = pd.read_csv(annotations_path)
    skipped: dict[str, int] = {}
    rows: list[BackfillRow] = []
    run_id = source_run_id(run_dir)

    if not is_backfillable_provider(cfg.judge.model):
        increment(skipped, "local_engine_unsupported", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="mt_bench")

    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge, multi_turn=True)
    judge_model = build_mt_judge_model(cfg, delegated=resolved_prompt.delegated)
    descriptor = _maybe_descriptor(judge_model)
    if descriptor is None:
        increment(skipped, "local_engine_unsupported", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="mt_bench")

    producer = judge_model.producer_metadata()
    fastchat = _mt_is_fastchat(df)

    for _, row in df.iterrows():
        candidates: list[tuple[str, str | None, str, str]] = []
        if fastchat:
            g1_output = prompt_text(row.get("g1_judgment"))
            g1_prompt = prompt_text(row.get("g1_user_prompt"))
            if g1_output is not None and g1_prompt is not None:
                candidates.append(
                    (
                        "direct",
                        prompt_text(row.get("system_prompt")),
                        g1_prompt,
                        g1_output,
                    )
                )
            g2_output = prompt_text(row.get("g2_judgment"))
            g2_prompt = prompt_text(row.get("g2_user_prompt"))
            if g2_output is not None and g2_prompt is not None:
                candidates.append(
                    (
                        "reversed",
                        prompt_text(row.get("system_prompt")),
                        g2_prompt,
                        g2_output,
                    )
                )
        else:
            output = prompt_text(row.get("judge_completion"))
            if output is None:
                increment(skipped, "judge_input_unverifiable")
                continue
            orientation = "reversed" if mt_swapped(row.get("swapped")) else "direct"
            user_prompt = prompt_text(row.get("user_prompt"))
            if user_prompt is None:
                increment(skipped, "judge_input_unverifiable")
                continue
            candidates.append(
                (
                    orientation,
                    prompt_text(row.get("system_prompt")),
                    user_prompt,
                    output,
                )
            )

        for orientation, system_prompt, user_prompt, output_text in candidates:
            prompt_input = chat_prompt_value(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            turn_value = row.get("turn")
            rows.append(
                BackfillRow(
                    task=cfg.task,
                    model_spec=judge_model.model_spec,
                    descriptor=descriptor,
                    canonical_input=judge_model.canonicalize_input(prompt_input),
                    output_text=output_text,
                    row_metadata={
                        "question_id": str(row.get("question_id", "")),
                        "category": row.get("category"),
                        "turn": int(turn_value) if pd.notna(turn_value) else None,
                        "orientation": orientation,
                        "prompt": row.get("prompt_name"),
                        "source_run_id": run_id,
                    },
                    producer_metadata=producer,
                )
            )

    return SourceExtraction(rows=rows, skipped=skipped, source_kind="mt_bench")


def extract_meta_eval_rows(run_dir: Path) -> SourceExtraction:
    args = load_meta_args(run_dir)
    df = pd.read_parquet(run_dir / "annotations.parquet")
    skipped: dict[str, int] = {}
    rows: list[BackfillRow] = []
    run_id = source_run_id(run_dir)

    if not is_backfillable_provider(args.judge_model):
        increment(skipped, "local_engine_unsupported", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="meta_eval")

    prompt_spec = resolve_prompt_mode(
        args.prompt_mode,
        provide_explanation=args.provide_explanation,
    )
    judge_model = build_meta_judge_model(args)
    descriptor = _maybe_descriptor(judge_model)
    if descriptor is None:
        increment(skipped, "local_engine_unsupported", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="meta_eval")

    if "judge_input" not in df.columns:
        increment(skipped, "judge_input_unverifiable", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="meta_eval")

    output_column = (
        "judge_completion" if "judge_completion" in df.columns else "judge_output"
    )
    if output_column not in df.columns:
        increment(skipped, "judge_input_unverifiable", len(df))
        return SourceExtraction(rows=[], skipped=skipped, source_kind="meta_eval")

    verify_a: list[str] = []
    verify_b: list[str] = []
    for _, row in df.iterrows():
        completion_a, completion_b = _meta_eval_verify_completions(row)
        verify_a.append(completion_a)
        verify_b.append(completion_b)

    rendered_inputs = render_judge_inputs(
        df["instruction"].astype(str).tolist(),
        verify_a,
        verify_b,
        system_prompt=prompt_spec.system_prompt,
        user_prompt_template=prompt_spec.user_prompt_template,
        truncate_input_chars=args.truncate_judge_input_chars,
        provide_explanation=args.provide_explanation,
    )

    task = meta_eval_cache_task(args.reference_arena)
    producer = judge_model.producer_metadata()

    for idx, (_, row) in enumerate(df.iterrows()):
        stored_input = row.get("judge_input")
        if pd.isna(stored_input) or stored_input is None:
            increment(skipped, "judge_input_unverifiable")
            continue
        rendered = rendered_inputs[idx].to_string()
        if str(stored_input) != rendered:
            increment(skipped, "judge_input_mismatch")
            continue
        output_text = prompt_text(row.get(output_column))
        if output_text is None:
            increment(skipped, "judge_output_missing")
            continue

        orientation = str(row.get("orientation", "forward"))
        prompt_input = rendered_inputs[idx]
        rows.append(
            BackfillRow(
                task=task,
                model_spec=judge_model.model_spec,
                descriptor=descriptor,
                canonical_input=judge_model.canonicalize_input(prompt_input),
                output_text=output_text,
                row_metadata={
                    "reference_arena": args.reference_arena,
                    "benchmark": str(row.get("benchmark", "")),
                    "question_id": str(row.get("question_id", "")),
                    "presented_model_a": str(
                        row.get("presented_model_a", row.get("model_a", ""))
                    ),
                    "presented_model_b": str(
                        row.get("presented_model_b", row.get("model_b", ""))
                    ),
                    "prompt_mode": args.prompt_mode,
                    "orientation": orientation,
                    "source_run_id": run_id,
                },
                producer_metadata=producer,
            )
        )

    return SourceExtraction(rows=rows, skipped=skipped, source_kind="meta_eval")
