"""Registered MT-Bench runner and evaluation pipeline.

Orchestrates multi-turn generation, FastChat-compatible pairwise judging,
and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    write_run_metadata_safely,
)
from judgearena.benchmarks.execution import (
    build_completion_cache,
    build_generation_kwargs,
    build_judgement_cache,
)
from judgearena.benchmarks.mt_bench.fastchat_compat import (
    judge_mt_bench_pairwise_fastchat,
)
from judgearena.benchmarks.mt_bench.preset_judging import judge_mt_bench_with_preset
from judgearena.benchmarks.pairwise.baselines import resolve_baseline_plan
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_task_data, resolve_dataset_filters
from judgearena.datasets.data import CompletionData, _require_columns
from judgearena.generate import generate_multiturn
from judgearena.log import get_logger
from judgearena.models import prepare_model
from judgearena.prompts.registry import ResolvedJudgePrompt, resolve_run_judge_prompt
from judgearena.reports import BattleReport
from judgearena.tasks.schema import MTBenchProtocol

logger = get_logger(__name__)

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.tasks.schema import ResolvedTaskSpec


def _align_mt_bench_completions(
    *, questions_df: pd.DataFrame, completions: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Align cached or generated MT-Bench completions to the question order."""
    indexed = completions.set_index("instruction_index")
    _require_columns(
        indexed, {"completion_turn_1", "completion_turn_2"}, "Two-turn completions"
    )
    missing_ids = questions_df.index.difference(indexed.index)
    if not missing_ids.empty:
        missing_ids_preview = ", ".join(str(x) for x in missing_ids[:5])
        raise ValueError(
            f"MT-Bench completions for '{model_name}' are missing "
            f"{len(missing_ids)} question(s). First missing ids: {missing_ids_preview}."
        )
    return indexed.loc[questions_df.index]


def _generate_mt_bench_completions(
    cfg: RunConfig,
    task: ResolvedTaskSpec,
    questions_df: pd.DataFrame,
    task_data: CompletionData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = task.spec.protocol

    def _run_generation(
        model_name: str, *, generation_kwargs: dict[str, object]
    ) -> pd.DataFrame:
        # MT-Bench's category-aware temperatures only kick in when the user has
        # not explicitly pinned a per-role temperature; otherwise the config
        # override should win for reproducibility.
        temperature_config = (
            None
            if "temperature" in generation_kwargs
            else dict(protocol.generation.category_temperatures)
        )
        return generate_multiturn(
            questions=questions_df,
            model=model_name,
            truncate_input_chars=cfg.generation.truncate_all_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            temperature_config=temperature_config,
            strip_thinking_before_turn_2_prompt=cfg.judge.strip_thinking_before_judging,
            inference_cache=build_completion_cache(cfg),
            **generation_kwargs,
        )

    def _load_or_generate(model_name: str, *, role: Literal["A", "B"]) -> pd.DataFrame:
        loaded_answers = task_data.model_outputs_for(model_name)
        if loaded_answers is not None:
            return _align_mt_bench_completions(
                questions_df=questions_df,
                completions=loaded_answers,
                model_name=model_name,
            )
        generation_kwargs = build_generation_kwargs(cfg, model_name, role=role)
        generated_answers = _run_generation(
            model_name,
            generation_kwargs=generation_kwargs,
        )
        return _align_mt_bench_completions(
            questions_df=questions_df,
            completions=generated_answers,
            model_name=model_name,
        )

    return _load_or_generate(cfg.model.name, role="A"), _load_or_generate(
        cfg.model.baseline, role="B"
    )


def _build_mt_bench_input_payloads(
    *,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
) -> dict[str, object]:
    return {
        "instruction_index": questions_df.index.tolist(),
        "turn_1": questions_df["turn_1"].tolist(),
        "turn_2": questions_df["turn_2"].tolist(),
        "completion_turn_1_A": completions_a["completion_turn_1"].tolist(),
        "completion_turn_2_A": completions_a["completion_turn_2"].tolist(),
        "completion_turn_1_B": completions_b["completion_turn_1"].tolist(),
        "completion_turn_2_B": completions_b["completion_turn_2"].tolist(),
    }


def _save_mt_bench_results(
    *,
    cfg: RunConfig,
    res_folder: Path,
    result_name: str,
    results: dict[str, object],
    annotations_df: pd.DataFrame,
    started_at_utc: datetime,
    input_payloads: dict[str, object],
    judge_system_prompt: str | None = None,
    judge_user_prompt_template: str | None = None,
) -> None:
    """Persist MT-Bench arguments, annotations, aggregate results, and metadata."""
    annotations_df.to_csv(res_folder / f"{result_name}-annotations.csv", index=False)

    write_run_metadata_safely(
        task=cfg.resolved_task,
        output_dir=res_folder,
        entrypoint="judgearena.benchmarks.mt_bench.runner.run_mt_bench_benchmark",
        run=cfg.model_dump(),
        results=results,
        input_payloads=input_payloads,
        judge_system_prompt=judge_system_prompt,
        judge_user_prompt_template=judge_user_prompt_template,
        started_at_utc=started_at_utc,
    )


def _build_mt_bench_battles(
    *,
    cfg: RunConfig,
    prefs: pd.Series,
    combined_metadata: list[dict[str, object]],
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical metric table for judged MT-Bench turns."""
    rows: list[dict[str, object]] = []
    for metadata, pref in zip(combined_metadata, prefs, strict=True):
        question_id = metadata["question_id"]
        turn = int(metadata["turn"])
        completion_column = f"completion_turn_{turn}"
        pref_hard = pref
        if pref < 0.5:
            pref_hard = 0.0
        elif pref > 0.5:
            pref_hard = 1.0
        rows.append(
            {
                **metadata,
                "instruction_index": f"{question_id}:turn-{turn}",
                "model": cfg.model.name,
                "baseline": cfg.model.baseline,
                "completion_model": completions_a.loc[question_id, completion_column],
                "completion_baseline": completions_b.loc[
                    question_id, completion_column
                ],
                "model_a": cfg.model.name,
                "model_b": cfg.model.baseline,
                "completion_a": completions_a.loc[question_id, completion_column],
                "completion_b": completions_b.loc[question_id, completion_column],
                "evaluation_model": cfg.model.name,
                "source": "llm-judge",
                "orientation": metadata.get("orientation", "single"),
                "judge": cfg.judge.model,
                "pref": pref,
                "pref_hard": pref_hard,
            }
        )
    return pd.DataFrame(rows)


def _finalize_mt_bench_run(
    *,
    cfg: RunConfig,
    protocol: MTBenchProtocol,
    res_folder: Path,
    result_name: str,
    prefs: pd.Series,
    annotations: list[dict[str, object]],
    combined_metadata: list[dict[str, object]],
    resolved_prompt: ResolvedJudgePrompt,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    started_at_utc: datetime,
    extra_result_fields: dict[str, object] | None = None,
) -> pd.Series:
    battles = _build_mt_bench_battles(
        cfg=cfg,
        prefs=pd.Series(prefs, dtype="float64"),
        combined_metadata=combined_metadata,
        completions_a=completions_a,
        completions_b=completions_b,
    )
    metrics = build_metrics(protocol.scoring.metrics)
    metric_results = calculate_metrics(
        battles,
        metrics,
        runtime_by_metric={
            "bradley_terry": {"rng": np.random.default_rng(cfg.run.seed)},
        },
    )
    report = BattleReport(
        task=cfg.task,
        model_a=cfg.model.name,
        model_b=cfg.model.baseline,
        judge_model=cfg.judge.model,
        metrics=metric_results,
        preferences=prefs.tolist(),
        metadata={
            **resolved_prompt.metadata(),
            "battle_thinking_token_budget": cfg.judge.battle_thinking_token_budget,
            "strip_thinking_before_judging": cfg.judge.strip_thinking_before_judging,
            **(extra_result_fields or {}),
            "date": datetime.now(UTC).isoformat(),
            "user": os.getenv("USER", ""),
        },
    )
    results = report.to_dict()
    report.render()
    report.save(res_folder / f"results-{result_name}.json")
    _save_mt_bench_results(
        cfg=cfg,
        res_folder=res_folder,
        result_name=result_name,
        results=results,
        annotations_df=pd.DataFrame(annotations),
        started_at_utc=started_at_utc,
        input_payloads=_build_mt_bench_input_payloads(
            questions_df=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
        ),
        judge_system_prompt=resolved_prompt.system_prompt,
        judge_user_prompt_template=resolved_prompt.user_prompt_template,
    )
    return prefs


def _run_mt_bench_fastchat(
    *,
    cfg: RunConfig,
    protocol: MTBenchProtocol,
    res_folder: Path,
    result_name: str,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
    started_at_utc: datetime,
) -> pd.Series:
    prefs, annotations, combined_metadata, num_inconsistent = (
        judge_mt_bench_pairwise_fastchat(
            judge_chat_model=judge_chat_model,
            judge_model=cfg.judge.model,
            questions=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            model_a=cfg.model.name,
            model_b=cfg.model.baseline,
            turns_mode=protocol.judge.turns_mode,
            swap_mode=cfg.judge.swap_mode,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            reference_categories=protocol.judge.reference_categories,
            prompt_preset=protocol.judge.fastchat_prompt_preset,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        )
    )
    return _finalize_mt_bench_run(
        cfg=cfg,
        protocol=protocol,
        res_folder=res_folder,
        result_name=result_name,
        prefs=prefs,
        annotations=annotations,
        combined_metadata=combined_metadata,
        resolved_prompt=resolved_prompt,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        started_at_utc=started_at_utc,
        extra_result_fields={"num_inconsistent": num_inconsistent},
    )


def _run_mt_bench_preset(
    *,
    cfg: RunConfig,
    protocol: MTBenchProtocol,
    res_folder: Path,
    result_name: str,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
    started_at_utc: datetime,
) -> pd.Series:
    prefs, annotations, combined_metadata = judge_mt_bench_with_preset(
        judge_chat_model=judge_chat_model,
        judge_model=cfg.judge.model,
        questions=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        model_a=cfg.model.name,
        model_b=cfg.model.baseline,
        turns_mode=protocol.judge.turns_mode,
        swap_mode=cfg.judge.swap_mode,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
        reference_categories=protocol.judge.reference_categories,
        prompt_preset=cfg.judge.prompt_preset or resolved_prompt.preset_name,
        system_file=cfg.judge.prompt.system_file if cfg.judge.prompt else None,
        user_file=cfg.judge.prompt.user_file if cfg.judge.prompt else None,
        parser=cfg.judge.prompt.parser if cfg.judge.prompt else None,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
    )
    return _finalize_mt_bench_run(
        cfg=cfg,
        protocol=protocol,
        res_folder=res_folder,
        result_name=result_name,
        prefs=prefs,
        annotations=annotations,
        combined_metadata=combined_metadata,
        resolved_prompt=resolved_prompt,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        started_at_utc=started_at_utc,
    )


def run_mt_bench_benchmark(cfg: RunConfig, task: ResolvedTaskSpec | None = None):
    """Run the registered MT-Bench generation, judging, and reporting lifecycle."""
    run_started_at = datetime.now(UTC)
    cfg = cfg.for_task(task)
    task = cfg.resolved_task
    protocol = task.spec.protocol
    if not isinstance(protocol, MTBenchProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define an MT-Bench protocol.")
    if cfg.judge.swap_mode not in ("fixed", "both"):
        raise ValueError(
            "MT-Bench supports only swap_mode='fixed' or 'both'; "
            f"got {cfg.judge.swap_mode!r}."
        )
    resolved_prompt = resolve_run_judge_prompt(
        cfg.resolved_task, cfg.judge, multi_turn=True
    )
    if (
        resolved_prompt.parser is not None
        and resolved_prompt.parser.requires_top_logprobs
    ):
        raise ValueError(
            "MT-Bench does not support parsers that require logprobs. "
            "Use a text-based judge parser."
        )
    task_data = load_task_data(task).select(
        resolve_dataset_filters(task, cfg.run.filters)
    )
    questions_df = task_data.instructions.head(cfg.generation.n_instructions)
    _require_columns(questions_df, {"turn_1", "turn_2"}, "Two-turn instructions")
    cfg.model.baseline = resolve_baseline_plan(
        task_id=cfg.task,
        task=task,
        runtime_baseline=cfg.model.baseline,
        instructions=questions_df,
    ).single_model
    result_name = (
        f"{cfg.task}-{cfg.model.name}-{cfg.model.baseline}-{cfg.judge.model}-"
        f"{cfg.judge.swap_mode}"
    ).replace("/", "_")
    run_timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder) / f"{result_name}-{run_timestamp}",
    )

    logger.info(
        "Generating multi-turn completions for MT-Bench with %s and %s.",
        cfg.model.name,
        cfg.model.baseline,
    )
    completions_a, completions_b = _generate_mt_bench_completions(
        cfg=cfg,
        task=task,
        questions_df=questions_df,
        task_data=task_data,
    )
    if resolved_prompt.delegated:
        logger.info(
            "MT-Bench keeps the original FastChat-style explanation-plus-verdict "
            "prompt when delegated to FastChat compatibility mode."
        )
    judge_model_kwargs = cfg.judge.model_kwargs(
        base_engine_kwargs=cfg.model.engine_kwargs,
        fallback_chat_template=cfg.model.chat_template,
    )
    if resolved_prompt.delegated and cfg.judge.temperature is None:
        judge_model_kwargs.setdefault(
            "temperature", protocol.judge.fastchat_temperature
        )
    judge_chat_model = prepare_model(
        model=cfg.judge.model,
        cache=build_judgement_cache(cfg),
        **judge_model_kwargs,
    )
    if resolved_prompt.delegated:
        return _run_mt_bench_fastchat(
            cfg=cfg,
            protocol=protocol,
            res_folder=res_folder,
            result_name=result_name,
            questions_df=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            judge_chat_model=judge_chat_model,
            resolved_prompt=resolved_prompt,
            started_at_utc=run_started_at,
        )
    return _run_mt_bench_preset(
        cfg=cfg,
        protocol=protocol,
        res_folder=res_folder,
        result_name=result_name,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
        resolved_prompt=resolved_prompt,
        started_at_utc=run_started_at,
    )
