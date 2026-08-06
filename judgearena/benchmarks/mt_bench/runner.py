"""Registered MT-Bench runner and evaluation pipeline.

Orchestrates multi-turn generation, FastChat-compatible pairwise judging,
and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts import prepare_run_directory, write_run_metadata_safely
from judgearena.benchmarks.mt_bench.fastchat_compat import (
    judge_mt_bench_pairwise_fastchat,
)
from judgearena.benchmarks.mt_bench.preset_judging import judge_mt_bench_with_preset
from judgearena.benchmarks.pairwise.baselines import native_pairwise_baseline
from judgearena.benchmarks.pairwise.scoring import PAIRWISE_SCORERS, ScoringInputs
from judgearena.datasets import load_instructions
from judgearena.datasets.mt_bench import (
    load_mt_bench_model_answers,
)
from judgearena.generate import generate_multiturn
from judgearena.log import get_logger
from judgearena.models import is_thinking_model, make_model
from judgearena.prompts.registry import ResolvedJudgePrompt, resolve_run_judge_prompt
from judgearena.tasks.schema import MTBenchProtocol
from judgearena.utils import (
    cache_function_dataframe,
    generation_cache_token,
)
from judgearena.utils.eval import BattleReport, _compute_grouped_stats

logger = get_logger(__name__)

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.tasks.schema import ResolvedTaskSpec


def _align_mt_bench_completions(
    *, questions_df: pd.DataFrame, completions: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Align cached or generated MT-Bench completions to the question order."""
    indexed = completions.set_index("instruction_index")
    missing_ids = questions_df.index.difference(indexed.index)
    if not missing_ids.empty:
        missing_ids_preview = ", ".join(str(x) for x in missing_ids[:5])
        raise ValueError(
            f"MT-Bench completions for '{model_name}' are missing "
            f"{len(missing_ids)} question(s). First missing ids: {missing_ids_preview}."
        )
    return indexed.loc[questions_df.index]


def _build_mt_bench_generation_kwargs(
    *, cfg: RunConfig, model_spec: str, role: str
) -> dict[str, object]:
    """Battle-model kwargs, adding a thinking-token sub-budget when requested."""
    if role == "A":
        generation_kwargs = cfg.model.evaluated_generation_kwargs()
    elif role == "B":
        generation_kwargs = cfg.model.baseline_generation_kwargs()
    else:
        raise ValueError(f"Unknown generation role: {role!r}")
    provider, _, model_name = model_spec.partition("/")
    if (
        cfg.judge.battle_thinking_token_budget is not None
        and provider == "VLLM"
        and is_thinking_model(model_name)
    ):
        max_tokens = int(generation_kwargs.get("max_tokens", cfg.model.max_out_tokens))
        generation_kwargs["thinking_token_budget"] = min(
            int(cfg.judge.battle_thinking_token_budget),
            max_tokens,
        )
    return generation_kwargs


def _generate_mt_bench_completions(
    cfg: RunConfig,
    protocol: MTBenchProtocol,
    questions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_prefix = cfg.task

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
            **generation_kwargs,
        )

    def _load_or_generate(model_name: str, *, role: str) -> pd.DataFrame:
        loaded_answers = load_mt_bench_model_answers(
            model_name, n_instructions=cfg.generation.n_instructions
        )
        if loaded_answers is not None:
            return _align_mt_bench_completions(
                questions_df=questions_df,
                completions=loaded_answers,
                model_name=model_name,
            )
        # Fold the resolved generation kwargs into the cache key so changing any
        # sampling param busts cached completions instead of reusing a stale run.
        generation_kwargs = _build_mt_bench_generation_kwargs(
            cfg=cfg, model_spec=model_name, role=role
        )
        sampling_token = generation_cache_token(generation_kwargs)
        generated_answers = cache_function_dataframe(
            lambda: _run_generation(model_name, generation_kwargs=generation_kwargs),
            ignore_cache=cfg.run.ignore_cache,
            cache_name=(
                f"{cache_prefix}_{model_name}_{cfg.generation.n_instructions}_"
                f"{sampling_token}"
            ),
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
        output_dir=res_folder,
        entrypoint="judgearena.benchmarks.mt_bench.runner.run_mt_bench_benchmark",
        run=cfg.model_dump(),
        results=results,
        input_payloads=input_payloads,
        judge_system_prompt=judge_system_prompt,
        judge_user_prompt_template=judge_user_prompt_template,
        started_at_utc=started_at_utc,
    )


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
    scorer = PAIRWISE_SCORERS[protocol.scoring.adapter]
    # MT-Bench battles carry per-turn prefs only; the win-rate scorer reads
    # just the pref column of the canonical battles frame.
    stats = scorer.summarize(
        ScoringInputs(battles=pd.DataFrame({"pref": pd.Series(prefs, dtype="float64")}))
    )
    report = BattleReport(
        task=cfg.task,
        model_a=cfg.model.name,
        model_b=cfg.model.baseline,
        judge_model=cfg.judge.model,
        summary=stats,
        per_category=_compute_grouped_stats(prefs, combined_metadata, "category"),
        per_turn=_compute_grouped_stats(prefs, combined_metadata, "turn"),
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
        provide_explanation=cfg.judge.provide_explanation,
        system_file=cfg.judge.prompt.system_file if cfg.judge.prompt else None,
        user_file=cfg.judge.prompt.user_file if cfg.judge.prompt else None,
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
    protocol = task.spec.protocol if task is not None else None
    if not isinstance(protocol, MTBenchProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define an MT-Bench protocol.")
    if cfg.model.baseline is None:
        baseline = native_pairwise_baseline(cfg.task)
        cfg.model.baseline = baseline if isinstance(baseline, str) else None
    if cfg.model.baseline is None:
        raise ValueError(
            f"--model_B is required for dataset '{cfg.task}'; "
            "no dataset-native baseline registered."
        )
    result_name = (
        f"{cfg.task}-{cfg.model.name}-{cfg.model.baseline}-{cfg.judge.model}-"
        f"{cfg.judge.swap_mode}"
    ).replace("/", "_")
    run_timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder) / f"{result_name}-{run_timestamp}",
    )

    questions_df = load_instructions(
        cfg.task, n_instructions=cfg.generation.n_instructions
    )
    logger.info(
        "Generating multi-turn completions for MT-Bench with %s and %s.",
        cfg.model.name,
        cfg.model.baseline,
    )
    completions_a, completions_b = _generate_mt_bench_completions(
        cfg=cfg,
        protocol=protocol,
        questions_df=questions_df,
    )
    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge, multi_turn=True)
    if resolved_prompt.delegated and not cfg.judge.provide_explanation:
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
    judge_chat_model = make_model(model=cfg.judge.model, **judge_model_kwargs)
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
