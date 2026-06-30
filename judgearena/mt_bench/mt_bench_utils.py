"""MT-Bench evaluation pipeline.

Orchestrates multi-turn generation, FastChat-compatible pairwise judging,
and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.generate import generate_multiturn
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.mt_bench import (
    load_mt_bench_model_answers,
    mt_bench_native_baseline,
)
from judgearena.log import get_logger
from judgearena.models import is_thinking_model, make_model
from judgearena.mt_bench.fastchat_compat import (
    FASTCHAT_TEMPERATURE_CONFIG,
    judge_mt_bench_pairwise_fastchat,
)
from judgearena.mt_bench.preset_judging import judge_mt_bench_with_preset
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    resolve_run_judge_prompt,
)
from judgearena.repro import _to_jsonable, write_run_metadata
from judgearena.utils import (
    cache_function_dataframe,
    compute_pref_summary,
    generation_cache_token,
)
from judgearena.utils.eval import _compute_grouped_stats, print_results

logger = get_logger(__name__)

if TYPE_CHECKING:
    from judgearena.config import RunConfig


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
    questions_df: pd.DataFrame,
    ignore_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_prefix = "mt-bench"

    def _run_generation(
        model_name: str, *, generation_kwargs: dict[str, object]
    ) -> pd.DataFrame:
        # MT-Bench's category-aware temperatures only kick in when the user has
        # not explicitly pinned a per-role temperature; otherwise the config
        # override should win for reproducibility.
        temperature_config = (
            None if "temperature" in generation_kwargs else FASTCHAT_TEMPERATURE_CONFIG
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
            ignore_cache=ignore_cache,
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
    res_folder.mkdir(parents=True, exist_ok=True)

    from judgearena.config import dump_config

    dump_config(cfg, res_folder / "config.yaml")

    annotations_df.to_csv(res_folder / f"{result_name}-annotations.csv", index=False)

    with open(res_folder / f"results-{result_name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)

    write_run_metadata(
        output_dir=res_folder,
        entrypoint="judgearena.mt_bench.mt_bench_utils.run_mt_bench",
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
    stats = compute_pref_summary(prefs)
    results = {
        "task": cfg.task,
        "model_A": cfg.model.name,
        "model_B": cfg.model.baseline,
        "judge_model": cfg.judge.model,
        **resolved_prompt.metadata(),
        "battle_thinking_token_budget": cfg.judge.battle_thinking_token_budget,
        "strip_thinking_before_judging": cfg.judge.strip_thinking_before_judging,
        **(extra_result_fields or {}),
        **stats,
        "per_category": _compute_grouped_stats(prefs, combined_metadata, "category"),
        "per_turn": _compute_grouped_stats(prefs, combined_metadata, "turn"),
        "preferences": prefs.tolist(),
        "date": datetime.now(UTC).isoformat(),
        "user": os.getenv("USER", ""),
    }
    print_results(results)
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
    res_folder: Path,
    result_name: str,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
    fastchat_prompt_preset: str,
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
            turns_mode="both",
            swap_mode=cfg.judge.swap_mode,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            prompt_preset=fastchat_prompt_preset,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        )
    )
    return _finalize_mt_bench_run(
        cfg=cfg,
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
        turns_mode="both",
        swap_mode=cfg.judge.swap_mode,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
        prompt_preset=cfg.judge.prompt_preset or resolved_prompt.preset_name,
        provide_explanation=cfg.judge.provide_explanation,
        system_file=cfg.judge.system_prompt_file,
        user_file=cfg.judge.user_prompt_file,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
    )
    return _finalize_mt_bench_run(
        cfg=cfg,
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


def run_mt_bench(
    cfg: RunConfig,
    ignore_cache: bool,
    *,
    res_folder: Path,
    result_name: str,
):
    """MT-Bench pipeline with preset or FastChat-original pairwise judging."""
    run_started_at = datetime.now(UTC)
    if cfg.model.baseline is None:
        cfg.model.baseline = mt_bench_native_baseline(cfg.task)
    if cfg.model.baseline is None:
        raise ValueError(
            f"--model_B is required for dataset '{cfg.task}'; "
            "no dataset-native baseline registered."
        )
    questions_df = load_instructions(
        "mt-bench", n_instructions=cfg.generation.n_instructions
    )
    logger.info(
        "Generating multi-turn completions for MT-Bench with %s and %s.",
        cfg.model.name,
        cfg.model.baseline,
    )
    completions_a, completions_b = _generate_mt_bench_completions(
        cfg=cfg,
        questions_df=questions_df,
        ignore_cache=ignore_cache,
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
        judge_model_kwargs.setdefault("temperature", 0.0)
    judge_chat_model = make_model(model=cfg.judge.model, **judge_model_kwargs)
    if resolved_prompt.delegated:
        return _run_mt_bench_fastchat(
            cfg=cfg,
            res_folder=res_folder,
            result_name=result_name,
            questions_df=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            judge_chat_model=judge_chat_model,
            resolved_prompt=resolved_prompt,
            fastchat_prompt_preset=DEFAULT_JUDGE_PROMPT_PRESET,
            started_at_utc=run_started_at,
        )
    return _run_mt_bench_preset(
        cfg=cfg,
        res_folder=res_folder,
        result_name=result_name,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
        resolved_prompt=resolved_prompt,
        started_at_utc=run_started_at,
    )
