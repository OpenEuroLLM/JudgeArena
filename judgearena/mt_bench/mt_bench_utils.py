"""MT-Bench evaluation pipeline.

Orchestrates multi-turn generation, FastChat-compatible pairwise judging,
and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.eval_utils import _compute_grouped_stats, print_results
from judgearena.generate import generate_multiturn
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.mt_bench import (
    load_mt_bench_model_answers,
    mt_bench_native_baseline,
)
from judgearena.log import get_logger
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
    make_model,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from judgearena.generate_and_evaluate import CliArgs


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


def _generate_mt_bench_completions(
    args: CliArgs,
    questions_df: pd.DataFrame,
    ignore_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_prefix = "mt-bench"

    def _run_generation(model_name: str) -> pd.DataFrame:
        return generate_multiturn(
            questions=questions_df,
            model=model_name,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            use_tqdm=args.use_tqdm,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            temperature_config=FASTCHAT_TEMPERATURE_CONFIG,
            **args.engine_kwargs,
        )

    def _load_or_generate(model_name: str) -> pd.DataFrame:
        loaded_answers = load_mt_bench_model_answers(
            model_name, n_instructions=args.n_instructions
        )
        if loaded_answers is not None:
            return _align_mt_bench_completions(
                questions_df=questions_df,
                completions=loaded_answers,
                model_name=model_name,
            )
        generated_answers = cache_function_dataframe(
            lambda: _run_generation(model_name),
            ignore_cache=ignore_cache,
            cache_name=f"{cache_prefix}_{model_name}_{args.n_instructions}",
        )
        return _align_mt_bench_completions(
            questions_df=questions_df,
            completions=generated_answers,
            model_name=model_name,
        )

    return _load_or_generate(args.model_A), _load_or_generate(args.model_B)


def _build_mt_bench_result_name(args: CliArgs, suffix: str | None = None) -> str:
    """Build a filesystem-safe MT-Bench result artifact prefix."""
    name = f"{args.task}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    if suffix:
        name += f"-{suffix}"
    return name.replace("/", "_")


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
    args: CliArgs,
    res_folder: Path,
    result_name: str,
    results: dict[str, object],
    annotations_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    started_at_utc: datetime,
    input_payloads: dict[str, object] | None = None,
    judge_system_prompt: str | None = None,
    judge_user_prompt_template: str | None = None,
) -> None:
    """Persist MT-Bench arguments, annotations, aggregate results, and metadata."""
    res_folder.mkdir(parents=True, exist_ok=True)

    with open(res_folder / f"args-{result_name}.json", "w") as f:
        json.dump(_to_jsonable(asdict(args)), f, indent=2, allow_nan=False)

    annotations_df.to_csv(res_folder / f"{result_name}-annotations.csv", index=False)

    with open(res_folder / f"results-{result_name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)

    write_run_metadata(
        output_dir=res_folder,
        entrypoint="judgearena.mt_bench.mt_bench_utils.run_mt_bench",
        run=asdict(args),
        results=results,
        input_payloads=input_payloads
        or {
            "instruction_index": questions_df.index.tolist(),
            "turn_1": questions_df["turn_1"].tolist(),
            "turn_2": questions_df["turn_2"].tolist(),
        },
        judge_system_prompt=judge_system_prompt,
        judge_user_prompt_template=judge_user_prompt_template,
        started_at_utc=started_at_utc,
    )


def _finalize_mt_bench_run(
    *,
    args: CliArgs,
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
        "task": args.task,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        **resolved_prompt.metadata(),
        **(extra_result_fields or {}),
        **stats,
        "per_category": _compute_grouped_stats(prefs, combined_metadata, "category"),
        "per_turn": _compute_grouped_stats(prefs, combined_metadata, "turn"),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print_results(results)
    _save_mt_bench_results(
        args=args,
        res_folder=res_folder,
        result_name=result_name,
        results=results,
        annotations_df=pd.DataFrame(annotations),
        questions_df=questions_df,
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
    args: CliArgs,
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
            judge_model=args.judge_model,
            questions=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            model_a=args.model_A,
            model_b=args.model_B,
            turns_mode="both",
            swap_mode=args.swap_mode,
            truncate_input_chars=args.truncate_judge_input_chars,
            use_tqdm=args.use_tqdm,
            prompt_preset=fastchat_prompt_preset,
        )
    )
    return _finalize_mt_bench_run(
        args=args,
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
    args: CliArgs,
    res_folder: Path,
    result_name: str,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
    started_at_utc: datetime,
) -> pd.Series:
    prefs, annotations, combined_metadata, _num_inconsistent = (
        judge_mt_bench_with_preset(
            judge_chat_model=judge_chat_model,
            judge_model=args.judge_model,
            questions=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            model_a=args.model_A,
            model_b=args.model_B,
            turns_mode="both",
            swap_mode=args.swap_mode,
            truncate_input_chars=args.truncate_judge_input_chars,
            use_tqdm=args.use_tqdm,
            prompt_preset=args.judge_prompt_preset or resolved_prompt.preset_name,
            provide_explanation=args.provide_explanation,
            system_file=args.judge_system_prompt_file,
            user_file=args.judge_user_prompt_file,
        )
    )
    return _finalize_mt_bench_run(
        args=args,
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
    args: CliArgs,
    ignore_cache: bool,
    *,
    res_folder: Path | None = None,
    result_name: str | None = None,
):
    """MT-Bench pipeline with preset or FastChat-original pairwise judging."""
    run_started_at = datetime.now(UTC)
    if args.model_B is None:
        args.model_B = mt_bench_native_baseline(args.task)
    if args.model_B is None:
        raise ValueError(
            f"--model_B is required for dataset '{args.task}'; "
            "no dataset-native baseline registered."
        )
    if result_name is None:
        result_name = _build_mt_bench_result_name(args, suffix="mtbench")
    if res_folder is None:
        res_folder = Path(args.result_folder) / result_name
        res_folder.mkdir(parents=True, exist_ok=True)
    questions_df = load_instructions("mt-bench", n_instructions=args.n_instructions)
    logger.info(
        "Generating multi-turn completions for MT-Bench with %s and %s.",
        args.model_A,
        args.model_B,
    )
    completions_a, completions_b = _generate_mt_bench_completions(
        args=args,
        questions_df=questions_df,
        ignore_cache=ignore_cache,
    )
    resolved_prompt = resolve_run_judge_prompt(args.task, args, multi_turn=True)
    if resolved_prompt.delegated and not args.provide_explanation:
        logger.info(
            "MT-Bench keeps the original FastChat-style explanation-plus-verdict "
            "prompt when delegated to FastChat compatibility mode."
        )
    judge_model_kwargs = {
        "model": args.judge_model,
        "max_tokens": args.max_out_tokens_judge,
        "max_model_len": args.max_judge_model_len,
        "chat_template": args.chat_template,
        **{**args.engine_kwargs, **args.judge_engine_kwargs},
    }
    if resolved_prompt.delegated:
        judge_model_kwargs["temperature"] = 0.0
    judge_chat_model = make_model(**judge_model_kwargs)
    if resolved_prompt.delegated:
        return _run_mt_bench_fastchat(
            args=args,
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
        args=args,
        res_folder=res_folder,
        result_name=result_name,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
        resolved_prompt=resolved_prompt,
        started_at_utc=run_started_at,
    )
