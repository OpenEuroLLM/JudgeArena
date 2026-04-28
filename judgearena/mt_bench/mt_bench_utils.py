"""MT-Bench evaluation pipeline.

Orchestrates multi-turn generation, FastChat-compatible pairwise judging,
and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import hashlib
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
from judgearena.judge_prompt_presets import DEFAULT_JUDGE_PROMPT_PRESET
from judgearena.log import get_logger
from judgearena.mt_bench.fastchat_compat import (
    FASTCHAT_TEMPERATURE_CONFIG,
    judge_mt_bench_pairwise_fastchat,
)
from judgearena.openrouter_reference_pricing import (
    OpenRouterReferencePricingTracker,
    build_openrouter_reference_pricing_summary,
    format_openrouter_reference_pricing_summary,
)
from judgearena.repro import _to_jsonable, write_run_metadata
from judgearena.utils import (
    LimitEventTracker,
    build_default_judge_model_kwargs,
    cache_function_dataframe,
    compute_pref_summary,
    is_thinking_model,
    make_model,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from judgearena.generate_and_evaluate import CliArgs


# Original MT-Bench prompts include a visible explanation before the final verdict,
# and Qwen can spend thousands of visible tokens after reasoning ends on turn 2.
_MIN_MT_BENCH_JUDGE_TOKENS = 24576
_MIN_MT_BENCH_JUDGE_MAX_MODEL_LEN = 28672


def _build_mt_bench_generation_kwargs(
    *, args: CliArgs, model_spec: str
) -> dict[str, object]:
    generation_model_kwargs = dict(args.engine_kwargs)
    provider, _, model_name = model_spec.partition("/")
    if (
        args.battle_thinking_token_budget is not None
        and provider == "VLLM"
        and is_thinking_model(model_name)
    ):
        generation_model_kwargs["thinking_token_budget"] = min(
            int(args.battle_thinking_token_budget),
            int(args.max_out_tokens_models),
        )
    return generation_model_kwargs


def _build_mt_bench_judge_model_kwargs(
    *, args: CliArgs, limit_event_tracker: LimitEventTracker | None
) -> dict[str, object]:
    judge_model_kwargs = build_default_judge_model_kwargs(
        args.judge_model,
        args.engine_kwargs,
        judge_engine_kwargs_override=args.judge_engine_kwargs,
    )
    if limit_event_tracker is not None:
        judge_model_kwargs["limit_event_tracker"] = limit_event_tracker
        judge_model_kwargs["limit_event_stage"] = "judge_model_init"
        judge_model_kwargs["limit_event_model_spec"] = args.judge_model
    return judge_model_kwargs


def _mt_bench_generation_cache_name(args: CliArgs, *, model_name: str) -> str:
    generation_config = {
        "truncate_all_input_chars": args.truncate_all_input_chars,
        "max_out_tokens_models": args.max_out_tokens_models,
        "max_model_len": args.max_model_len,
        "chat_template": args.chat_template,
        "battle_thinking_token_budget": args.battle_thinking_token_budget,
        "engine_kwargs": _build_mt_bench_generation_kwargs(
            args=args, model_spec=model_name
        ),
    }
    generation_config_hash = hashlib.sha256(
        json.dumps(generation_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"mt-bench_{model_name}_{args.n_instructions}_{generation_config_hash}"


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
    usage_tracker: OpenRouterReferencePricingTracker,
    limit_event_tracker: LimitEventTracker | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load baseline MT-Bench answers or generate fresh multi-turn outputs."""

    def _run_generation(model_name: str, usage_phase: str) -> pd.DataFrame:
        return generate_multiturn(
            questions=questions_df,
            model=model_name,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            use_tqdm=args.use_tqdm,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            temperature_config=FASTCHAT_TEMPERATURE_CONFIG,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            limit_event_tracker=limit_event_tracker,
            **_build_mt_bench_generation_kwargs(args=args, model_spec=model_name),
        )

    def _load_or_generate(model_name: str, usage_phase: str) -> pd.DataFrame:
        loaded_answers = load_mt_bench_model_answers(
            model_name, n_instructions=args.n_instructions
        )
        if loaded_answers is not None:
            print(f"Using pre-generated MT-Bench answers for '{model_name}'.")
            return _align_mt_bench_completions(
                questions_df=questions_df,
                completions=loaded_answers,
                model_name=model_name,
            )
        generated_answers = cache_function_dataframe(
            lambda: _run_generation(model_name, usage_phase),
            ignore_cache=ignore_cache,
            cache_name=_mt_bench_generation_cache_name(args, model_name=model_name),
        )
        return _align_mt_bench_completions(
            questions_df=questions_df,
            completions=generated_answers,
            model_name=model_name,
        )

    completions_a = _load_or_generate(args.model_A, "generation_model_A")
    completions_b = _load_or_generate(args.model_B, "generation_model_B")
    return completions_a, completions_b


def _build_mt_bench_result_name(args: CliArgs, suffix: str | None = None) -> str:
    """Build a filesystem-safe MT-Bench result artifact prefix."""
    name = f"{args.task}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    if suffix:
        name += f"-{suffix}"
    return name.replace("/", "_")


def _save_mt_bench_results(
    *,
    args: CliArgs,
    res_folder: Path,
    result_name: str,
    results: dict[str, object],
    annotations_df: pd.DataFrame,
    questions_df: pd.DataFrame,
    pricing_reference: dict[str, object] | None,
    started_at_utc: datetime,
    name_suffix: str | None = None,
) -> None:
    """Persist MT-Bench arguments, annotations, and aggregate results."""
    name = _build_mt_bench_result_name(args, suffix=name_suffix)
    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(_to_jsonable(asdict(args)), f, indent=2, allow_nan=False)

    annotations_df.to_csv(res_folder / f"{result_name}-annotations.csv", index=False)

    with open(res_folder / f"results-{result_name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)

    write_run_metadata(
        output_dir=res_folder,
        entrypoint="judgearena.mt_bench.mt_bench_utils.run_mt_bench",
        run=asdict(args),
        results=results,
        input_payloads={
            "instruction_index": questions_df.index.tolist(),
            "turn_1": questions_df["turn_1"].tolist(),
            "turn_2": questions_df["turn_2"].tolist(),
        },
        started_at_utc=started_at_utc,
        pricing_reference=pricing_reference,
    )


def _run_mt_bench_fastchat(
    *,
    args: CliArgs,
    res_folder: Path,
    result_name: str,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
    prompt_preset: str,
    usage_tracker: OpenRouterReferencePricingTracker,
    limit_event_tracker: LimitEventTracker | None,
    started_at_utc: datetime,
) -> pd.Series:
    """Run FastChat-style MT-Bench judging and save the resulting artifacts."""
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
            truncate_input_chars=args.effective_judge_truncation(),
            use_tqdm=args.use_tqdm,
            prompt_preset=prompt_preset,
            strip_thinking_before_judging=args.strip_thinking_before_judging,
            usage_tracker=usage_tracker,
            usage_phase="judge",
            limit_event_tracker=limit_event_tracker,
        )
    )

    stats = compute_pref_summary(prefs)
    results = {
        "task": args.task,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        "judge_prompt_preset": prompt_preset,
        "strip_thinking_before_judging": args.strip_thinking_before_judging,
        "battle_thinking_token_budget": args.battle_thinking_token_budget,
        "num_inconsistent": num_inconsistent,
        **stats,
        "limit_events": limit_event_tracker.build_summary()
        if limit_event_tracker is not None
        else {},
        "per_category": _compute_grouped_stats(prefs, combined_metadata, "category"),
        "per_turn": _compute_grouped_stats(prefs, combined_metadata, "turn"),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print_results(results)
    pricing_reference = build_openrouter_reference_pricing_summary(
        tracker=usage_tracker,
        phase_model_specs={
            "generation_model_A": args.model_A,
            "generation_model_B": args.model_B,
            "judge": args.judge_model,
        },
    )
    print(format_openrouter_reference_pricing_summary(pricing_reference))
    _save_mt_bench_results(
        args=args,
        res_folder=res_folder,
        result_name=result_name,
        results=results,
        annotations_df=pd.DataFrame(annotations),
        questions_df=questions_df,
        pricing_reference=pricing_reference,
        started_at_utc=started_at_utc,
        name_suffix="mtbench",
    )
    return prefs


def run_mt_bench(
    args: CliArgs,
    ignore_cache: bool,
    *,
    res_folder: Path,
    result_name: str,
):
    """MT-Bench pipeline with FastChat-compatible pairwise judging."""
    run_started_at = datetime.now(UTC)
    usage_tracker = OpenRouterReferencePricingTracker()
    limit_event_tracker = LimitEventTracker()
    prompt_preset = args.judge_prompt_preset or DEFAULT_JUDGE_PROMPT_PRESET
    if prompt_preset == DEFAULT_JUDGE_PROMPT_PRESET and not args.provide_explanation:
        logger.info(
            "MT-Bench ignores provide_explanation=False and keeps the original "
            "FastChat-style explanation-plus-verdict prompt."
        )
    if args.model_B is None:
        args.model_B = mt_bench_native_baseline(args.task)
    if args.model_B is None:
        raise ValueError(
            f"--model_B is required for dataset '{args.task}'; "
            "no dataset-native baseline registered."
        )
    if args.max_out_tokens_judge < _MIN_MT_BENCH_JUDGE_TOKENS:
        logger.info(
            "MT-Bench judge prompts require room for budgeted thinking, the "
            "original explanation, and the final verdict; "
            f"overriding max_out_tokens_judge from {args.max_out_tokens_judge} "
            f"to {_MIN_MT_BENCH_JUDGE_TOKENS}."
        )
        args.max_out_tokens_judge = _MIN_MT_BENCH_JUDGE_TOKENS
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
        usage_tracker=usage_tracker,
        limit_event_tracker=limit_event_tracker,
    )
    effective_judge_max_model_len = args.effective_judge_max_model_len()
    if (
        effective_judge_max_model_len is not None
        and effective_judge_max_model_len < _MIN_MT_BENCH_JUDGE_MAX_MODEL_LEN
    ):
        logger.info(
            "MT-Bench judge prompts require a larger total context window for "
            "prompt plus completion; "
            f"overriding judge max_model_len from {effective_judge_max_model_len} "
            f"to {_MIN_MT_BENCH_JUDGE_MAX_MODEL_LEN}."
        )
        args.max_judge_model_len = _MIN_MT_BENCH_JUDGE_MAX_MODEL_LEN
        effective_judge_max_model_len = _MIN_MT_BENCH_JUDGE_MAX_MODEL_LEN
    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        temperature=0.0,
        max_model_len=effective_judge_max_model_len,
        chat_template=args.chat_template,
        **_build_mt_bench_judge_model_kwargs(
            args=args, limit_event_tracker=limit_event_tracker
        ),
    )
    return _run_mt_bench_fastchat(
        args=args,
        res_folder=res_folder,
        result_name=result_name,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
        prompt_preset=prompt_preset,
        usage_tracker=usage_tracker,
        limit_event_tracker=limit_event_tracker,
        started_at_utc=run_started_at,
    )
