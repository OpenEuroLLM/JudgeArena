"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from judgearena.cli_common import (
    BaseCliArgs,
    add_common_arguments,
    parse_engine_kwargs,
    parse_optional_bool,
)
from judgearena.evaluate import judge_and_parse_prefs, resolve_judge_prompts
from judgearena.generate import generate_base, generate_instructions
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.arena_hard import (
    arena_hard_native_baseline,
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.judge_prompt_presets import DEFAULT_JUDGE_PROMPT_PRESET
from judgearena.mt_bench.mt_bench_utils import run_mt_bench
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
    data_root,
    download_hf,
    is_thinking_model,
    make_model,
    read_df,
)


def try_load_dataset_completions(
    dataset: str, model: str, n_instructions: int | None
) -> pd.DataFrame | None:
    """Try loading pre-existing completions from the dataset.

    Some datasets (e.g. alpaca-eval) ship with completions for well-known
    models such as ``gpt4_1106_preview``.  When ``model`` matches a column in
    ``model_outputs/{dataset}.csv.zip``, those completions are returned
    directly so that no model instantiation / generation is needed.

    Returns a DataFrame with columns ``completion`` and ``instruction_index``,
    or ``None`` when no pre-existing completions are found.
    """
    local_path_tables = data_root / "tables"
    if is_arena_hard_dataset(dataset):
        download_arena_hard(dataset=dataset, local_tables_path=local_path_tables)
    else:
        download_hf(name=dataset, local_path=local_path_tables)
    output_path = local_path_tables / "model_outputs" / f"{dataset}.csv.zip"
    if not output_path.exists():
        return None
    df_outputs = read_df(output_path)
    df_outputs.loc[:, "output"] = df_outputs.loc[:, "output"].fillna("")
    df_outputs = df_outputs.pivot_table(
        index="instruction_index", columns="model", values="output", aggfunc="last"
    ).sort_index()
    if model not in df_outputs.columns:
        return None
    print(f"Found pre-existing completions for '{model}' in dataset '{dataset}'.")
    completions = df_outputs.loc[:, model]
    if n_instructions is not None:
        completions = completions.head(n_instructions)
    return pd.DataFrame(
        {
            "completion": completions.values,
            "instruction_index": completions.index.tolist(),
        }
    )


@dataclass
class CliArgs(BaseCliArgs):
    """CLI arguments for the generate-and-evaluate entrypoint."""

    dataset: str | None = None
    model_A: str | None = None
    model_B: str | None = None
    use_tqdm: bool = False

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Generate completion and evaluate with a judge",
        )
        parser.add_argument(
            "--dataset",
            help="The dataset to use. For instance `alpaca-eval`, `arena-hard-v2.0`, "
            "`arena-hard-v0.1`, `m-arena-hard-v0.1-EU`, `m-arena-hard-v2.0-uk` for "
            "instruction tuning cases or `french-contexts`, `spanish-contexts` for "
            "base models.",
        )
        parser.add_argument(
            "--model_A",
            required=True,
            help="Name of the LLM to use for a generation, must be a valid choice for `generation_provider`",
        )
        parser.add_argument(
            "--model_B",
            default=None,
            help=(
                "Name of the baseline LLM for a generation. Optional for Arena-Hard "
                "datasets (which ship a dataset-native default per category; see "
                "`ARENA_HARD_BASELINES`) and MT-Bench (see `MT_BENCH_BASELINES`, "
                "defaults to `gpt-4`). Required for every other dataset."
            ),
        )
        parser.add_argument(
            "--use_tqdm",
            nargs="?",
            const=True,
            default=False,
            type=parse_optional_bool,
            help="If specified, use tqdm, does not work with all model providers, vLLM in particular.",
        )
        add_common_arguments(parser)
        args = parser.parse_args()

        return cls(
            dataset=args.dataset,
            model_A=args.model_A,
            model_B=args.model_B,
            use_tqdm=args.use_tqdm,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            provide_explanation=args.provide_explanation,
            swap_mode=args.swap_mode,
            ignore_cache=args.ignore_cache,
            judge_prompt_preset=args.judge_prompt_preset,
            battle_thinking_token_budget=args.battle_thinking_token_budget,
            strip_thinking_before_judging=args.strip_thinking_before_judging,
            truncate_all_input_chars=args.truncate_all_input_chars,
            truncate_judge_input_chars=args.truncate_judge_input_chars,
            max_out_tokens_models=args.max_out_tokens_models,
            max_out_tokens_judge=args.max_out_tokens_judge,
            max_model_len=args.max_model_len,
            max_judge_model_len=args.max_judge_model_len,
            chat_template=args.chat_template,
            result_folder=args.result_folder,
            engine_kwargs=parse_engine_kwargs(args.engine_kwargs),
            judge_engine_kwargs=parse_engine_kwargs(args.judge_engine_kwargs),
        )


@dataclass(frozen=True)
class BaselinePlan:
    """Row-aligned baseline assignment for `--model_B`.

    Mirrors upstream's `JUDGE_SETTINGS[question["category"]]["baseline"]` lookup
    in `arena-hard-auto/gen_judgment.py`: a flat plan assigns one baseline to
    every row, a per-row plan assigns a different baseline per category (v2.0
    mixes `o3-mini-2025-01-31` on hard prompts with `gemini-2.0-flash-001` on
    creative writing).
    """

    baseline_by_index: pd.Series

    @classmethod
    def flat(cls, model: str, *, index: pd.Index) -> "BaselinePlan":
        return cls(
            baseline_by_index=pd.Series(model, index=index, name="model_B", dtype=str)
        )

    @classmethod
    def per_row(cls, series: pd.Series) -> "BaselinePlan":
        return cls(baseline_by_index=series.astype(str).rename("model_B"))

    @property
    def unique_models(self) -> list[str]:
        return sorted(self.baseline_by_index.dropna().unique().tolist())

    @property
    def is_flat(self) -> bool:
        return len(self.unique_models) == 1

    @property
    def single_model(self) -> str:
        if not self.is_flat:
            raise ValueError(
                "BaselinePlan is per-row; use baseline_by_index for row-level lookups"
            )
        return self.unique_models[0]

    @property
    def display_name(self) -> str:
        return self.single_model if self.is_flat else "+".join(self.unique_models)

    def aligned_to(self, index: pd.Index) -> pd.Series:
        return self.baseline_by_index.loc[index]


def _resolve_baseline_plan(
    args: CliArgs, instructions_df: pd.DataFrame
) -> BaselinePlan:
    """Explicit `--model_B` wins; otherwise fall back to the dataset-native
    assignment. Non-arena-hard datasets without an override raise.
    """
    if args.model_B is not None:
        return BaselinePlan.flat(args.model_B, index=instructions_df.index)
    if not is_arena_hard_dataset(args.dataset):
        raise ValueError(
            f"--model_B is required for dataset '{args.dataset}'; only Arena-Hard "
            "datasets ship a dataset-native baseline."
        )
    native = arena_hard_native_baseline(args.dataset)
    if isinstance(native, str):
        return BaselinePlan.flat(native, index=instructions_df.index)
    if isinstance(native, Mapping):
        if "category" not in instructions_df.columns:
            raise ValueError(
                f"{args.dataset} requires a 'category' column for per-category "
                "baseline routing; re-run dataset download to regenerate the "
                "instructions table."
            )
        per_row = instructions_df["category"].map(native)
        if per_row.isna().any():
            unknown = sorted(
                instructions_df.loc[per_row.isna(), "category"].unique().tolist()
            )
            raise ValueError(
                f"Unknown Arena-Hard categories for {args.dataset}: {unknown}. "
                f"Known: {sorted(native.keys())}"
            )
        return BaselinePlan.per_row(per_row)
    raise ValueError(f"Unsupported baseline shape for dataset '{args.dataset}'.")


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


def _build_generation_model_kwargs(
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


def _build_judge_model_kwargs(
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


def _generation_cache_name(args: CliArgs, *, model_spec: str) -> str:
    generation_config = {
        "truncate_all_input_chars": args.truncate_all_input_chars,
        "max_out_tokens_models": args.max_out_tokens_models,
        "max_model_len": args.max_model_len,
        "chat_template": args.chat_template,
        "battle_thinking_token_budget": args.battle_thinking_token_budget,
        "engine_kwargs": _build_generation_model_kwargs(
            args=args, model_spec=model_spec
        ),
    }
    generation_config_hash = hashlib.sha256(
        json.dumps(generation_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return f"{args.dataset}_{model_spec}_{args.n_instructions}_{generation_config_hash}"


def print_results(results):
    """Print battle results in a nice formatted way"""

    print("\n" + "=" * 60)
    print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
    print(f"📊 Dataset: {results['dataset']}")
    print(
        f"🤖 Competitors: Model A: {results['model_A']} vs Model B: {results['model_B']}"
    )
    print(f"⚖️ Judge: {results['judge_model']}")
    print("📈 Results Summary:")
    print(f"   Total Battles: {results['num_battles']}")
    print(f"   Win Rate (A): {results['winrate']:.1%}")
    print(f"   ✅ Wins:   {results['num_wins']}")
    print(f"   ❌ Losses: {results['num_losses']}")
    print(f"   🤝 Ties:   {results['num_ties']}")
    print("=" * 60 + "\n")


def main(args: CliArgs):
    """
    1) take as input:
     * dataset, make sure instruct-completion works
     * model to generate output from
     * llm used for judge
     * number of annotations
     * path to save annotations
    2) create completions
    3) create annotations
    """

    run_started_at = datetime.now(UTC)
    usage_tracker = OpenRouterReferencePricingTracker()
    limit_event_tracker = LimitEventTracker()

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # if not args.ignore_cache:
    #     set_langchain_cache()
    ignore_cache = args.ignore_cache

    if args.dataset == "mt-bench":
        return run_mt_bench(args, ignore_cache)

    # Currrently, we run context evaluation
    is_fluency_task = "fluency" in args.dataset
    if is_fluency_task:
        # if args.dataset = "fluency-french", we map to "french-contexts.csv"
        # to match files in https://huggingface.co/datasets/geoalgo/multilingual-contexts-to-be-completed
        lang = args.dataset.split("-")[-1]
        instructions = load_contexts(f"{lang}-contexts.csv")
        instructions_df = pd.DataFrame({"instruction": instructions.values})
        instructions_df.index = instructions.index
    else:
        instructions_df = load_instructions(
            dataset=args.dataset, n_instructions=args.n_instructions
        )
        instructions = instructions_df["instruction"]

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions_df = instructions_df.head(n_instructions)
        instructions = instructions.head(n_instructions)

    baseline_plan = _resolve_baseline_plan(args=args, instructions_df=instructions_df)

    print(
        f"Using dataset {args.dataset} and evaluating {args.model_A} vs baseline "
        f"{baseline_plan.display_name}."
    )
    print(
        f"Generating completions for dataset {args.dataset} with model {args.model_A} "
        f"and baseline {baseline_plan.display_name} "
        "(or loading them directly if present)"
    )

    generation_function = generate_base if is_fluency_task else generate_instructions

    def _run_generation(model_spec: str, usage_phase: str) -> pd.DataFrame:
        return generation_function(
            instructions=instructions,
            model=model_spec,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            use_tqdm=args.use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            limit_event_tracker=limit_event_tracker,
            **_build_generation_model_kwargs(args=args, model_spec=model_spec),
        )

    def _align_completion_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        return df.set_index("instruction_index").loc[instructions.index].reset_index()

    def _load_or_generate_completions(model_spec: str, usage_phase: str) -> pd.Series:
        preloaded = try_load_dataset_completions(
            args.dataset, model_spec, n_instructions
        )
        if preloaded is not None:
            aligned = _align_completion_dataframe(preloaded)
        else:
            aligned = _align_completion_dataframe(
                cache_function_dataframe(
                    lambda: _run_generation(model_spec, usage_phase),
                    ignore_cache=ignore_cache,
                    cache_name=_generation_cache_name(args, model_spec=model_spec),
                )
            )
        return aligned.set_index("instruction_index").loc[
            instructions.index, "completion"
        ]

    completions_A = _load_or_generate_completions(args.model_A, "generation_model_A")

    baseline_per_index = baseline_plan.aligned_to(instructions.index)
    if baseline_plan.is_flat:
        completions_B = _load_or_generate_completions(
            baseline_plan.single_model, "generation_model_B"
        )
    else:
        # Per-row plan: fetch one completion set per unique baseline, then stitch
        # them together so completions_B[uid] uses the baseline that
        # ARENA_HARD_BASELINES routes uid's category to.
        per_baseline_completions: dict[str, pd.Series] = {}
        for baseline_model in baseline_plan.unique_models:
            per_baseline_completions[baseline_model] = _load_or_generate_completions(
                baseline_model, f"generation_model_B::{baseline_model}"
            )
        completions_B = pd.Series(
            [
                per_baseline_completions[model].loc[uid]
                for uid, model in baseline_per_index.items()
            ],
            index=instructions.index,
            name="completion",
        )

    print(f"\nFirst instruction/context: {instructions.values[0]}")

    print(f"\nFirst completion of {args.model_A}")
    print(completions_A.values[0])
    print(f"\nFirst completion of {baseline_plan.display_name}")
    print(completions_B.values[0])
    print(f"Evaluating completions with judge {args.judge_model}.")

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        max_model_len=args.effective_judge_max_model_len(),
        chat_template=args.chat_template,
        **_build_judge_model_kwargs(args=args, limit_event_tracker=limit_event_tracker),
    )

    name = (
        f"{args.dataset}-{args.model_A}-{baseline_plan.display_name}-{args.judge_model}"
    )
    name += f"-{args.swap_mode}"
    name = name.replace("/", "_")

    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    print(f"Saving results to {res_folder}")
    if is_fluency_task:
        system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
        model based on the quality of completion of a sentence. You will see a sentence to be completed and two \
        completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not \
        over-confidently prefer one assistant or the other and also make sure to not bias your preference based on \
        the ordering or on the length of the answers."""
    else:
        # the default system prompt of annotate is to compare instruction tuned models.
        system_prompt = None
    resolved_prompt = resolve_judge_prompts(
        provide_explanation=args.provide_explanation,
        prompt_preset=args.judge_prompt_preset or DEFAULT_JUDGE_PROMPT_PRESET,
        system_prompt=system_prompt,
    )

    annotations, annotations_reversed, prefs = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        case_ids=instructions.head(n_instructions).index.tolist(),
        swap_mode=args.swap_mode,
        provide_explanation=args.provide_explanation,
        prompt_preset=resolved_prompt.preset_name,
        parser_mode=resolved_prompt.parser_mode,
        strip_thinking_before_judging=args.strip_thinking_before_judging,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        truncate_input_chars=args.effective_judge_truncation(),
        use_tqdm=args.use_tqdm,
        usage_tracker=usage_tracker,
        usage_phase="judge",
        usage_model_spec=args.judge_model,
        limit_event_tracker=limit_event_tracker,
    )

    eval_instruction_index = instructions.head(n_instructions).index.tolist()
    baseline_per_eval = baseline_per_index.loc[eval_instruction_index]

    df = pd.DataFrame(annotations)
    df["instruction_index"] = eval_instruction_index
    df["model_A"] = args.model_A
    df["model_B"] = baseline_per_eval.tolist()
    df["judge"] = args.judge_model

    if args.swap_mode == "both":
        df_reversed = pd.DataFrame(annotations_reversed)
        df_reversed["instruction_index"] = eval_instruction_index
        df_reversed["model_A"] = baseline_per_eval.tolist()
        df_reversed["model_B"] = args.model_A
        df_reversed["judge"] = args.judge_model
        df = pd.concat([df, df_reversed])

    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    summary = compute_pref_summary(prefs)

    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": baseline_plan.display_name,
        "baseline_assignment": "per-row" if not baseline_plan.is_flat else "flat",
        "baseline_models": baseline_plan.unique_models,
        "judge_model": args.judge_model,
        "judge_prompt_preset": resolved_prompt.preset_name,
        "strip_thinking_before_judging": args.strip_thinking_before_judging,
        "battle_thinking_token_budget": args.battle_thinking_token_budget,
        **summary,
        "limit_events": limit_event_tracker.build_summary(),
        "preferences": prefs.tolist(),
    }
    print(
        f"{args.model_A} vs {baseline_plan.display_name} judged by {args.judge_model}"
    )
    print_results(results)
    phase_model_specs: dict[str, str] = {
        "generation_model_A": args.model_A,
        "judge": args.judge_model,
    }
    if baseline_plan.is_flat:
        phase_model_specs["generation_model_B"] = baseline_plan.single_model
    else:
        for baseline_model in baseline_plan.unique_models:
            phase_model_specs[f"generation_model_B::{baseline_model}"] = baseline_model
    pricing_reference = build_openrouter_reference_pricing_summary(
        tracker=usage_tracker,
        phase_model_specs=phase_model_specs,
    )
    print(format_openrouter_reference_pricing_summary(pricing_reference))

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)

    eval_instructions = instructions.head(n_instructions).tolist()
    eval_completions_A = completions_A.head(n_instructions).tolist()
    eval_completions_B = completions_B.head(n_instructions).tolist()

    try:
        write_run_metadata(
            output_dir=res_folder,
            entrypoint="judgearena.generate_and_evaluate.main",
            run=asdict(args),
            results=results,
            input_payloads={
                "instruction_index": eval_instruction_index,
                "instructions": eval_instructions,
                "completions_A": eval_completions_A,
                "completions_B": eval_completions_B,
                "baseline_model_B": baseline_per_eval.tolist(),
            },
            judge_system_prompt=resolved_prompt.system_prompt,
            judge_user_prompt_template=resolved_prompt.user_prompt_template,
            started_at_utc=run_started_at,
            pricing_reference=pricing_reference,
        )
    except OSError as e:
        print(f"Warning: failed to write run metadata: {e}")

    return prefs


def cli():
    args = CliArgs.parse_args()
    print(f"Running with CLI args: {args.__dict__}")
    main(args)


if __name__ == "__main__":
    cli()
