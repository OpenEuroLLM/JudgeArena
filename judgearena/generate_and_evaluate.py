"""
This script generates completions for a given task (dataset) and model,
and then evaluates them using a judge model.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pandas as pd

from judgearena.cli_common import BaseCliArgs
from judgearena.evaluate import judge_and_parse_prefs, resolve_run_judge_prompt
from judgearena.generate import generate_base, generate_instructions
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.arena_hard import (
    ARENA_HARD_BASELINES,
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.instruction_dataset.m_arenahard import (
    M_ARENA_HARD_BASELINES,
    split_m_arena_hard_dataset,
)
from judgearena.instruction_dataset.mt_bench import MT_BENCH_BASELINES
from judgearena.log import (
    attach_file_handler,
    get_logger,
    make_run_log_path,
)
from judgearena.mt_bench.mt_bench_utils import run_mt_bench
from judgearena.repro import _to_jsonable, write_run_metadata
from judgearena.utils import (
    cache_function_dataframe,
    compute_pref_summary,
    data_root,
    download_hf,
    make_model,
    read_df,
)

logger = get_logger(__name__)

ALPACA_EVAL_BASELINES: dict[str, str] = {
    "alpaca-eval": "gpt4_1106_preview",
}

PAIRWISE_BASELINES: dict[str, str | Mapping[str, str]] = {
    **ALPACA_EVAL_BASELINES,
    **ARENA_HARD_BASELINES,
    **M_ARENA_HARD_BASELINES,
    **MT_BENCH_BASELINES,
}


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
    logger.info(
        "Found pre-existing completions for '%s' in dataset '%s'.", model, dataset
    )
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

    task: str | None = None
    model_A: str | None = None
    model_B: str | None = None
    use_tqdm: bool = False


@dataclass(frozen=True)
class BaselinePlan:
    """Row-aligned baseline assignment for native pairwise baselines."""

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
            raise ValueError("BaselinePlan has more than one model.")
        return self.unique_models[0]

    @property
    def display_name(self) -> str:
        return self.single_model if self.is_flat else "+".join(self.unique_models)

    def aligned_to(self, index: pd.Index) -> pd.Series:
        return self.baseline_by_index.loc[index]


def native_pairwise_baseline(task: str) -> str | Mapping[str, str] | None:
    """Return the dataset-native pairwise baseline, if the task defines one."""
    if task in PAIRWISE_BASELINES:
        return PAIRWISE_BASELINES[task]
    parsed_m_arena_hard = split_m_arena_hard_dataset(task)
    if parsed_m_arena_hard is not None:
        version_key, _lang_or_subset = parsed_m_arena_hard
        return PAIRWISE_BASELINES[version_key]
    return None


def _resolve_baseline_plan(
    args: CliArgs, instructions_df: pd.DataFrame
) -> BaselinePlan:
    if args.model_B is not None:
        return BaselinePlan.flat(args.model_B, index=instructions_df.index)

    native = native_pairwise_baseline(args.task)
    if native is None:
        raise ValueError(
            f"--model_B is required for dataset '{args.task}'; no dataset-native "
            "baseline is registered."
        )
    if isinstance(native, str):
        return BaselinePlan.flat(native, index=instructions_df.index)
    if isinstance(native, Mapping):
        if "category" not in instructions_df.columns:
            raise ValueError(
                f"{args.task} requires a 'category' column for per-category "
                "baseline routing."
            )
        per_row = instructions_df["category"].map(native)
        if per_row.isna().any():
            unknown = sorted(
                instructions_df.loc[per_row.isna(), "category"].unique().tolist()
            )
            raise ValueError(
                f"Unknown Arena-Hard categories for {args.task}: {unknown}. "
                f"Known: {sorted(native.keys())}"
            )
        return BaselinePlan.per_row(per_row)
    raise ValueError(f"Unsupported baseline shape for dataset '{args.task}'.")


def _build_judge_engine_kwargs(args: CliArgs) -> dict[str, object]:
    judge_kwargs = dict(args.engine_kwargs)
    judge_kwargs.update(args.judge_engine_kwargs)
    return judge_kwargs


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


def print_results(results):
    """Print battle results in a nice formatted way"""

    print("\n" + "=" * 60)
    print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
    print(f"📊 Task: {results['task']}")
    print(
        f"🤖 Competitors: Model A: {results['model_A']} vs Model B: {results['model_B']}"
    )
    print(f"⚖️ Judge: {results['judge_model']}")
    print("📈 Results Summary:")
    num_battles = results["num_battles"]
    num_missing = results.get("num_missing", 0)
    swap_mode = results.get("swap_mode", "fixed")
    if num_missing > 0:
        parsed = num_battles - num_missing
        print(
            f"   Total Battles: {num_battles}  ⚠️  {num_missing} unparseable (parsed: {parsed}/{num_battles})"
        )
    elif swap_mode == "both":
        print(
            f"   Total Battles: {num_battles} (2×{num_battles // 2} — each instruction judged in both orders to detect positional bias)"
        )
    else:
        print(f"   Total Battles: {num_battles}")
    print(f"   Win Rate (A): {results['winrate']:.1%}")
    print(f"   ✅ Wins:   {results['num_wins']}")
    print(f"   ❌ Losses: {results['num_losses']}")
    print(f"   🤝 Ties:   {results['num_ties']}")
    if results.get("result_folder"):
        print(f"📁 Results: {results['result_folder']}")
    print("=" * 60 + "\n")


def main(args: CliArgs):
    """
    1) take as input:
     * task (dataset), make sure instruct-completion works
     * model to generate output from
     * llm used for judge
     * number of annotations
     * path to save annotations
    2) create completions
    3) create annotations
    """

    run_started_at = datetime.now(UTC)

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # if not args.ignore_cache:
    #     set_langchain_cache()
    ignore_cache = args.ignore_cache

    if args.task == "mt-bench":
        model_b = args.model_B or native_pairwise_baseline(args.task)
        if not isinstance(model_b, str):
            raise ValueError("MT-Bench requires a flat native baseline.")
        name = f"{args.task}-{args.model_A}-{model_b}-{args.judge_model}"
        name += f"-{args.swap_mode}"
        name = name.replace("/", "_")
        run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
        res_folder = Path(args.result_folder) / f"{name}-{run_ts}"
        res_folder.mkdir(parents=True, exist_ok=True)
        if not args.no_log_file:
            attach_file_handler(make_run_log_path(res_folder))
        return run_mt_bench(
            args,
            ignore_cache,
            res_folder=res_folder,
            result_name=name,
        )

    # Currrently, we run context evaluation
    is_fluency_task = "fluency" in args.task
    if is_fluency_task:
        # if args.task = "fluency-french", we map to "french-contexts.csv"
        # to match files in https://huggingface.co/datasets/geoalgo/multilingual-contexts-to-be-completed
        lang = args.task.split("-")[-1]
        instructions = load_contexts(f"{lang}-contexts.csv")
        instructions_df = pd.DataFrame({"instruction": instructions})
    else:
        instructions_df = load_instructions(
            dataset=args.task, n_instructions=args.n_instructions
        )
        instructions = instructions_df.loc[:, "instruction"]

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions_df = instructions_df.head(n_instructions)
        instructions = instructions.head(n_instructions)

    baseline_plan = _resolve_baseline_plan(args=args, instructions_df=instructions_df)

    name = f"{args.task}-{args.model_A}-{baseline_plan.display_name}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    name = name.replace("/", "_")
    run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = Path(args.result_folder) / f"{name}-{run_ts}"
    res_folder.mkdir(parents=True, exist_ok=True)
    if not args.no_log_file:
        attach_file_handler(make_run_log_path(res_folder))

    logger.info(
        "Using task %s and evaluating %s against baseline %s.",
        args.task,
        args.model_A,
        baseline_plan.display_name,
    )

    logger.info(
        "Generating completions for task %s with model %s and baseline %s "
        "(or loading them directly if present)",
        args.task,
        args.model_A,
        baseline_plan.display_name,
    )

    # TODO currently we just support base models for fluency, we could also support instruction-tuned models
    gen_fun = (
        partial(
            generate_base,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            use_tqdm=args.use_tqdm,
            **args.engine_kwargs,
        )
        if is_fluency_task
        else partial(
            generate_instructions,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            use_tqdm=args.use_tqdm,
            **args.engine_kwargs,
        )
    )

    def _align_completion_series(df: pd.DataFrame) -> pd.Series:
        return df.set_index("instruction_index").loc[instructions.index, "completion"]

    def _load_or_generate_completions(model_spec: str) -> pd.Series:
        preloaded = try_load_dataset_completions(args.task, model_spec, n_instructions)
        if preloaded is not None:
            return _align_completion_series(preloaded)
        generated = cache_function_dataframe(
            lambda: gen_fun(
                instructions=instructions,
                model=model_spec,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=ignore_cache,
            cache_name=f"{args.task}_{model_spec}_{args.n_instructions}",
        )
        return _align_completion_series(generated)

    completions_A = _load_or_generate_completions(args.model_A)

    baseline_per_index = baseline_plan.aligned_to(instructions.index)
    if baseline_plan.is_flat:
        completions_B = _load_or_generate_completions(baseline_plan.single_model)
    else:
        per_baseline_completions = {
            model: _load_or_generate_completions(model)
            for model in baseline_plan.unique_models
        }
        completions_B = pd.Series(
            [
                per_baseline_completions[model].loc[instruction_index]
                for instruction_index, model in baseline_per_index.items()
            ],
            index=instructions.index,
            name="completion",
        )

    logger.debug("First instruction/context: %s", instructions.values[0])
    logger.debug("First completion of %s:\n%s", args.model_A, completions_A.values[0])
    logger.debug(
        "First completion of %s:\n%s",
        baseline_plan.display_name,
        completions_B.values[0],
    )
    logger.info("Evaluating completions with judge %s.", args.judge_model)

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        max_model_len=args.max_judge_model_len,
        chat_template=args.chat_template,
        **_build_judge_engine_kwargs(args),
    )

    # save argument for results analysis
    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    logger.info("Saving results to %s", res_folder)
    resolved_prompt = resolve_run_judge_prompt(args.task, args)

    annotations, annotations_reversed, prefs = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        swap_mode=args.swap_mode,
        provide_explanation=args.provide_explanation,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        truncate_input_chars=args.truncate_judge_input_chars,
        use_tqdm=args.use_tqdm,
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

    # compute and report statistics
    summary = compute_pref_summary(prefs)

    results = {
        "task": args.task,
        "model_A": args.model_A,
        "model_B": baseline_plan.display_name,
        "baseline_assignment": "per-row" if not baseline_plan.is_flat else "flat",
        "baseline_models": baseline_plan.unique_models,
        "judge_model": args.judge_model,
        **resolved_prompt.metadata(),
        "swap_mode": args.swap_mode,
        "result_folder": str(res_folder),
        **summary,
        "preferences": prefs.tolist(),
    }
    logger.info(
        "%s vs %s judged by %s",
        args.model_A,
        baseline_plan.display_name,
        args.judge_model,
    )
    print_results(results)

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
        )
    except OSError as e:
        logger.warning("Failed to write run metadata: %s", e)

    return prefs
