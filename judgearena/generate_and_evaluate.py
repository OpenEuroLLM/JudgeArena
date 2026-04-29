"""
This script generates completions for a given task (dataset) and model,
and then evaluates them using a judge model.
"""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pandas as pd

from judgearena.cli_common import (
    BaseCliArgs,
    GenerationConfig,
    gen_config_to_invoke_kwargs,
)
from judgearena.evaluate import judge_and_parse_prefs, resolve_judge_prompts
from judgearena.generate import generate_base, generate_instructions
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
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
    print(f"   Total Battles: {results['num_battles']}")
    print(f"   Win Rate (A): {results['winrate']:.1%}")
    print(f"   ✅ Wins:   {results['num_wins']}")
    print(f"   ❌ Losses: {results['num_losses']}")
    print(f"   🤝 Ties:   {results['num_ties']}")
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

    # Build the result folder early so the file handler captures the entire run.
    # Include a timestamp so each run gets its own unique directory.
    name = f"{args.task}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    name = name.replace("/", "_")
    run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = Path(args.result_folder) / f"{name}-{run_ts}"
    res_folder.mkdir(parents=True, exist_ok=True)
    if not args.no_log_file:
        attach_file_handler(make_run_log_path(res_folder))

    logger.info(
        "Using task %s and evaluating models %s and %s.",
        args.task,
        args.model_A,
        args.model_B,
    )

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # if not args.ignore_cache:
    #     set_langchain_cache()
    ignore_cache = args.ignore_cache

    if args.task == "mt-bench":
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
    else:
        instructions = load_instructions(
            dataset=args.task, n_instructions=args.n_instructions
        ).loc[:, "instruction"]

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    logger.info(
        "Generating completions for task %s with model %s and %s "
        "(or loading them directly if present)",
        args.task,
        args.model_A,
        args.model_B,
    )

    # Per-role generation configs (resolved by the CLI dispatcher).  Each
    # config carries every knob that can affect the generated text -
    # temperature, top_p, top_k, seed, max_tokens, max_model_len,
    # chat_template, engine_kwargs - so model A, model B and the judge
    # can be configured independently.
    gen_a: GenerationConfig = args.gen_A
    gen_b: GenerationConfig = args.gen_B
    gen_judge: GenerationConfig = args.gen_judge

    def _build_gen_fn(gen: GenerationConfig):
        invoke_kwargs = gen_config_to_invoke_kwargs(gen)
        # ``max_tokens`` is passed as a dedicated arg below.
        invoke_kwargs.pop("max_tokens", None)
        if is_fluency_task:
            # TODO currently we just support base models for fluency, we
            # could also support instruction-tuned models.
            return partial(
                generate_base,
                truncate_input_chars=args.truncate_all_input_chars,
                max_tokens=gen.max_tokens,
                use_tqdm=args.use_tqdm,
                **invoke_kwargs,
            )
        return partial(
            generate_instructions,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=gen.max_tokens,
            use_tqdm=args.use_tqdm,
            **invoke_kwargs,
        )

    dataset_completions_A = try_load_dataset_completions(
        args.task, args.model_A, n_instructions
    )
    if dataset_completions_A is not None:
        completions_A = dataset_completions_A.set_index("instruction_index").loc[
            :, "completion"
        ]
    else:
        gen_fn_a = _build_gen_fn(gen_a)
        completions_A = cache_function_dataframe(
            lambda: gen_fn_a(
                instructions=instructions,
                model=args.model_A,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=ignore_cache,
            cache_name=f"{args.task}_{args.model_A}_{args.n_instructions}",
        ).set_index("instruction_index")
        completions_A = completions_A.loc[:, "completion"]

    dataset_completions_B = try_load_dataset_completions(
        args.task, args.model_B, n_instructions
    )
    if dataset_completions_B is not None:
        completions_B = dataset_completions_B.set_index("instruction_index").loc[
            :, "completion"
        ]
    else:
        gen_fn_b = _build_gen_fn(gen_b)
        completions_B = cache_function_dataframe(
            lambda: gen_fn_b(
                instructions=instructions,
                model=args.model_B,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=ignore_cache,
            cache_name=f"{args.task}_{args.model_B}_{args.n_instructions}",
        ).set_index("instruction_index")
        completions_B = completions_B.loc[:, "completion"]
    logger.debug("First instruction/context: %s", instructions.values[0])
    logger.debug("First completion of %s:\n%s", args.model_A, completions_A.values[0])
    logger.debug("First completion of %s:\n%s", args.model_B, completions_B.values[0])
    logger.info("Evaluating completions with judge %s.", args.judge_model)

    judge_chat_model = make_model(
        model=args.judge_model,
        **gen_config_to_invoke_kwargs(gen_judge),
    )

    # save argument for results analysis
    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(asdict(args), f, indent=2)

    logger.info("Saving results to %s", res_folder)
    if is_fluency_task:
        system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
        model based on the quality of completion of a sentence. You will see a sentence to be completed and two \
        completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not \
        over-confidently prefer one assistant or the other and also make sure to not bias your preference based on \
        the ordering or on the length of the answers."""
    else:
        # the default system prompt of annotate is to compare instruction tuned models.
        system_prompt = None
    (
        effective_judge_system_prompt,
        judge_user_prompt_template,
    ) = resolve_judge_prompts(
        provide_explanation=args.provide_explanation,
        system_prompt=system_prompt,
    )

    annotations, annotations_reversed, prefs = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        swap_mode=args.swap_mode,
        provide_explanation=args.provide_explanation,
        system_prompt=effective_judge_system_prompt,
        user_prompt_template=judge_user_prompt_template,
        truncate_input_chars=args.truncate_all_input_chars,
        use_tqdm=args.use_tqdm,
    )

    df = pd.DataFrame(annotations)
    df["instruction_index"] = instructions.head(n_instructions).index.tolist()
    df["model_A"] = args.model_A
    df["model_B"] = args.model_B
    df["judge"] = args.judge_model

    if args.swap_mode == "both":
        df_reversed = pd.DataFrame(annotations_reversed)
        df_reversed["instruction_index"] = instructions.head(
            n_instructions
        ).index.tolist()
        df_reversed["model_A"] = args.model_B
        df_reversed["model_B"] = args.model_A
        df_reversed["judge"] = args.judge_model
        df = pd.concat([df, df_reversed])

    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    # compute and report statistics
    summary = compute_pref_summary(prefs)

    results = {
        "task": args.task,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        **summary,
        "preferences": prefs.tolist(),
    }
    logger.info("%s vs %s judged by %s", args.model_A, args.model_B, args.judge_model)
    print_results(results)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)

    eval_instruction_index = instructions.head(n_instructions).index.tolist()
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
            },
            judge_system_prompt=effective_judge_system_prompt,
            judge_user_prompt_template=judge_user_prompt_template,
            started_at_utc=run_started_at,
        )
    except OSError as e:
        logger.warning("Failed to write run metadata: %s", e)

    return prefs
