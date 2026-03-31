"""
This script generates completions for a given dataset and model,
and then evaluates them using a judge model.
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd

from openjury.config import CliArgs
from openjury.evaluate import (
    PairScore,
)
from openjury.eval_utils import _make_judge_annotation, print_results
from openjury.generate import generate_instructions, generate_base
from openjury.instruction_dataset import load_instructions
from openjury.repro import _to_jsonable
from openjury.mt_bench.mt_bench_utils import (
    run_mt_bench,
)
from openjury.utils import (
    cache_function_dataframe,
    compute_pref_summary,
    data_root,
    download_hf,
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
    print(f"Found pre-existing completions for '{model}' in {dataset} dataset.")
    completions = df_outputs.loc[:, model]
    if n_instructions is not None:
        completions = completions.head(n_instructions)
    return pd.DataFrame(
        {
            "completion": completions.values,
            "instruction_index": completions.index.tolist(),
        }
    )


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


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

    print(
        f"Using dataset {args.dataset} and evaluating models {args.model_A} and {args.model_B}."
    )

    # Not working with vllm, not detecting model changes and serving the same cache for two different models...
    # if not args.ignore_cache:
    #     set_langchain_cache()
    ignore_cache = args.ignore_cache

    # MT-Bench has its own pipeline: multi-turn generation + category-aware judging
    if args.dataset == "mt-bench":
        return run_mt_bench(args, ignore_cache)

    # Currrently, we run context evaluation
    is_fluency_task = "fluency" in args.dataset
    if is_fluency_task:
        # if args.dataset = "fluency-french", we map to "french-contexts.csv"
        # to match files in https://huggingface.co/datasets/geoalgo/multilingual-contexts-to-be-completed
        lang = args.dataset.split("-")[-1]
        instructions = load_contexts(f"{lang}-contexts.csv")
    else:
        instructions = load_instructions(
            dataset=args.dataset, n_instructions=args.n_instructions
        ).loc[:, "instruction"]

    n_instructions = args.n_instructions if args.n_instructions else len(instructions)
    if args.n_instructions is not None:
        instructions = instructions[:n_instructions]

    print(
        f"Generating completions for dataset {args.dataset} with model {args.model_A} and "
        f"{args.model_B} (or loading them directly if present)"
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
    dataset_completions_A = try_load_dataset_completions(
        args.dataset, args.model_A, n_instructions
    )
    if dataset_completions_A is not None:
        completions_A = dataset_completions_A.set_index("instruction_index").loc[:, "completion"]
    else:
        completions_A = cache_function_dataframe(
            lambda: gen_fun(
                instructions=instructions,
                model=args.model_A,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=ignore_cache,
            cache_name=f"{args.dataset}_{args.model_A}_{args.n_instructions}",
        ).set_index("instruction_index")
        completions_A = completions_A.loc[:, "completion"]

    dataset_completions_B = try_load_dataset_completions(
        args.dataset, args.model_B, n_instructions
    )
    if dataset_completions_B is not None:
        completions_B = dataset_completions_B.set_index("instruction_index").loc[:, "completion"]
    else:
        completions_B = cache_function_dataframe(
            lambda: gen_fun(
                instructions=instructions,
                model=args.model_B,
                use_tqdm=args.use_tqdm,
            ),
            ignore_cache=ignore_cache,
            cache_name=f"{args.dataset}_{args.model_B}_{args.n_instructions}",
        ).set_index("instruction_index")
        completions_B = completions_B.loc[:, "completion"]
    print(f"\nFirst instruction/context: {instructions.values[0]}")

    print(f"\nFirst completion of {args.model_A}")
    print(completions_A.values[0])
    print(f"\nFirst completion of {args.model_B}")
    print(completions_B.values[0])
    print(f"Evaluating completions with judge {args.judge_model}.")

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
        **args.engine_kwargs,
    )
    if is_fluency_task:
        system_prompt = """You are a highly efficient assistant, who evaluates and selects the best large language \
        model based on the quality of completion of a sentence. You will see a sentence to be completed and two \
        completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not \
        over-confidently prefer one assistant or the other and also make sure to not bias your preference based on \
        the ordering or on the length of the answers."""
    else:
        # the default system prompt of annotate is to compare instruction tuned models.

        system_prompt = None

    instruction_subset = instructions.head(n_instructions)
    instruction_indices = instruction_subset.index.tolist()
    metadata = [{"instruction_index": idx} for idx in instruction_indices]
    score_parser = PairScore()
    judge_out = _make_judge_annotation(
        judge_chat_model=judge_chat_model,
        instructions=instruction_subset.tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        metadata=metadata,
        score_parser=score_parser,
        provide_explanation=args.provide_explanation,
        swap_mode=args.swap_mode,
        truncate_input_chars=args.truncate_all_input_chars,
        use_tqdm=args.use_tqdm,
        system_prompt=system_prompt,
    )

    name = f"{args.dataset}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    name = name.replace("/", "_")

    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    # save argument for results analysis
    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(_to_jsonable(asdict(args)), f, indent=2, allow_nan=False)

    print(f"Saving results to {res_folder}")
    df = pd.DataFrame(judge_out.annotations)
    df["instruction_index"] = [
        meta["instruction_index"] for meta in judge_out.metadata_for_annotations
    ]
    df["model_A"] = args.model_A
    df["model_B"] = args.model_B
    df["judge"] = args.judge_model

    if args.swap_mode == "both":
        df_reversed = pd.DataFrame(judge_out.annotations_reversed)
        df_reversed["instruction_index"] = [
            meta["instruction_index"] for meta in judge_out.metadata_for_reversed_annotations
        ]
        df_reversed["model_A"] = args.model_B
        df_reversed["model_B"] = args.model_A
        df_reversed["judge"] = args.judge_model
        df = pd.concat([df, df_reversed])

    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    stats = compute_pref_summary(judge_out.preferences)
    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        **stats,
        "preferences": judge_out.preferences.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print(f"{args.model_A} vs {args.model_B} judged by {args.judge_model}")
    print_results(results)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)

    return judge_out.preferences


def cli():
    args = CliArgs.parse_args()
    print(f"Running with CLI args: {args.__dict__}")
    main(args)


if __name__ == "__main__":
    cli()
