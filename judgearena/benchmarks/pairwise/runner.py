"""
This script generates completions for a given task (dataset) and model,
and then evaluates them using a judge model.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts import prepare_run_directory, write_run_metadata_safely
from judgearena.benchmarks.execution import build_generation_kwargs, build_judge
from judgearena.benchmarks.mt_bench.mt_bench_utils import run_mt_bench
from judgearena.benchmarks.pairwise.baselines import native_pairwise_baseline
from judgearena.datasets import load_instructions
from judgearena.datasets.fluency import is_fluency_task as task_is_fluency
from judgearena.datasets.fluency import load_fluency_contexts
from judgearena.evaluate import judge_and_parse_prefs, resolve_run_judge_prompt
from judgearena.generate import generate_base, generate_instructions
from judgearena.log import get_logger
from judgearena.tasks.registry import get_packaged_task
from judgearena.utils import (
    cache_function_dataframe,
    compute_pref_summary,
    data_root,
    download_hf,
    generation_cache_token,
    read_df,
)
from judgearena.utils.eval import BattleReport

if TYPE_CHECKING:
    from judgearena.config import RunConfig

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
    resolved_task = get_packaged_task(dataset)
    if resolved_task is not None:
        from judgearena.datasets.registry import resolve_dataset_adapter

        adapter = resolve_dataset_adapter(resolved_task.spec.dataset.adapter)
        df_outputs = adapter.load_model_outputs(resolved_task, local_path_tables)
        if df_outputs is None:
            return None
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


@dataclass(frozen=True)
class BaselinePlan:
    """Row-aligned baseline assignment for `--model_B`."""

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
                "BaselinePlan is per-row; use baseline_by_index for row-level lookups."
            )
        return self.unique_models[0]

    @property
    def display_name(self) -> str:
        return self.single_model if self.is_flat else "+".join(self.unique_models)

    def aligned_to(self, index: pd.Index) -> pd.Series:
        return self.baseline_by_index.loc[index]


def _resolve_baseline_plan(
    *, task: str, model_b: str | None, instructions_df: pd.DataFrame
) -> BaselinePlan:
    """Resolve explicit or dataset-native baseline assignment."""
    if model_b is not None:
        return BaselinePlan.flat(model_b, index=instructions_df.index)

    native = native_pairwise_baseline(task)
    if native is None:
        raise ValueError(
            f"model.baseline is required for task '{task}'; no dataset-native "
            "baseline is registered."
        )
    if isinstance(native, str):
        return BaselinePlan.flat(native, index=instructions_df.index)
    if isinstance(native, Mapping):
        if "category" not in instructions_df.columns:
            raise ValueError(
                f"{task} requires a 'category' column for per-category "
                "baseline routing; re-run dataset download to regenerate the "
                "instructions table."
            )
        per_row = instructions_df["category"].map(native)
        if per_row.isna().any():
            unknown = sorted(
                instructions_df.loc[per_row.isna(), "category"].unique().tolist()
            )
            raise ValueError(
                f"Unknown Arena-Hard categories for {task}: {unknown}. "
                f"Known: {sorted(native.keys())}"
            )
        return BaselinePlan.per_row(per_row)
    raise ValueError(f"Unsupported baseline shape for dataset '{task}'.")


def run_pairwise(cfg: "RunConfig"):
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
    # if not cfg.run.ignore_cache:
    #     set_langchain_cache()
    ignore_cache = cfg.run.ignore_cache

    if cfg.task == "mt-bench":
        model_b = cfg.model.baseline or native_pairwise_baseline(cfg.task)
        if not isinstance(model_b, str):
            raise ValueError("MT-Bench requires a flat native baseline.")
        name = f"{cfg.task}-{cfg.model.name}-{model_b}-{cfg.judge.model}"
        name += f"-{cfg.judge.swap_mode}"
        name = name.replace("/", "_")
        run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
        res_folder = prepare_run_directory(
            cfg, Path(cfg.run.result_folder) / f"{name}-{run_ts}"
        )
        return run_mt_bench(
            cfg,
            ignore_cache,
            res_folder=res_folder,
            result_name=name,
        )

    # Currrently, we run context evaluation
    is_fluency_task = task_is_fluency(cfg.task)
    if is_fluency_task:
        # if cfg.task = "fluency-french", we map to the "French" config of
        # https://huggingface.co/datasets/geoalgo/multilingual-fluency
        instructions = load_fluency_contexts(data_root, cfg.task)
        instructions_df = pd.DataFrame({"instruction": instructions.values})
        instructions_df.index = instructions.index
    else:
        instructions_df = load_instructions(
            dataset=cfg.task, n_instructions=cfg.generation.n_instructions
        )
        instructions = instructions_df.loc[:, "instruction"]

    n_instructions = (
        cfg.generation.n_instructions
        if cfg.generation.n_instructions
        else len(instructions)
    )
    if cfg.generation.n_instructions is not None:
        instructions_df = instructions_df.head(n_instructions)
        instructions = instructions.head(n_instructions)

    baseline_plan = _resolve_baseline_plan(
        task=cfg.task, model_b=cfg.model.baseline, instructions_df=instructions_df
    )

    name = f"{cfg.task}-{cfg.model.name}-{baseline_plan.display_name}-{cfg.judge.model}"
    name += f"-{cfg.judge.swap_mode}"
    name = name.replace("/", "_")
    run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = prepare_run_directory(
        cfg, Path(cfg.run.result_folder) / f"{name}-{run_ts}"
    )

    logger.info(
        "Using task %s and evaluating %s against baseline %s.",
        cfg.task,
        cfg.model.name,
        baseline_plan.display_name,
    )

    logger.info(
        "Generating completions for task %s with model %s and baseline %s "
        "(or loading them directly if present)",
        cfg.task,
        cfg.model.name,
        baseline_plan.display_name,
    )

    # TODO currently we just support base models for fluency, we could also support instruction-tuned models
    generation_function = generate_base if is_fluency_task else generate_instructions

    def _run_generation(
        model_spec: str, *, generation_kwargs: dict[str, object]
    ) -> pd.DataFrame:
        return generation_function(
            instructions=instructions,
            model=model_spec,
            truncate_input_chars=cfg.generation.truncate_all_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            **generation_kwargs,
        )

    def _align_completion_series(df: pd.DataFrame) -> pd.Series:
        return df.set_index("instruction_index").loc[instructions.index, "completion"]

    def _load_or_generate_completions(model_spec: str, *, role: str) -> pd.Series:
        preloaded = try_load_dataset_completions(cfg.task, model_spec, n_instructions)
        if preloaded is not None:
            return _align_completion_series(preloaded)
        # Fold the resolved generation kwargs into the cache key so that changing
        # any sampling param (temperature, seed, top_p/k, max_tokens, ...) busts
        # the cached completions instead of silently reusing a stale run.
        generation_kwargs = build_generation_kwargs(cfg, model_spec, role=role)
        sampling_token = generation_cache_token(generation_kwargs)
        generated = cache_function_dataframe(
            lambda: _run_generation(model_spec, generation_kwargs=generation_kwargs),
            ignore_cache=ignore_cache,
            cache_name=(
                f"{cfg.task}_{model_spec}_{cfg.generation.n_instructions}_"
                f"{sampling_token}"
            ),
        )
        return _align_completion_series(generated)

    completions_A = _load_or_generate_completions(cfg.model.name, role="A")

    baseline_per_index = baseline_plan.aligned_to(instructions.index)
    if baseline_plan.is_flat:
        completions_B = _load_or_generate_completions(
            baseline_plan.single_model, role="B"
        )
    else:
        per_baseline_completions = {
            model: _load_or_generate_completions(model, role="B")
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
    logger.debug("First completion of %s:\n%s", cfg.model.name, completions_A.values[0])
    logger.debug(
        "First completion of %s:\n%s",
        baseline_plan.display_name,
        completions_B.values[0],
    )
    logger.info("Evaluating completions with judge %s.", cfg.judge.model)

    judge_chat_model = build_judge(cfg)

    logger.info("Saving results to %s", res_folder)
    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)

    annotations, annotations_reversed, prefs = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instructions.head(n_instructions).tolist(),
        completions_A=completions_A.head(n_instructions).tolist(),
        completions_B=completions_B.head(n_instructions).tolist(),
        swap_mode=cfg.judge.swap_mode,
        provide_explanation=cfg.judge.provide_explanation,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        parser_mode=resolved_prompt.parser_mode,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
    )

    eval_instruction_index = instructions.head(n_instructions).index.tolist()
    baseline_per_eval = baseline_per_index.loc[eval_instruction_index]
    df = pd.DataFrame(annotations)
    df["instruction_index"] = eval_instruction_index
    df["model_A"] = cfg.model.name
    df["model_B"] = baseline_per_eval.tolist()
    df["judge"] = cfg.judge.model

    if cfg.judge.swap_mode == "both":
        df_reversed = pd.DataFrame(annotations_reversed)
        df_reversed["instruction_index"] = eval_instruction_index
        df_reversed["model_A"] = baseline_per_eval.tolist()
        df_reversed["model_B"] = cfg.model.name
        df_reversed["judge"] = cfg.judge.model
        df = pd.concat([df, df_reversed])

    df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    # compute and report statistics
    summary = compute_pref_summary(prefs)

    report = BattleReport(
        task=cfg.task,
        model_a=cfg.model.name,
        model_b=baseline_plan.display_name,
        judge_model=cfg.judge.model,
        summary=summary,
        swap_mode=cfg.judge.swap_mode,
        result_folder=str(res_folder),
        preferences=prefs.tolist(),
        metadata={
            "baseline_assignment": "per-row" if not baseline_plan.is_flat else "flat",
            "baseline_models": baseline_plan.unique_models,
            **resolved_prompt.metadata(),
            "strip_thinking_before_judging": cfg.judge.strip_thinking_before_judging,
            "battle_thinking_token_budget": cfg.judge.battle_thinking_token_budget,
        },
    )
    results = report.to_dict()
    logger.info(
        "%s vs %s judged by %s",
        cfg.model.name,
        baseline_plan.display_name,
        cfg.judge.model,
    )
    report.render()
    report.save(res_folder / f"results-{name}.json")

    eval_instructions = instructions.head(n_instructions).tolist()
    eval_completions_A = completions_A.head(n_instructions).tolist()
    eval_completions_B = completions_B.head(n_instructions).tolist()

    write_run_metadata_safely(
        output_dir=res_folder,
        entrypoint="judgearena.benchmarks.pairwise.runner.run_pairwise",
        run=cfg.model_dump(),
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

    return prefs
