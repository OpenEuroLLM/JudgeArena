"""
This script generates completions for a given task (dataset) and model,
and then evaluates them using a judge model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from judgearena.config import RunConfig, dump_config, inference_cache_session
from judgearena.evaluate import judge_and_parse_prefs, resolve_run_judge_prompt
from judgearena.generate import generate_base, generate_instructions
from judgearena.inference_cache import InferenceCache
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
from judgearena.models import (
    build_default_judge_model_kwargs,
    is_thinking_model,
    make_model,
)
from judgearena.mt_bench.mt_bench_utils import run_mt_bench
from judgearena.pairwise_baselines import (
    ALPACA_EVAL_BASELINES,
    PAIRWISE_BASELINES,
    native_pairwise_baseline,
)
from judgearena.repro import write_run_metadata
from judgearena.utils import compute_pref_summary, data_root, download_hf, read_df
from judgearena.utils.eval import BattleReport

logger = get_logger(__name__)

__all__ = [
    "ALPACA_EVAL_BASELINES",
    "PAIRWISE_BASELINES",
    "BaselinePlan",
    "main",
    "native_pairwise_baseline",
    "try_load_dataset_completions",
]


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


@dataclass(frozen=True)
class BaselinePlan:
    """Row-aligned baseline assignment for `--model_B`."""

    baseline_by_index: pd.Series

    @classmethod
    def flat(cls, model: str, *, index: pd.Index) -> BaselinePlan:
        return cls(
            baseline_by_index=pd.Series(model, index=index, name="model_B", dtype=str)
        )

    @classmethod
    def per_row(cls, series: pd.Series) -> BaselinePlan:
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


def _build_generation_kwargs(
    cfg: RunConfig, model_spec: str, *, role: str
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


def load_contexts(dataset: str) -> pd.Series:
    path = data_root / "contexts" / dataset
    return pd.read_csv(path).loc[:, "instruction"]


def _setup_result_folder(
    cfg: RunConfig, result_name: str, run_started_at: datetime
) -> Path:
    run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = Path(cfg.run.result_folder) / f"{result_name}-{run_ts}"
    res_folder.mkdir(parents=True, exist_ok=True)
    if not cfg.run.no_log_file:
        attach_file_handler(make_run_log_path(res_folder))
    return res_folder


def _pairwise_result_name(cfg: RunConfig, baseline_plan: BaselinePlan) -> str:
    name = (
        f"{cfg.task}-{cfg.model.name}-{baseline_plan.display_name}-{cfg.judge.model}"
        f"-{cfg.judge.swap_mode}"
    )
    return name.replace("/", "_")


def _mt_bench_result_name(cfg: RunConfig, model_b: str) -> str:
    name = (
        f"{cfg.task}-{cfg.model.name}-{model_b}-{cfg.judge.model}-{cfg.judge.swap_mode}"
    )
    return name.replace("/", "_")


def _load_task_instructions(cfg: RunConfig) -> tuple[pd.DataFrame, pd.Series, bool]:
    is_fluency_task = "fluency" in cfg.task
    if is_fluency_task:
        lang = cfg.task.split("-")[-1]
        instructions = load_contexts(f"{lang}-contexts.csv")
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
    return instructions_df, instructions, is_fluency_task


def _align_completion_series(df: pd.DataFrame, *, index: pd.Index) -> pd.Series:
    return df.set_index("instruction_index").loc[index, "completion"]


def _load_or_generate_completions(
    *,
    cfg: RunConfig,
    model_spec: str,
    role: str,
    instructions: pd.Series,
    generation_function: Callable[..., pd.DataFrame],
    cache: InferenceCache | None,
    n_instructions: int,
) -> pd.Series:
    preloaded = try_load_dataset_completions(cfg.task, model_spec, n_instructions)
    if preloaded is not None:
        return _align_completion_series(preloaded, index=instructions.index)

    generation_kwargs = _build_generation_kwargs(cfg, model_spec, role=role)
    generated = generation_function(
        instructions=instructions,
        model=model_spec,
        truncate_input_chars=cfg.generation.truncate_all_input_chars,
        use_tqdm=cfg.run.use_tqdm,
        cache=cache,
        **generation_kwargs,
    )
    return _align_completion_series(generated, index=instructions.index)


def _generate_battle_completions(
    *,
    cfg: RunConfig,
    instructions: pd.Series,
    baseline_plan: BaselinePlan,
    generation_function: Callable[..., pd.DataFrame],
    cache: InferenceCache | None,
    n_instructions: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    completions_a = _load_or_generate_completions(
        cfg=cfg,
        model_spec=cfg.model.name,
        role="A",
        instructions=instructions,
        generation_function=generation_function,
        cache=cache,
        n_instructions=n_instructions,
    )

    baseline_per_index = baseline_plan.aligned_to(instructions.index)
    if baseline_plan.is_flat:
        completions_b = _load_or_generate_completions(
            cfg=cfg,
            model_spec=baseline_plan.single_model,
            role="B",
            instructions=instructions,
            generation_function=generation_function,
            cache=cache,
            n_instructions=n_instructions,
        )
    else:
        per_baseline_completions = {
            model: _load_or_generate_completions(
                cfg=cfg,
                model_spec=model,
                role="B",
                instructions=instructions,
                generation_function=generation_function,
                cache=cache,
                n_instructions=n_instructions,
            )
            for model in baseline_plan.unique_models
        }
        completions_b = pd.Series(
            [
                per_baseline_completions[model].loc[instruction_index]
                for instruction_index, model in baseline_per_index.items()
            ],
            index=instructions.index,
            name="completion",
        )

    return completions_a, completions_b, baseline_per_index


def _judge_row_metadata(
    *,
    instruction_indices: list[Any],
    model_a: str,
    baseline_per_row: pd.Series,
) -> list[dict[str, Any]]:
    return [
        {
            "instruction_index": str(instruction_index),
            "model_A": model_a,
            "model_B": str(baseline_per_row.loc[instruction_index]),
        }
        for instruction_index in instruction_indices
    ]


def _build_judge_model(cfg: RunConfig):
    return make_model(
        model=cfg.judge.model,
        **build_default_judge_model_kwargs(
            cfg.judge.model,
            cfg.model.engine_kwargs,
            judge_engine_kwargs_override=cfg.judge.model_kwargs(
                fallback_chat_template=cfg.model.chat_template,
            ),
        ),
    )


def _persist_pairwise_results(
    *,
    cfg: RunConfig,
    res_folder: Path,
    name: str,
    baseline_plan: BaselinePlan,
    instructions: pd.Series,
    completions_a: pd.Series,
    completions_b: pd.Series,
    baseline_per_index: pd.Series,
    annotations: list,
    annotations_reversed: list | None,
    prefs: pd.Series,
    resolved_prompt,
    run_started_at: datetime,
) -> pd.Series:
    n_instructions = len(instructions)
    eval_instruction_index = instructions.index.tolist()
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

    try:
        write_run_metadata(
            output_dir=res_folder,
            entrypoint="judgearena.generate_and_evaluate.main",
            run=cfg.model_dump(),
            results=results,
            input_payloads={
                "instruction_index": eval_instruction_index,
                "instructions": instructions.head(n_instructions).tolist(),
                "completions_A": completions_a.head(n_instructions).tolist(),
                "completions_B": completions_b.head(n_instructions).tolist(),
                "baseline_model_B": baseline_per_eval.tolist(),
            },
            judge_system_prompt=resolved_prompt.system_prompt,
            judge_user_prompt_template=resolved_prompt.user_prompt_template,
            started_at_utc=run_started_at,
        )
    except OSError as exc:
        logger.warning("Failed to write run metadata: %s", exc)

    return prefs


def _run_pairwise_task(cfg: RunConfig, *, run_started_at: datetime) -> pd.Series:
    instructions_df, instructions, is_fluency_task = _load_task_instructions(cfg)
    n_instructions = len(instructions)

    baseline_plan = _resolve_baseline_plan(
        task=cfg.task, model_b=cfg.model.baseline, instructions_df=instructions_df
    )
    name = _pairwise_result_name(cfg, baseline_plan)
    res_folder = _setup_result_folder(cfg, name, run_started_at)

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

    generation_function = generate_base if is_fluency_task else generate_instructions

    with inference_cache_session(cfg) as cache:
        completions_a, completions_b, baseline_per_index = _generate_battle_completions(
            cfg=cfg,
            instructions=instructions,
            baseline_plan=baseline_plan,
            generation_function=generation_function,
            cache=cache,
            n_instructions=n_instructions,
        )

        logger.debug("First instruction/context: %s", instructions.values[0])
        logger.debug(
            "First completion of %s:\n%s", cfg.model.name, completions_a.values[0]
        )
        logger.debug(
            "First completion of %s:\n%s",
            baseline_plan.display_name,
            completions_b.values[0],
        )
        logger.info("Evaluating completions with judge %s.", cfg.judge.model)

        dump_config(cfg, res_folder / "config.yaml")
        logger.info("Saving results to %s", res_folder)

        resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
        judge_chat_model = _build_judge_model(cfg)
        eval_instruction_index = instructions.index.tolist()
        row_metadata = _judge_row_metadata(
            instruction_indices=eval_instruction_index,
            model_a=cfg.model.name,
            baseline_per_row=baseline_per_index,
        )

        annotations, annotations_reversed, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=instructions.tolist(),
            completions_A=completions_a.tolist(),
            completions_B=completions_b.tolist(),
            swap_mode=cfg.judge.swap_mode,
            provide_explanation=cfg.judge.provide_explanation,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
            system_prompt=resolved_prompt.system_prompt,
            user_prompt_template=resolved_prompt.user_prompt_template,
            prompt_preset=resolved_prompt.preset_name,
            parser_mode=resolved_prompt.parser_mode,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            cache=cache,
            row_metadata=row_metadata,
        )

        return _persist_pairwise_results(
            cfg=cfg,
            res_folder=res_folder,
            name=name,
            baseline_plan=baseline_plan,
            instructions=instructions,
            completions_a=completions_a,
            completions_b=completions_b,
            baseline_per_index=baseline_per_index,
            annotations=annotations,
            annotations_reversed=annotations_reversed,
            prefs=prefs,
            resolved_prompt=resolved_prompt,
            run_started_at=run_started_at,
        )


def main(cfg: RunConfig):
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

    if cfg.task == "mt-bench":
        model_b = cfg.model.baseline or native_pairwise_baseline(cfg.task)
        if not isinstance(model_b, str):
            raise ValueError("MT-Bench requires a flat native baseline.")
        result_name = _mt_bench_result_name(cfg, model_b)
        res_folder = _setup_result_folder(cfg, result_name, run_started_at)
        with inference_cache_session(cfg) as cache:
            return run_mt_bench(
                cfg,
                cache=cache,
                res_folder=res_folder,
                result_name=result_name,
            )

    return _run_pairwise_task(cfg, run_started_at=run_started_at)
