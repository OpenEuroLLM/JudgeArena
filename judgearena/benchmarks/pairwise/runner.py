"""
This script generates completions for a given task (dataset) and model,
and then evaluates them using a judge model.
"""

import random
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts import prepare_run_directory, write_run_metadata_safely
from judgearena.benchmarks.execution import build_generation_kwargs, build_judge
from judgearena.benchmarks.pairwise.baselines import resolve_baseline_plan
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets.pairwise import load_pairwise_task_data
from judgearena.evaluate import judge_and_parse_prefs, resolve_run_judge_prompt
from judgearena.generate import generate_base, generate_instructions
from judgearena.log import get_logger
from judgearena.reports import BattleReport
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import ResolvedTaskSpec
from judgearena.utils import cache_function_dataframe, generation_cache_token

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def _build_judge_batches(
    cfg: "RunConfig",
    resolved_task: ResolvedTaskSpec,
    instructions_df: pd.DataFrame,
    eval_index: pd.Index,
    resolved_prompt,
) -> list[tuple[object, pd.Index]]:
    """Group evaluation rows that share the same resolved judge prompt.

    Tasks may map categories to prompt presets (e.g. Arena-Hard v2.0 judges
    creative writing with a different system prompt). Runtime prompt overrides
    (an explicit preset or prompt files) disable the per-category mapping.
    """
    category_prompts = getattr(
        resolved_task.spec.protocol.judge, "category_prompts", {}
    )
    if (
        not category_prompts
        or (
            cfg.judge.prompt_preset is not None
            and cfg.judge.prompt_preset
            != resolved_task.spec.protocol.judge.default_prompt_preset
        )
        or resolved_prompt.source != "preset"
        or "category" not in instructions_df.columns
    ):
        return [(resolved_prompt, eval_index)]

    from judgearena.prompts.registry import resolve_judge_prompt

    categories = instructions_df.loc[eval_index, "category"]
    groups: list[tuple[object, pd.Index]] = []
    for category in dict.fromkeys(categories):
        group_index = categories.index[categories == category]
        preset = category_prompts.get(category)
        prompt = (
            resolved_prompt if preset is None else resolve_judge_prompt(preset=preset)
        )
        groups.append((prompt, group_index))
    return groups


def _random_swap_mask(instructions: pd.Series) -> pd.Series:
    """Return deterministic per-instruction pair-order flips."""
    return pd.Series(
        [
            random.Random(f"is_switched_outputs{instruction}0").choices(
                [False, True], k=1
            )[0]
            for instruction in instructions
        ],
        index=instructions.index,
    )


def run_pairwise(cfg: "RunConfig", resolved_task: ResolvedTaskSpec | None = None):
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

    resolved_task = resolved_task or get_packaged_task(cfg.task)
    if resolved_task is None:
        raise ValueError(f"Unknown task {cfg.task!r}.")
    task_data = load_pairwise_task_data(
        resolved_task,
        n_instructions=cfg.generation.n_instructions,
    )
    instructions_df = task_data.instructions
    instructions = instructions_df.loc[:, "instruction"]
    metric_columns = {
        "instruction_index",
        "model",
        "baseline",
        "completion_model",
        "completion_baseline",
        "pref",
        "orientation",
        "judge",
        "judge_prompt_preset",
        "judge_temperature",
        "judge_max_out_tokens",
        "model_a",
        "model_b",
        "completion_a",
        "completion_b",
        "evaluation_model",
        "source",
        "pref_hard",
        *instructions_df.columns,
    }
    missing_groups = sorted(
        {
            field
            for request in resolved_task.spec.protocol.scoring.metrics
            for field in request.group_by
        }
        - metric_columns
    )
    if missing_groups:
        raise ValueError(f"Metric group_by columns are unavailable: {missing_groups}.")

    n_instructions = (
        cfg.generation.n_instructions
        if cfg.generation.n_instructions
        else len(instructions)
    )
    if cfg.generation.n_instructions is not None:
        instructions_df = instructions_df.head(n_instructions)
        instructions = instructions.head(n_instructions)

    baseline_plan = resolve_baseline_plan(
        task_id=cfg.task,
        task=resolved_task,
        runtime_baseline=cfg.model.baseline,
        instructions=instructions_df,
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
    generation_function = (
        generate_base
        if resolved_task.spec.protocol.generation.mode == "base_completion"
        else generate_instructions
    )

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

    def _load_or_generate_completions(
        model_spec: str,
        *,
        role: str,
        instruction_ids: pd.Index | None = None,
    ) -> pd.Series:
        preloaded = task_data.model_completion(
            model_spec,
            instruction_ids=instruction_ids,
        )
        if preloaded is not None:
            return preloaded
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
        completions = _align_completion_series(generated)
        return (
            completions if instruction_ids is None else completions.loc[instruction_ids]
        )

    completions_A = _load_or_generate_completions(cfg.model.name, role="A")

    baseline_per_index = baseline_plan.aligned_to(instructions.index)
    if baseline_plan.is_single_model:
        completions_B = _load_or_generate_completions(
            baseline_plan.single_model, role="B"
        )
    else:
        per_baseline_completions = {
            model: _load_or_generate_completions(
                model,
                role="B",
                instruction_ids=baseline_per_index.index[baseline_per_index == model],
            )
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

    eval_index = instructions.head(n_instructions).index
    prompt_groups = _build_judge_batches(
        cfg, resolved_task, instructions_df, eval_index, resolved_prompt
    )
    judge_prompt_variants = []
    for group_prompt, _ in prompt_groups:
        prompt_metadata = group_prompt.metadata()
        if prompt_metadata not in judge_prompt_variants:
            judge_prompt_variants.append(prompt_metadata)

    # The baseline fills the first judged slot unless the deterministic mask
    # switches the pair. Preferences are re-oriented to the canonical
    # model/baseline frame after parsing.
    swap_mask = (
        _random_swap_mask(instructions) if cfg.judge.swap_mode == "random" else None
    )
    if swap_mask is not None:
        judged_A = completions_B.mask(swap_mask, completions_A)
        judged_B = completions_A.mask(swap_mask, completions_B)
    else:
        judged_A, judged_B = completions_A, completions_B

    annotations = []
    annotations_reversed = [] if cfg.judge.swap_mode == "both" else None
    direct_prefs, reversed_prefs = [], []
    for group_prompt, group_index in prompt_groups:
        group_annotations, group_reversed, group_prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=instructions.loc[group_index].tolist(),
            completions_A=judged_A.loc[group_index].tolist(),
            completions_B=judged_B.loc[group_index].tolist(),
            swap_mode=cfg.judge.swap_mode,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
            system_prompt=group_prompt.system_prompt,
            user_prompt_template=group_prompt.user_prompt_template,
            prompt_preset=group_prompt.preset_name,
            parse=group_prompt.parser,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            use_tqdm=cfg.run.use_tqdm,
        )
        annotations.extend(group_annotations)
        if group_reversed is not None:
            annotations_reversed.extend(group_reversed)
            # judge_and_parse_prefs returns [direct..., reversed...] per call;
            # split so the global order stays all-direct then all-reversed.
            direct_prefs.append(group_prefs.iloc[: len(group_index)])
            reversed_prefs.append(group_prefs.iloc[len(group_index) :])
        else:
            direct_prefs.append(group_prefs)
    prefs = pd.concat(direct_prefs + reversed_prefs).reset_index(drop=True)

    eval_instruction_index = [
        index for _, group_index in prompt_groups for index in group_index
    ]
    eval_prompt_presets = [
        group_prompt.preset_name
        for group_prompt, group_index in prompt_groups
        for _ in group_index
    ]
    baseline_per_eval = baseline_per_index.loc[eval_instruction_index]
    df = pd.DataFrame(annotations)
    df["instruction_index"] = eval_instruction_index
    if swap_mask is not None:
        swapped_eval = swap_mask.loc[eval_instruction_index].reset_index(drop=True)
        prefs = prefs.astype("float64")
        # Unswitched rows judged the baseline in slot A, so P(slot B wins) is
        # already P(model wins); invert those to the canonical P(baseline wins).
        prefs = prefs.where(swapped_eval, 1 - prefs)
        df["model_A"] = [
            cfg.model.name if swapped else baseline
            for swapped, baseline in zip(swapped_eval, baseline_per_eval, strict=True)
        ]
        df["model_B"] = [
            baseline if swapped else cfg.model.name
            for swapped, baseline in zip(swapped_eval, baseline_per_eval, strict=True)
        ]
    else:
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

    # Metrics see one canonically-oriented row per judgment. Under
    # swap_mode="both", every physical battle contributes both answer orders.
    repeats = 2 if cfg.judge.swap_mode == "both" else 1
    battle_data = {
        "instruction_index": list(eval_instruction_index) * repeats,
        "model": cfg.model.name,
        "baseline": baseline_per_eval.tolist() * repeats,
        "completion_model": completions_A.loc[eval_instruction_index].tolist()
        * repeats,
        "completion_baseline": completions_B.loc[eval_instruction_index].tolist()
        * repeats,
        "model_a": cfg.model.name,
        "model_b": baseline_per_eval.tolist() * repeats,
        "completion_a": completions_A.loc[eval_instruction_index].tolist() * repeats,
        "completion_b": completions_B.loc[eval_instruction_index].tolist() * repeats,
        "evaluation_model": cfg.model.name,
        "source": "llm-judge",
        "pref": pd.Series(prefs, dtype="float64").to_numpy(),
        "orientation": (
            ["direct"] * len(eval_instruction_index)
            + ["reversed"] * len(eval_instruction_index)
        )
        if repeats == 2
        else ["single"] * len(eval_instruction_index),
        "judge": cfg.judge.model,
        "judge_prompt_preset": eval_prompt_presets * repeats,
        "judge_temperature": cfg.judge.temperature,
        "judge_max_out_tokens": cfg.judge.max_out_tokens,
    }
    for column in instructions_df.columns:
        if column not in battle_data:
            battle_data[column] = (
                instructions_df.loc[eval_instruction_index, column].tolist() * repeats
            )
    battles = pd.DataFrame(battle_data)
    battles["pref_hard"] = battles["pref"].map(
        lambda pref: (
            float("nan")
            if pd.isna(pref)
            else 0.0
            if pref < 0.5
            else 1.0
            if pref > 0.5
            else 0.5
        )
    )
    metrics = build_metrics(resolved_task.spec.protocol.scoring.metrics)
    metric_results = calculate_metrics(battles, metrics)

    report = BattleReport(
        task=cfg.task,
        model_a=cfg.model.name,
        model_b=baseline_plan.display_name,
        judge_model=cfg.judge.model,
        metrics=metric_results,
        swap_mode=cfg.judge.swap_mode,
        result_folder=str(res_folder),
        preferences=prefs.tolist(),
        metadata={
            "baseline_assignment": "per-row"
            if not baseline_plan.is_single_model
            else "flat",
            "baseline_models": baseline_plan.unique_models,
            **resolved_prompt.metadata(),
            "judge_prompts": judge_prompt_variants,
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

    eval_instructions = instructions.loc[eval_instruction_index].tolist()
    eval_completions_A = completions_A.loc[eval_instruction_index].tolist()
    eval_completions_B = completions_B.loc[eval_instruction_index].tolist()

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
        judge_prompt_variants=judge_prompt_variants,
        started_at_utc=run_started_at,
    )

    return prefs
