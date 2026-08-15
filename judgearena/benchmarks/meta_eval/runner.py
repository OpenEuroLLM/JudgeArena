"""Score a judge against human-labeled arena battles."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    safe_filename,
    write_run_metadata_safely,
)
from judgearena.benchmarks.arena import resolve_task_languages
from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.meta_eval.agreement import (
    agreement_view,
    compute_agreement_metrics,
)
from judgearena.benchmarks.meta_eval.annotate import annotate_sample
from judgearena.benchmarks.meta_eval.cost import annotation_telemetry
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    normalize_human_winner,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.benchmarks.meta_eval.scoring import (
    META_EVAL_SCORERS,
    ranking_annotations,
)
from judgearena.datasets import load_battles
from judgearena.evaluate import resolve_run_judge_prompt
from judgearena.log import get_logger
from judgearena.tasks.schema import MetaEvalProtocol, ResolvedTaskSpec
from judgearena.utils.eval import Report

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)

SAMPLE_FILENAME = "sample.parquet"
ANNOTATIONS_FILENAME = "annotations.parquet"
SUMMARY_FILENAME = "summary.csv"
SAMPLE_COLUMNS = ("question_id", "model_a", "model_b", "winner", "lang")


class MetaEvalReport(Report):
    """Agreement of one judge with the human votes in a sampled arena slice."""

    task: str
    """Task ID that selected the arena and its language variant."""
    arena: str
    """Arena supplying the human votes."""
    judge_model: str
    """Judge under evaluation."""
    languages: list[str]
    """Language codes the sample was restricted to (empty means all)."""
    top_models: list[str]
    """Most-battled models; only battles between two of them are sampled."""
    n_battles: int
    """Sampled battles, counting a battle once per model it was drawn for."""
    n_annotations: int
    """Judge passes written to annotations.parquet (one or two per sampled battle)."""
    swap_mode: str
    """Position-bias handling used for judging: "fixed" or "both"."""
    battles_per_language: dict[str, int]
    """Sampled battle count per language code."""
    human_winner_counts: dict[str, int]
    """Sampled battle count per human verdict (model_a, model_b, tie)."""
    agreement: dict[str, dict[str, float | int | str]]
    """Accuracy and Cohen's kappa on all battles and on the no-human-tie subset."""
    language_summary: dict[str, dict[str, str | int]]
    """English vs multilingual ranking agreement, fit on forward-order rows."""
    elo_gap_all: list[dict[str, float | int | str | bool]]
    """Held-out Elo MAE vs annotation budget, keeping LLM-predicted ties."""
    elo_gap_exclude_ties: list[dict[str, float | int | str | bool]]
    """Held-out Elo MAE vs annotation budget, dropping LLM-predicted ties after sampling."""
    judge_passes_per_battle: int
    """1 for swap_mode=fixed, 2 for swap_mode=both."""
    estimated_input_tokens: int
    """Sum of chars/4 token estimates over judge inputs."""
    estimated_output_tokens: int
    """Sum of chars/4 token estimates over judge completions."""
    token_count_source: str
    """How tokens were counted (chars/4; never a provider usage API)."""
    total_cost_usd: float
    """OpenRouter reference price applied to the token estimates; NaN (written as
    null) when no local pricing covers the judge."""
    cost_per_1k_judgements_usd: float
    """Mean estimated USD per 1,000 judge passes, NaN if pricing is missing."""
    cost_source_counts: dict[str, int]
    """Per-row cost_source value counts (estimated vs unavailable)."""

    def render(self) -> None:
        print(f"\n=== Meta-eval: {self.task} ===")
        print(f"Arena: {self.arena}  |  Judge: {self.judge_model}")
        print(
            f"Models: {len(self.top_models)}  |  Battles: {self.n_battles}  |  "
            f"Judge passes: {self.n_annotations} ({self.swap_mode})"
        )
        print(f"  Languages: {_format_counts(self.battles_per_language)}")
        print(f"  Human votes: {_format_counts(self.human_winner_counts)}")
        all_view = self.agreement["all"]
        no_tie = self.agreement["no_human_ties"]
        print(
            f"  Agreement (all, n={all_view['n']}): "
            f"acc {all_view['accuracy_formatted']}  "
            f"κ {all_view['kappa_formatted']}"
        )
        print(
            f"  Agreement (no human ties, n={no_tie['n']}): "
            f"acc {no_tie['accuracy_formatted']}  "
            f"κ {no_tie['kappa_formatted']}"
        )
        for split, metrics in self.language_summary.items():
            print(
                f"  {split} (n={metrics['n']}): κ {metrics['kappa']}  "
                f"ρ {metrics['spearman']}  MAE {metrics['mae_elo']}"
            )
        print(
            f"  Tokens (chars/4): {self.estimated_input_tokens} in / "
            f"{self.estimated_output_tokens} out"
        )
        if math.isnan(self.total_cost_usd):
            print("  Cost: n/a (no local OpenRouter reference pricing)")
        else:
            print(
                f"  Cost (OpenRouter reference): ${self.total_cost_usd:.4f}  "
                f"(${self.cost_per_1k_judgements_usd:.4f}/1k passes)"
            )


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def run_meta_eval(cfg: RunConfig, task: ResolvedTaskSpec | None = None) -> dict:
    """Sample arena battles, judge them, and report agreement with human labels."""
    protocol = task.spec.protocol if task is not None else None
    if not isinstance(protocol, MetaEvalProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define a meta-eval protocol.")
    if cfg.meta_eval is None:
        raise ValueError(f"Task {cfg.task!r} requires meta-eval runtime settings.")

    run_started_at = datetime.now(UTC)
    languages = resolve_task_languages(
        task, cfg.meta_eval.languages, setting="meta_eval.languages"
    )

    logger.info("Loading human battles from %s", protocol.arena)
    battles = load_battles(task)
    battles["winner"] = battles["winner"].map(normalize_human_winner)
    if languages:
        battles = battles[battles["lang"].isin(languages)]
    if battles.empty:
        raise MetaEvalSamplingError(
            f"Task {cfg.task!r} has no battles in languages {languages}."
        )

    top_models, df_top = select_top_models(battles, top_models=cfg.meta_eval.top_models)
    sample = sample_battles_per_model(
        df_top,
        top_models,
        battles_per_model=cfg.meta_eval.battles_per_model,
        seed=cfg.run.seed,
    )
    logger.info(
        "Sampled %d battles among the top %d models.", len(sample), len(top_models)
    )

    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
    annotations = annotate_sample(
        sample,
        cfg,
        judge_chat_model=build_judge(cfg),
        resolved_prompt=resolved_prompt,
    )
    metrics = compute_agreement_metrics(
        annotations["winner"].tolist(),
        annotations["winner_llm"].tolist(),
        n_bootstraps=cfg.meta_eval.n_bootstraps,
        seed=cfg.run.seed,
    )
    agreement = {
        "all": agreement_view(metrics, exclude_human_ties=False),
        "no_human_ties": agreement_view(metrics, exclude_human_ties=True),
    }
    scorer = META_EVAL_SCORERS[protocol.scoring.adapter]
    ranking_ann = ranking_annotations(annotations)
    language_summary = scorer.language_splits(
        ranking_ann,
        exclude_human_ties=not cfg.meta_eval.include_human_ties,
        n_bootstraps=cfg.meta_eval.n_bootstraps,
        seed=cfg.run.seed,
    )
    elo_gap_kwargs = {
        "df_top": df_top,
        "df_ann": ranking_ann,
        "top_models": top_models,
        "n_battles_list": list(cfg.meta_eval.elo_gap_battles),
        "n_seeds": cfg.meta_eval.elo_gap_seeds,
    }
    elo_gap_all = scorer.elo_gap(
        **elo_gap_kwargs, seed=cfg.run.seed, exclude_ties=False
    )
    elo_gap_no_tie = scorer.elo_gap(
        **elo_gap_kwargs, seed=cfg.run.seed + 1000, exclude_ties=True
    )
    telemetry = annotation_telemetry(annotations, swap_mode=cfg.judge.swap_mode)

    report = MetaEvalReport(
        task=cfg.task,
        arena=protocol.arena,
        judge_model=cfg.judge.model,
        languages=languages,
        top_models=top_models,
        n_battles=len(sample),
        n_annotations=len(annotations),
        swap_mode=cfg.judge.swap_mode,
        battles_per_language=sample["lang"].value_counts().to_dict(),
        human_winner_counts=sample["winner"].value_counts().to_dict(),
        agreement=agreement,
        language_summary=language_summary,
        elo_gap_all=elo_gap_all.to_dict(orient="records"),
        elo_gap_exclude_ties=elo_gap_no_tie.to_dict(orient="records"),
        **telemetry,
    )
    results = report.to_dict()
    report.render()

    res_dir = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder)
        / f"{safe_filename(cfg.task)}-{safe_filename(cfg.judge.model)}",
    )
    result_path = report.save(res_dir / "results.json")
    sample[list(SAMPLE_COLUMNS)].to_parquet(res_dir / SAMPLE_FILENAME, index=False)
    annotations.to_parquet(res_dir / ANNOTATIONS_FILENAME, index=False)
    pd.DataFrame(
        [{"split": split, **metrics} for split, metrics in language_summary.items()]
    ).to_csv(res_dir / SUMMARY_FILENAME, index=False)
    write_run_metadata_safely(
        output_dir=res_dir,
        entrypoint="judgearena.benchmarks.meta_eval.runner.run_meta_eval",
        run=cfg.model_dump(),
        results=results,
        input_payloads={"question_id": sample["question_id"].astype(str).tolist()},
        judge_system_prompt=resolved_prompt.system_prompt,
        judge_user_prompt_template=resolved_prompt.user_prompt_template,
        started_at_utc=run_started_at,
    )
    logger.info("Meta-eval results written to %s", res_dir)
    return {**results, "result_path": str(result_path)}
