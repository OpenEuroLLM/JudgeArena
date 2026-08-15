"""Score a judge against human-labeled arena battles."""

from __future__ import annotations

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
    prompt_preset: str
    """Judge prompt preset the verdicts were parsed under."""
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
    language_summary = scorer.language_splits(
        ranking_annotations(annotations),
        exclude_human_ties=not cfg.meta_eval.include_human_ties,
        n_bootstraps=cfg.meta_eval.n_bootstraps,
        seed=cfg.run.seed,
    )

    report = MetaEvalReport(
        task=cfg.task,
        arena=protocol.arena,
        judge_model=cfg.judge.model,
        prompt_preset=resolved_prompt.preset_name,
        languages=languages,
        top_models=top_models,
        n_battles=len(sample),
        n_annotations=len(annotations),
        swap_mode=cfg.judge.swap_mode,
        battles_per_language=sample["lang"].value_counts().to_dict(),
        human_winner_counts=sample["winner"].value_counts().to_dict(),
        agreement=agreement,
        language_summary=language_summary,
    )
    results = report.to_dict()
    report.render()

    # The prompt preset and the swap mode both change the verdicts, so they have
    # to be part of the folder name: otherwise a second run under a different
    # preset or swap mode silently overwrites the first run's artifacts.
    res_dir = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder)
        / f"{safe_filename(cfg.task)}-{safe_filename(resolved_prompt.preset_name)}-"
        f"{safe_filename(cfg.judge.model)}-{cfg.judge.swap_mode}",
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
