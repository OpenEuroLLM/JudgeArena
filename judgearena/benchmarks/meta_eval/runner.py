"""Select the human-labeled battles a judge meta-evaluation will score."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from judgearena.artifacts import (
    prepare_run_directory,
    safe_filename,
    write_run_metadata_safely,
)
from judgearena.benchmarks.arena import resolve_task_languages
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    normalize_human_winner,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.datasets import load_battles
from judgearena.log import get_logger
from judgearena.tasks.schema import MetaEvalProtocol, ResolvedTaskSpec
from judgearena.utils.eval import Report

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)

SAMPLE_FILENAME = "sample.parquet"
# Identity and label columns only: the conversations stay in the pinned arena
# snapshot, and question_id joins back to them.
SAMPLE_COLUMNS = ("question_id", "model_a", "model_b", "winner", "lang")


class MetaEvalSampleReport(Report):
    """The human-labeled battle sample a meta-evaluation run will judge."""

    task: str
    """Task ID that selected the arena and its language variant."""
    arena: str
    """Arena supplying the human votes."""
    judge_model: str
    """Judge the sample was drawn for."""
    languages: list[str]
    """Language codes the sample was restricted to (empty means all)."""
    top_models: list[str]
    """Most-battled models; only battles between two of them are sampled."""
    n_battles: int
    """Sampled battles, counting a battle once per model it was drawn for."""
    battles_per_language: dict[str, int]
    """Sampled battle count per language code."""
    human_winner_counts: dict[str, int]
    """Sampled battle count per human verdict (model_a, model_b, tie)."""

    def render(self) -> None:
        print(f"\n=== Meta-eval sample: {self.task} ===")
        print(f"Arena: {self.arena}  |  Judge: {self.judge_model}")
        print(f"Models: {len(self.top_models)}  |  Battles: {self.n_battles}")
        print(f"  Languages: {_format_counts(self.battles_per_language)}")
        print(f"  Human votes: {_format_counts(self.human_winner_counts)}")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))


def run_meta_eval(cfg: RunConfig, task: ResolvedTaskSpec | None = None) -> dict:
    """Sample the battles a judge will be scored against."""
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

    report = MetaEvalSampleReport(
        task=cfg.task,
        arena=protocol.arena,
        judge_model=cfg.judge.model,
        languages=languages,
        top_models=top_models,
        n_battles=len(sample),
        battles_per_language=sample["lang"].value_counts().to_dict(),
        human_winner_counts=sample["winner"].value_counts().to_dict(),
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
    write_run_metadata_safely(
        output_dir=res_dir,
        entrypoint="judgearena.benchmarks.meta_eval.runner.run_meta_eval",
        run=cfg.model_dump(),
        results=results,
        input_payloads={"question_id": sample["question_id"].astype(str).tolist()},
        started_at_utc=run_started_at,
    )
    logger.info("Meta-eval sample written to %s", res_dir)
    return {**results, "result_path": str(result_path)}
