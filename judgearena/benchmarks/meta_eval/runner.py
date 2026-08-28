"""Score a judge against human-labeled arena battles."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts import (
    atomic_write_path,
    prepare_unique_run_directory,
    scoped_run_file_logging,
    write_run_metadata_safely,
)
from judgearena.benchmarks.arena import resolve_task_languages
from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.meta_eval.annotate import (
    aggregate_battle_preferences,
    annotate_sample,
    validate_battle_conversations,
)
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.benchmarks.meta_eval.scoring import (
    resolve_meta_eval_scorer,
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
BATTLE_RESULTS_FILENAME = "battle-results.parquet"
SUMMARY_FILENAME = "summary.csv"
SAMPLE_COLUMNS = (
    "battle_id",
    "question_id",
    "model_a",
    "model_b",
    "winner",
    "lang",
)


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
    """Unique sampled physical battles."""
    n_annotations: int
    """Judge passes attempted (one or two per unique sampled battle)."""
    n_parsed_annotations: int
    """Judge passes whose output parsed successfully."""
    n_scored_battles: int
    """Physical battles with at least one parsed pass and included in metrics."""
    battle_parse_status: dict[str, int]
    """Physical-battle counts with complete, partial, or missing parses."""
    swap_mode: str
    """Position-bias handling used for judging: "fixed" or "both"."""
    battles_per_language: dict[str, int]
    """Sampled battle count per language code."""
    human_winner_counts: dict[str, int]
    """Sampled battle count per human verdict (model_a, model_b, tie)."""
    agreement: dict[str, dict[str, float | int | str]]
    """Accuracy and Cohen's kappa on all battles and on the no-human-tie subset."""
    language_summary: dict[str, dict[str, str | int]]
    """English vs multilingual ranking agreement on physical-battle results."""
    elo_gap_all: list[dict[str, float | int | str | bool]]
    """Held-out Elo MAE vs annotation budget, keeping LLM-predicted ties."""
    elo_gap_exclude_ties: list[dict[str, float | int | str | bool]]
    """Held-out Elo MAE vs annotation budget, dropping LLM-predicted ties after sampling."""
    elo_gap_soft: list[dict[str, float | int | str | bool]]
    """Held-out Elo MAE vs annotation budget using continuous judge preferences."""

    def render(self) -> None:
        print(f"\n=== Meta-eval: {self.task} ===")
        print(f"Arena: {self.arena}  |  Judge: {self.judge_model}")
        print(
            f"Models: {len(self.top_models)}  |  Battles: {self.n_battles}  |  "
            f"Judge passes: {self.n_annotations} ({self.swap_mode})"
        )
        print(
            f"  Parsed passes: {self.n_parsed_annotations}/{self.n_annotations}  |  "
            f"Scored battles: {self.n_scored_battles}/{self.n_battles}"
        )
        print(f"  Parse status: {_format_counts(self.battle_parse_status)}")
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
            print(f"  {split} (n={metrics['n']}): κ {metrics['kappa']}")
            print(
                f"    Hard ranking: ρ {metrics['spearman']}  MAE {metrics['mae_elo']}"
            )
            print(
                f"    Soft ranking: ρ {metrics['spearman_soft']}  "
                f"MAE {metrics['mae_soft_elo']}"
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
    battles = load_battles(task).copy()
    required_columns = {
        "question_id",
        "model_a",
        "model_b",
        "winner",
        "lang",
        "conversation_a",
        "conversation_b",
    }
    missing_columns = sorted(required_columns - set(battles.columns))
    if missing_columns:
        raise MetaEvalSamplingError(
            f"Task {cfg.task!r} is missing required battle columns: {missing_columns}."
        )
    for column in ("question_id", "model_a", "model_b", "winner", "lang"):
        if battles[column].isna().any():
            raise MetaEvalSamplingError(
                f"Task {cfg.task!r} contains null values in {column}."
            )
    if (
        not battles["model_a"]
        .map(lambda model: isinstance(model, str) and bool(model))
        .all()
        or not battles["model_b"]
        .map(lambda model: isinstance(model, str) and bool(model))
        .all()
    ):
        raise MetaEvalSamplingError(
            f"Task {cfg.task!r} contains invalid model identifiers."
        )

    battles["battle_id"] = protocol.arena + ":" + battles["question_id"].astype(str)
    if battles["battle_id"].duplicated().any():
        raise MetaEvalSamplingError(
            f"Task {cfg.task!r} contains duplicate physical battle IDs."
        )
    invalid_winners = sorted(
        set(
            battles.loc[
                ~battles["winner"].isin({"model_a", "model_b", "tie"}), "winner"
            ]
        )
    )
    if invalid_winners:
        raise MetaEvalSamplingError(
            f"Task {cfg.task!r} contains invalid human winners: {invalid_winners}."
        )
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
    validate_battle_conversations(sample)
    logger.info(
        "Sampled %d battles among the top %d models.", len(sample), len(top_models)
    )

    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
    if resolved_prompt.delegated:
        raise ValueError(
            f"Meta-eval cannot use delegated prompt preset "
            f"{resolved_prompt.preset_name!r}."
        )
    scorer = resolve_meta_eval_scorer(protocol.scoring.adapter)
    res_dir = prepare_unique_run_directory(
        cfg,
        cfg.run.result_folder,
        task=cfg.task,
    )
    with scoped_run_file_logging(cfg, res_dir):
        atomic_write_path(
            res_dir / SAMPLE_FILENAME,
            lambda path: sample[list(SAMPLE_COLUMNS)].to_parquet(path, index=False),
        )

        annotations = annotate_sample(
            sample,
            cfg,
            judge_chat_model=build_judge(cfg),
            resolved_prompt=resolved_prompt,
        )
        atomic_write_path(
            res_dir / ANNOTATIONS_FILENAME,
            lambda path: annotations.to_parquet(path, index=False),
        )
        battle_results = aggregate_battle_preferences(
            annotations, swap_mode=cfg.judge.swap_mode
        )
        atomic_write_path(
            res_dir / BATTLE_RESULTS_FILENAME,
            lambda path: battle_results.to_parquet(path, index=False),
        )

        scoring = scorer(
            df_top,
            battle_results,
            top_models,
            n_bootstraps=cfg.meta_eval.n_bootstraps,
            include_human_ties=cfg.meta_eval.include_human_ties,
            elo_gap_battles=list(cfg.meta_eval.elo_gap_battles),
            elo_gap_seeds=cfg.meta_eval.elo_gap_seeds,
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
            n_parsed_annotations=int(annotations["parse_ok"].sum()),
            n_scored_battles=int(battle_results["parse_ok"].sum()),
            battle_parse_status=battle_results["parse_status"].value_counts().to_dict(),
            swap_mode=cfg.judge.swap_mode,
            battles_per_language=sample["lang"].value_counts().to_dict(),
            human_winner_counts=sample["winner"].value_counts().to_dict(),
            agreement=scoring["agreement"],
            language_summary=scoring["language_summary"],
            elo_gap_all=scoring["elo_gap_all"],
            elo_gap_exclude_ties=scoring["elo_gap_exclude_ties"],
            elo_gap_soft=scoring["elo_gap_soft"],
        )
        results = report.to_dict()
        report.render()

        result_path = res_dir / "results.json"
        atomic_write_path(result_path, lambda path: report.save(path))
        summary = pd.DataFrame(
            [
                {"split": split, **metrics}
                for split, metrics in scoring["language_summary"].items()
            ]
        )
        atomic_write_path(
            res_dir / SUMMARY_FILENAME,
            lambda path: summary.to_csv(path, index=False),
        )
        write_run_metadata_safely(
            output_dir=res_dir,
            entrypoint="judgearena.benchmarks.meta_eval.runner.run_meta_eval",
            run=cfg.model_dump(),
            results=results,
            input_payloads={
                "battle_id": sample["battle_id"].astype(str).tolist(),
                "question_id": sample["question_id"].astype(str).tolist(),
            },
            judge_system_prompt=resolved_prompt.system_prompt,
            judge_user_prompt_template=resolved_prompt.user_prompt_template,
            started_at_utc=run_started_at,
        )
        logger.info("Meta-eval results written to %s", res_dir)
        return {**results, "result_path": str(result_path)}
