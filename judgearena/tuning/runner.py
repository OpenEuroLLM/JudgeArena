"""Multi-fidelity search over judge configurations on meta-eval battles.

Follows Salinas et al., "Tuning LLM Judge Design Decisions for 1/1000 of the
Cost" (ICML 2025), with the fidelity being validation battles per model. A neps
optimizer (priorband by default, seeded with the base config as prior) proposes
each trial, which runs as the meta-eval task so trials share its judgement
cache. The best configuration per judge model at the highest fidelity is then
scored once on the held-out test split.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import tiktoken

from judgearena.artifacts import prepare_run_directory, safe_filename
from judgearena.config import RunConfig, TuneJudgeArgs, dump_config
from judgearena.log import get_logger
from judgearena.tuning.search_space import (
    apply_overrides,
    axis_overrides,
    build_neps_space,
    decode_config,
)

if TYPE_CHECKING:
    from judgearena.tasks.schema import ResolvedTaskSpec

logger = get_logger(__name__)

TrialExecutor = Callable[[Path], None]
"""Run the trial config at a path, raising ``CalledProcessError`` on failure."""

_SUMMARY_COLUMNS = ["config_id", "judge_model", "agreement", "cost_per_1k_battles"]


def run_trial_subprocess(config_path: Path) -> None:
    """Run one trial in a fresh process so each judge releases its GPU memory."""
    subprocess.run(
        [sys.executable, "-m", "judgearena", "--config_path", str(config_path)],
        check=True,
    )


@cache
def _judge_token_encoding() -> tiktoken.Encoding:
    return tiktoken.encoding_for_model("gpt-4o")


def _read_trial(trial_dir: Path, price_per_million_tokens: float) -> dict:
    (result_path,) = trial_dir.glob("*/results.json")
    metrics = json.loads(result_path.read_text())["metrics"]
    agreement = metrics["meta_eval_agreement"]["all"]
    annotations = pd.read_parquet(
        result_path.parent / "annotations.parquet",
        columns=["battle_id", "judge_input", "judge_completion"],
    )
    texts = [*annotations["judge_input"], *annotations["judge_completion"].fillna("")]
    encoding = _judge_token_encoding()
    tokens = sum(len(encoding.encode(text, disallowed_special=())) for text in texts)
    tokens_per_battle = tokens / annotations["battle_id"].nunique()
    return {
        "agreement": agreement["accuracy_attempted"],
        "cohen_kappa": agreement["cohen_kappa"],
        "coverage": agreement["coverage"],
        "n_battles": agreement["n_attempted"],
        "spearman": metrics.get("meta_eval_ranking", {})
        .get("soft", {})
        .get("spearman"),
        "tokens_per_battle": tokens_per_battle,
        "cost_per_1k_battles": tokens_per_battle * price_per_million_tokens / 1000,
        "result_dir": str(result_path.parent),
    }


@dataclass(frozen=True)
class _TrialRunner:
    """Materialize trial configs from one base config and record their results."""

    base: dict
    prices: dict[str, float]
    execute_trial: TrialExecutor

    def config(
        self,
        overrides: dict[str, object],
        *,
        split: str,
        battles_per_model: int,
        folder: Path,
    ) -> RunConfig:
        values = apply_overrides(
            self.base,
            {
                **overrides,
                "meta_eval.split": split,
                "meta_eval.battles_per_model": battles_per_model,
                "run.result_folder": str(folder),
            },
        )
        return RunConfig(**values)

    def run(
        self,
        overrides: dict[str, object],
        *,
        split: str,
        battles_per_model: int,
        folder: Path,
    ) -> dict:
        trial_cfg = self.config(
            overrides, split=split, battles_per_model=battles_per_model, folder=folder
        )
        encoded = json.dumps(overrides, sort_keys=True)
        record = {
            "config_id": hashlib.sha256(encoded.encode()).hexdigest()[:12],
            "judge_model": trial_cfg.judge.model,
            "overrides": encoded,
            "battles_per_model": battles_per_model,
        }
        folder.mkdir(parents=True, exist_ok=True)
        config_path = folder / "config.yaml"
        dump_config(trial_cfg, config_path)
        try:
            self.execute_trial(config_path)
        except subprocess.CalledProcessError as exc:
            logger.warning("Trial %s failed: %s", folder, exc)
            return {**record, "status": "failed"}
        price = self.prices[trial_cfg.judge.model]
        return {**record, "status": "completed", **_read_trial(folder, price)}


def _objective(record: dict, algorithm: str) -> dict | Exception:
    if record["status"] != "completed":
        return RuntimeError(f"Trial {record['config_id']} failed.")
    disagreement = 1 - record["agreement"]
    if algorithm == "mo_hyperband":
        return {"objective_to_minimize": [disagreement, record["cost_per_1k_battles"]]}
    return {"objective_to_minimize": disagreement}


def _search(
    runner: _TrialRunner, tuning: TuneJudgeArgs, tune_dir: Path
) -> pd.DataFrame:
    """Run the neps ask-and-tell loop on the validation split."""
    from neps import AskAndTell
    from neps.optimizers import algorithms

    space = build_neps_space(tuning, runner.base)
    optimizer = AskAndTell(getattr(algorithms, tuning.algorithm)(space, eta=tuning.eta))
    # neps resamples identical configs on small categorical spaces.
    results: dict[tuple[str, int], dict] = {}
    records = []
    for _ in range(tuning.max_evaluations):
        trial = optimizer.ask()
        overrides, battles_per_model = decode_config(trial.config)
        key = (json.dumps(overrides, sort_keys=True), battles_per_model)
        if key not in results:
            results[key] = runner.run(
                overrides,
                split="validation",
                battles_per_model=battles_per_model,
                folder=tune_dir / "trials" / trial.metadata.id,
            )
        records.append({"neps_trial_id": trial.metadata.id, **results[key]})
        optimizer.tell(trial, _objective(results[key], tuning.algorithm))
    return pd.DataFrame(records)


def _meta_eval_task(task: ResolvedTaskSpec) -> str:
    """Return the meta-eval task ID trials run as, keeping the variant suffix."""
    meta_eval_task = task.spec.protocol.meta_eval_task
    if task.selection is None:
        return meta_eval_task
    return f"{meta_eval_task}-{task.selection.name}"


def run_tune_judge(
    cfg: RunConfig,
    task: ResolvedTaskSpec,
    *,
    execute_trial: TrialExecutor = run_trial_subprocess,
) -> pd.DataFrame:
    """Tune judge settings on validation battles and score the picks on test."""
    tuning = cfg.tune_judge
    started_at = datetime.now(UTC)
    tune_dir = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder)
        / f"tune-{safe_filename(cfg.task)}-{started_at:%Y%m%d_%H%M%S}",
    )
    base = cfg.model_dump(mode="json", exclude={"tune_judge"})
    if base["judge"]["prompt"] is None and base["judge"]["prompt_preset"] is None:
        # Make the task's default preset explicit so it can be the prior.
        default_preset = task.spec.protocol.judge.default_prompt_preset
        base["judge"]["prompt_preset"] = default_preset
    runner = _TrialRunner(
        base={**base, "task": _meta_eval_task(task)},
        prices=tuning.price_per_million_tokens,
        execute_trial=execute_trial,
    )
    # Build every choice before judging so invalid settings fail up front.
    judge_models = {cfg.judge.model} | {
        runner.config(
            overrides, split="validation", battles_per_model=1, folder=tune_dir
        ).judge.model
        for overrides in axis_overrides(tuning)
    }
    missing_prices = sorted(judge_models - set(tuning.price_per_million_tokens))
    if missing_prices:
        raise ValueError(
            f"tune_judge.price_per_million_tokens is missing models: {missing_prices}"
        )
    logger.info("Tuning %s with neps %s in %s", cfg.task, tuning.algorithm, tune_dir)

    trials = _search(runner, tuning, tune_dir)
    trials.to_parquet(tune_dir / "trials.parquet", index=False)
    final = trials[
        (trials["status"] == "completed")
        & (trials["battles_per_model"] == tuning.max_battles_per_model)
    ].drop_duplicates("config_id")
    if final.empty:
        raise RuntimeError(
            "No configuration completed the highest fidelity; "
            "increase tune_judge.max_evaluations."
        )
    picks = final.loc[final.groupby("judge_model")["agreement"].idxmax()]
    test_results = pd.DataFrame(
        [
            runner.run(
                json.loads(pick.overrides),
                split="test",
                battles_per_model=(
                    tuning.test_battles_per_model or tuning.max_battles_per_model
                ),
                folder=tune_dir / "test" / pick.config_id,
            )
            for pick in picks.itertuples()
        ]
    )
    test_results.to_parquet(tune_dir / "test_results.parquet", index=False)

    print(f"\n=== Judge tuning: {cfg.task} ({tuning.algorithm}) ===")
    print(f"Validation ({tuning.max_battles_per_model} battles per model):")
    ranked = final.sort_values("agreement", ascending=False)
    print(ranked[_SUMMARY_COLUMNS].to_string(index=False))
    print("Test:")
    summary = test_results.reindex(columns=[*_SUMMARY_COLUMNS, "status"])
    print(summary.to_string(index=False))
    print(f"Results: {tune_dir}")
    return test_results
