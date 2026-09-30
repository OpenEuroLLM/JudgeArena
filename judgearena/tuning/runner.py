"""Successive-halving search over judge configurations on meta-eval battles.

Follows Salinas et al., "Tuning LLM Judge Design Decisions for 1/1000 of the
Cost" (ICML 2025). Each rung judges more battles per model with the surviving
configurations and keeps the best by a non-dominated sort on judge cost and
human agreement. The search only sees the validation split; the best
configuration per judge model is then scored once on the held-out test split.
"""

from __future__ import annotations

import json
import math
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
from judgearena.tuning.search_space import apply_overrides, expand_grid
from judgearena.tuning.selection import select_survivors

if TYPE_CHECKING:
    from judgearena.tasks.schema import ResolvedTaskSpec

logger = get_logger(__name__)

TrialExecutor = Callable[[Path], None]
"""Run the trial config at a path, raising ``CalledProcessError`` on failure."""

_SUMMARY_COLUMNS = ["trial_id", "judge_model", "agreement", "cost_per_1k_battles"]


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
    overrides: dict[str, dict[str, object]]
    prices: dict[str, float]
    execute_trial: TrialExecutor

    def config(
        self, trial_id: str, *, split: str, battles_per_model: int, folder: Path
    ) -> RunConfig:
        values = apply_overrides(
            self.base,
            {
                **self.overrides[trial_id],
                "meta_eval.split": split,
                "meta_eval.battles_per_model": battles_per_model,
                "run.result_folder": str(folder),
            },
        )
        return RunConfig(**values)

    def run(
        self, trial_id: str, *, split: str, battles_per_model: int, folder: Path
    ) -> dict:
        trial_cfg = self.config(
            trial_id, split=split, battles_per_model=battles_per_model, folder=folder
        )
        record = {
            "trial_id": trial_id,
            "judge_model": trial_cfg.judge.model,
            "overrides": json.dumps(self.overrides[trial_id], sort_keys=True),
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


def _successive_halving(
    runner: _TrialRunner, tuning: TuneJudgeArgs, tune_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return every rung record and the completed survivors of the last rung."""
    alive = list(runner.overrides)
    records = []
    for rung, battles_per_model in enumerate(tuning.rungs):
        rung_records = [
            {
                "rung": rung,
                **runner.run(
                    trial_id,
                    split="validation",
                    battles_per_model=battles_per_model,
                    folder=tune_dir / f"rung-{rung}" / trial_id,
                ),
            }
            for trial_id in alive
        ]
        records.extend(rung_records)
        completed = pd.DataFrame(
            [record for record in rung_records if record["status"] == "completed"]
        )
        if completed.empty:
            raise RuntimeError(f"Every judge configuration failed at rung {rung}.")
        is_last = rung == len(tuning.rungs) - 1
        survivors = select_survivors(
            completed["agreement"].to_numpy(dtype=float),
            completed["cost_per_1k_battles"].to_numpy(dtype=float),
            n_keep=(
                len(completed)
                if is_last
                else math.ceil(len(alive) * tuning.keep_fraction)
            ),
            min_agreement=tuning.min_agreement,
        )
        if not survivors:
            raise RuntimeError(
                f"No configuration exceeded min_agreement at rung {rung}."
            )
        alive = completed["trial_id"].iloc[survivors].tolist()
        logger.info("Rung %d kept %d of %d.", rung, len(alive), len(rung_records))
    return pd.DataFrame(records), completed.iloc[survivors]


def run_tune_judge(
    cfg: RunConfig,
    task: ResolvedTaskSpec | None = None,
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
    runner = _TrialRunner(
        base=cfg.model_dump(mode="json", exclude={"tune_judge"}),
        overrides={
            trial.id: trial.overrides for trial in expand_grid(tuning.search_space)
        },
        prices=tuning.price_per_million_tokens,
        execute_trial=execute_trial,
    )
    # Build every trial before judging so invalid settings fail up front.
    judge_models = {
        runner.config(
            trial_id, split="validation", battles_per_model=1, folder=tune_dir
        ).judge.model
        for trial_id in runner.overrides
    }
    missing_prices = sorted(judge_models - set(tuning.price_per_million_tokens))
    if missing_prices:
        raise ValueError(
            f"tune_judge.price_per_million_tokens is missing models: {missing_prices}"
        )
    logger.info("Tuning %d judge configurations in %s", len(runner.overrides), tune_dir)

    trials, final = _successive_halving(runner, tuning, tune_dir)
    trials.to_parquet(tune_dir / "trials.parquet", index=False)
    picks = final.loc[final.groupby("judge_model")["agreement"].idxmax(), "trial_id"]
    test_results = pd.DataFrame(
        [
            runner.run(
                trial_id,
                split="test",
                battles_per_model=tuning.test_battles_per_model or tuning.rungs[-1],
                folder=tune_dir / "test" / trial_id,
            )
            for trial_id in picks
        ]
    )
    test_results.to_parquet(tune_dir / "test_results.parquet", index=False)

    print(f"\n=== Judge tuning: {cfg.task} ===")
    print("Validation (last rung):")
    ranked = final.sort_values("agreement", ascending=False)
    print(ranked[_SUMMARY_COLUMNS].to_string(index=False))
    print("Test:")
    summary = test_results.reindex(columns=[*_SUMMARY_COLUMNS, "status"])
    print(summary.to_string(index=False))
    print(f"Results: {tune_dir}")
    return test_results
