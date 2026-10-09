"""Run NePS searches using meta-evaluation subprocesses."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.config import RunConfig, dump_config
from judgearena.log import get_logger
from judgearena.tuning.search_space import (
    apply_overrides,
    axis_overrides,
    build_neps_space,
    decode_config,
    parameter_specs,
)

if TYPE_CHECKING:
    from judgearena.tasks.schema import ResolvedTaskSpec

from judgearena.pricing import TokenPrice, reference_cost, resolve_prices
from judgearena.tuning.session import collect_trials, prepare_session
from judgearena.usage import request_usage_from_json

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


def _read_trial(trial_dir: Path, price: TokenPrice | None) -> dict:
    (result_path,) = trial_dir.glob("*/results.json")
    metrics = json.loads(result_path.read_text())["metrics"]
    agreement = metrics["meta_eval_agreement"]["all"]
    annotations = pd.read_parquet(result_path.parent / "annotations.parquet")
    n_battles = annotations["battle_id"].nunique()
    total_input = total_output = 0
    total_cost = 0.0
    has_usage_columns = {"usage_json", "error"}.issubset(annotations.columns)
    has_complete_usage = has_usage_columns
    if has_usage_columns:
        for row in annotations.itertuples(index=False):
            request_usage = request_usage_from_json(row.usage_json)
            if request_usage is None and row.error == "context_length":
                continue
            input_tokens = request_usage.input_tokens if request_usage else None
            output_tokens = request_usage.output_tokens if request_usage else None
            if input_tokens is None or output_tokens is None:
                has_complete_usage = False
                continue
            total_input += input_tokens
            total_output += output_tokens
            if price is not None:
                total_cost += reference_cost(request_usage, price)
    if price is not None and not has_complete_usage:
        raise ValueError(
            "Cannot calculate tuning cost: annotations are missing native input/output "
            "token usage in "
            f"{result_path.parent / 'annotations.parquet'}. Rerun with a fresh "
            "store_root so usage is recorded, or use agreement-only tuning."
        )
    tokens_per_battle = (
        (total_input + total_output) / n_battles if has_complete_usage else None
    )
    cost = total_cost / n_battles * 1000 if price is not None else None
    return {
        "agreement": agreement["accuracy_attempted"],
        "cohen_kappa": agreement["cohen_kappa"],
        "coverage": agreement["coverage"],
        "n_battles": agreement["n_attempted"],
        "spearman": metrics.get("meta_eval_ranking", {})
        .get("soft", {})
        .get("spearman"),
        "tokens_per_battle": tokens_per_battle,
        "cost_per_1k_battles": cost,
        "result_dir": str(result_path.parent),
    }


@dataclass(frozen=True)
class _TrialRunner:
    """Materialize trial configs from one base config and record their results."""

    base: dict
    prices: dict[str, TokenPrice]
    execute_trial: TrialExecutor
    ignore_failed_trials: bool = False

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
        if split == "test" and any(folder.glob("*/results.json")):
            return {
                **record,
                "status": "completed",
                **_read_trial(folder, self.prices.get(trial_cfg.judge.model)),
            }
        try:
            self.execute_trial(config_path)
        except subprocess.CalledProcessError as exc:
            if not self.ignore_failed_trials:
                raise
            logger.warning("Trial %s failed: %s", folder, exc)
            return {**record, "status": "failed"}
        price = self.prices.get(trial_cfg.judge.model)
        return {**record, "status": "completed", **_read_trial(folder, price)}


def _objective(record: dict, objectives: list[str]) -> dict:
    if record["status"] != "completed":
        return {
            "objective_to_minimize": (
                float("inf")
                if len(objectives) == 1
                else [float("inf")] * len(objectives)
            ),
            "info_dict": record,
        }
    values = [
        1 - record[key] if key == "agreement" else record["cost_per_1k_battles"]
        for key in objectives
    ]
    return {
        "objective_to_minimize": values[0] if len(values) == 1 else values,
        "info_dict": record,
    }


def _search(runner: _TrialRunner, cfg: RunConfig, tune_dir: Path, space) -> None:
    """Let NePS persist and coordinate the search across workers."""
    import neps

    def evaluate(pipeline_directory: Path, **config):
        overrides, battles = decode_config(config)
        record = runner.run(
            overrides,
            split="validation",
            battles_per_model=battles,
            folder=tune_dir / "trials" / pipeline_directory.name,
        )
        return _objective(record, cfg.tune_judge.objectives)

    options = dict(cfg.tune_judge.neps)
    options.pop("ignore_errors", None)
    # NePS 0.17's space compatibility check does not inspect optimizer mappings.
    optimizer = cfg.tune_judge.optimizer
    if isinstance(optimizer, dict):
        options["optimizer"] = (
            optimizer["name"],
            {key: value for key, value in optimizer.items() if key != "name"},
        )
    root = tune_dir / "neps"
    continuing = (root / "pipeline_space.pkl").exists()
    if cfg.tune_judge.search_only:
        options.pop("total_evaluations_to_spend", None)
        options.pop("total_fidelities_to_spend", None)
    elif not continuing:
        import torch

        random.seed(cfg.run.seed)
        np.random.seed(cfg.run.seed)
        torch.manual_seed(cfg.run.seed)
    neps.run(
        evaluate_pipeline=evaluate,
        pipeline_space=None if continuing else space,
        root_directory=root,
        continue_until_max_evaluation_completed=False,
        # Only subprocess failures may be ignored; accounting errors stop the run.
        ignore_errors=False,
        **options,
    )


def run_tune_judge(
    cfg: RunConfig,
    task: ResolvedTaskSpec | None,
    *,
    execute_trial: TrialExecutor = run_trial_subprocess,
) -> pd.DataFrame:
    """Tune judge settings on validation battles and score the picks on test."""
    tuning = cfg.tune_judge
    base = cfg.model_dump(mode="json", exclude={"tune_judge"})
    if base["judge"]["prompt"] is None and base["judge"]["prompt_preset"] is None:
        from judgearena.tasks.registry import get_packaged_task

        base["judge"]["prompt_preset"] = get_packaged_task(
            tuning.meta_eval_task
        ).spec.protocol.judge.default_prompt_preset
    specs = parameter_specs(tuning.search_space, base, tuning.fidelity)
    space = build_neps_space(specs)
    max_battles = tuning.fidelity["battles_per_model"]["upper"]
    runner = _TrialRunner(
        base={**base, "task": tuning.meta_eval_task},
        prices={},
        execute_trial=execute_trial,
    )
    models = set()
    for overrides in axis_overrides(specs):
        trial_cfg = runner.config(
            overrides,
            split="validation",
            battles_per_model=1,
            folder=Path(cfg.run.result_folder),
        )
        if "judge.model" not in specs or "judge.model" in overrides:
            models.add(trial_cfg.judge.model)
    if "judge.model" not in specs:
        models.add(cfg.judge.model)
    tune_dir = prepare_session(cfg)
    prices = {}
    if "cost" in tuning.objectives:
        price_path = tune_dir / "prices.json"
        if price_path.exists():
            prices = {
                model: TokenPrice(**price)
                for model, price in json.loads(price_path.read_text()).items()
            }
        else:
            prices = resolve_prices(
                models,
                tuning.price_per_million_tokens,
                require_cost=True,
            )
            if not tuning.search_only:
                price_path.write_text(
                    json.dumps(
                        {model: asdict(price) for model, price in prices.items()},
                        indent=2,
                    )
                )
    runner = _TrialRunner(
        runner.base,
        prices,
        execute_trial,
        ignore_failed_trials=tuning.neps.get("ignore_errors", False),
    )
    logger.info("Tuning %s in %s", cfg.task, tune_dir)
    _search(runner, cfg, tune_dir, space)
    if tuning.search_only:
        return pd.DataFrame()
    trials = collect_trials(tune_dir / "neps", wait=True)
    trials.to_parquet(tune_dir / "trials.parquet", index=False)
    return _evaluate_picks(runner, tuning, tune_dir, trials, max_battles)


def _evaluate_picks(runner, tuning, tune_dir, trials, max_battles) -> pd.DataFrame:
    final = trials[
        (trials["status"] == "completed") & (trials["battles_per_model"] == max_battles)
    ].drop_duplicates("config_id")
    if final.empty:
        raise RuntimeError(
            "No configuration completed the highest fidelity; increase the NePS global budget"
        )
    ranked = final.sort_values(
        ["agreement", "cost_per_1k_battles", "config_id"], ascending=[False, True, True]
    )
    from neps.optimizers.utils.multiobjective.epsnet import pareto_efficient

    priced = ranked.dropna(subset=["cost_per_1k_battles"])
    costs = priced[["agreement", "cost_per_1k_battles"]].to_numpy(dtype=float).copy()
    costs[:, 0] *= -1
    pareto = priced.loc[pareto_efficient(costs)] if len(priced) else priced
    pareto.to_parquet(tune_dir / "pareto.parquet", index=False)
    picks = ranked.groupby("judge_model", sort=False).head(1)
    test_results = pd.DataFrame(
        [
            runner.run(
                json.loads(pick.overrides),
                split="test",
                battles_per_model=tuning.test_battles_per_model or max_battles,
                folder=tune_dir / "test" / pick.config_id,
            )
            for pick in picks.itertuples()
        ]
    )
    test_results.to_parquet(tune_dir / "test_results.parquet", index=False)
    print("\n=== Judge tuning ===")
    print("Validation:")
    print(ranked[_SUMMARY_COLUMNS].to_string(index=False))
    print("Agreement/cost Pareto front (validation):")
    print(pareto[_SUMMARY_COLUMNS].to_string(index=False))
    print("Test:")
    print(
        test_results.reindex(columns=[*_SUMMARY_COLUMNS, "status"]).to_string(
            index=False
        )
    )
    print(f"Results: {tune_dir}")
    return test_results
