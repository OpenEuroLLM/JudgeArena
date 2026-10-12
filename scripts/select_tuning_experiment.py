#!/usr/bin/env python3
"""Select validation winners and prepare (or execute) held-out meta-evals."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judgearena.config import dump_config, load_config  # noqa: E402
from judgearena.pricing import TokenPrice  # noqa: E402
from judgearena.tuning.budget import clip_trials_to_common_budget  # noqa: E402
from judgearena.tuning.runner import _read_trial, run_trial_subprocess  # noqa: E402


def select(
    manifest: dict, budget_manifest: dict | None = None
) -> tuple[pd.DataFrame, list[dict], dict]:
    limit = manifest.get("max_battles_per_model", 90)
    all_runs = (budget_manifest or manifest)["runs"]
    trial_tables = {
        str(run["run_dir"]): pd.read_parquet(Path(run["run_dir"]) / "trials.parquet")
        for run in all_runs
    }
    clipped, budget = clip_trials_to_common_budget(
        trial_tables,
        {int(k): v for k, v in (budget_manifest or manifest)["panel_sizes"].items()},
    )
    completed = []
    for run in manifest["runs"]:
        root = Path(run["run_dir"])
        trials = clipped[str(root)]
        finalists = trials[
            (trials.status == "completed") & (trials.battles_per_model == limit)
        ].copy()
        if not finalists.empty:
            finalists["method"] = run["method"]
            finalists["seed"] = run["seed"]
            finalists["run_dir"] = str(root)
            finalists["price_file"] = str(root / "prices.json")
            completed.append(finalists)
    pooled = pd.concat(completed, ignore_index=True) if completed else pd.DataFrame()
    selected, missing = [], []
    for method in dict.fromkeys(run["method"] for run in manifest["runs"]):
        finalists = pooled[pooled.method == method] if not pooled.empty else pooled
        finalists = (
            finalists.drop_duplicates("config_id").sort_values(
                ["agreement", "cost_per_1k_battles", "config_id"],
                ascending=[False, True, True],
                na_position="last",
            )
            if not finalists.empty
            else finalists
        )
        for name in manifest["models"]:
            choice = (
                finalists[finalists.judge_model == name].head(1)
                if not finalists.empty
                else finalists
            )
            if choice.empty:
                missing.append({"method": method, "judge_model": name})
            else:
                selected.append(choice.iloc[0].to_dict())
    return pd.DataFrame(selected), missing, budget


def prepare(manifest: dict, picks: pd.DataFrame, output: Path, execute: bool) -> None:
    results = []
    for pick in picks.to_dict("records"):
        config_path = Path(pick["result_dir"]).parent / "config.yaml"
        cfg = load_config(config_path)
        cfg.meta_eval.split = "test"
        cfg.meta_eval.battles_per_model = manifest["test_battles_per_model"]
        cfg.run.seed = manifest.get("panel_seed", 0)
        folder = output / "tests" / str(pick["method"]) / str(pick["config_id"])
        cfg.run.result_folder = str(folder)
        folder.mkdir(parents=True, exist_ok=True)
        test_config = folder / "config.yaml"
        dump_config(cfg, test_config)
        if execute:
            record = {
                key: pick[key] for key in ("method", "seed", "config_id", "judge_model")
            }
            record["validation_agreement"] = pick["agreement"]
            price_data = json.loads(Path(pick["price_file"]).read_text())
            prices = {model: TokenPrice(**value) for model, value in price_data.items()}
            price = prices.get(cfg.judge.model)
            try:
                if not list(folder.glob("*/results.json")):
                    run_trial_subprocess(test_config)
                results.append(
                    {**record, **_read_trial(folder, price), "status": "completed"}
                )
            except subprocess.CalledProcessError as error:
                results.append({**record, "status": "failed", "error": str(error)})
            pd.DataFrame(results).to_parquet(
                output / "test_results.parquet", index=False
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Run held-out inference")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    budget_path = args.manifest.parent / "manifest.json"
    budget_manifest = (
        json.loads(budget_path.read_text()) if budget_path.exists() else manifest
    )
    picks, missing, budget = select(manifest, budget_manifest)
    picks.to_parquet(args.output / "selections.parquet", index=False)
    (args.output / "selections.json").write_text(
        json.dumps(
            {
                "selections": picks.to_dict("records"),
                "missing_models": missing,
                "budget": budget,
            },
            indent=2,
            default=str,
        )
    )
    prepare(manifest, picks, args.output, args.execute)
    print(f"Selected {len(picks)} configs; missing {len(missing)} model/method pairs.")


if __name__ == "__main__":
    main()
