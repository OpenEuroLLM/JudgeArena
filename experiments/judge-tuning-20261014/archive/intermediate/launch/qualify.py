"""Capped Stage 0: one model process, reused backend, frozen validation panels."""

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pandas as pd

from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.meta_eval.runner import run_meta_eval
from judgearena.config import RunConfig, dump_config, load_config
from judgearena.pricing import TokenPrice
from judgearena.tasks.registry import get_packaged_task
from judgearena.tuning.runner import _read_trial
from judgearena.tuning.search_space import apply_overrides
from judgearena.usage import track_usage

ROOT = Path("/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014")


def model_checks(index):
    manifest = json.loads((ROOT / "configs/manifest.json").read_text())
    model = manifest["models"][index]
    base = load_config(ROOT / "configs/calibration-base.yaml").model_dump(mode="json")
    base["judge"]["model"] = model
    prepared = build_judge(RunConfig(**base))
    task = deepcopy(get_packaged_task(base["task"]))
    # Calibration needs agreement only; bootstrap uncertainty is computed jointly later.
    scoring = task.spec.protocol.scoring
    metric = next(m for m in scoring.metrics if m.metric == "meta_eval_agreement")
    metric.parameters["n_bootstraps"] = 0
    task = replace(
        task,
        spec=task.spec.model_copy(
            update={
                "protocol": task.spec.protocol.model_copy(
                    update={
                        "scoring": scoring.model_copy(update={"metrics": (metric,)})
                    }
                )
            }
        ),
    )
    price = TokenPrice(**manifest["prices"][model], source="frozen_v1_reference")
    cohort = [
        row for row in manifest["cohort"] if row["overrides"]["judge.model"] == model
    ]
    checks = [
        ("qualification", preset, "fixed", 1, None) for preset in manifest["presets"]
    ]
    checks += [
        (
            "calibration",
            row["overrides"]["judge.prompt_preset"],
            row["overrides"]["judge.swap_mode"],
            2,
            row["config_id"],
        )
        for row in cohort
    ]
    records = []
    for kind, preset, order, quota, config_id in checks:
        folder = ROOT / kind / f"m{index}-{preset}-{order}"
        folder.mkdir(parents=True, exist_ok=True)
        cfg = RunConfig(
            **apply_overrides(
                base,
                {
                    "judge.prompt_preset": preset,
                    "judge.swap_mode": order,
                    "meta_eval.battles_per_model": quota,
                    "run.result_folder": str(folder),
                },
            )
        )
        dump_config(cfg, folder / "config.yaml")
        (folder / "calibration-scoring.json").write_text(
            json.dumps(
                {
                    "metrics": ["meta_eval_agreement"],
                    "n_bootstraps": 0,
                    "reason": "Joint calibration uncertainty calculated after inference; normal search metrics unchanged",
                }
            )
        )
        start = time.monotonic()
        with track_usage() as tracker:
            if not list(folder.glob("*/results.json")):
                run_meta_eval(cfg, task, judge_chat_model=prepared)
        row = {
            "kind": kind,
            "model": model,
            "preset": preset,
            "order": order,
            "config_id": config_id,
            "elapsed_s": time.monotonic() - start,
            **_read_trial(folder, price),
        }
        annotations = pd.read_parquet(Path(row["result_dir"]) / "annotations.parquet")
        completions = annotations.judge_completion.fillna("")
        row.update(
            n_annotations=len(annotations),
            reasoning_markers=int(
                completions.str.contains(
                    r"<think>|<\|channel>thought", regex=True
                ).sum()
            ),
            finished_reasoning=int(
                completions.str.contains(r"</think>|<channel\|>", regex=True).sum()
            ),
            output_cap_requests=int(
                annotations.usage_json.fillna("{}")
                .map(
                    lambda value: (
                        (json.loads(value) or {}).get("output_tokens", 0) >= 8192
                    )
                )
                .sum()
            ),
        )
        row["fresh_usage"] = tracker.snapshot().to_dict()
        records.append(row)
        (ROOT / f"model-{index}-progress.json").write_text(
            json.dumps(records, indent=2, default=str)
        )
        print(json.dumps(row, default=str), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int)
    args = parser.parse_args()
    if args.model_index is not None:
        model_checks(args.model_index)
        return
    started = time.time()
    outcomes = []
    for index in [2, 0, 1]:
        with (ROOT / f"model-{index}.log").open("a") as log:
            result = subprocess.run(
                [sys.executable, __file__, "--model-index", str(index)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        outcomes.append({"model_index": index, "exit_code": result.returncode})
        elapsed = time.time() - started
        (ROOT / "stage0-progress.json").write_text(
            json.dumps(
                {
                    "job_id": os.getenv("SLURM_JOB_ID"),
                    "elapsed_s": elapsed,
                    "generation_phase_l40s_hours": elapsed * 4 / 3600,
                    "outcomes": outcomes,
                },
                indent=2,
            )
        )
        if result.returncode:
            raise RuntimeError(
                f"Stage0 model process {index} failed; artifacts preserved"
            )


if __name__ == "__main__":
    main()
