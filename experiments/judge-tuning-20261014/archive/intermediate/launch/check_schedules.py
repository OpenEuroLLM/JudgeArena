"""Real runner/NePS/GP schedule checks with synthetic outcomes, no inference."""

import json
from collections import Counter
from pathlib import Path

import neps.optimizers.primo as primo_module
import pandas as pd

from judgearena.config import RunConfig, load_config
from judgearena.tuning.runner import run_tune_judge

LOCAL = Path("/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014")
manifest = json.loads((LOCAL / "configs/manifest.json").read_text())
panels = json.loads((LOCAL / "panels/panels.json").read_text())["validation"]


def check(arm, seed):
    base = load_config(LOCAL / "configs/calibration-base.yaml").model_dump(mode="json")
    root = LOCAL / "schedules-verified" / f"{arm}-{seed}"
    base["task"] = "tune-judge"
    base["run"].update(result_folder=str(root / "results"), store_root=None)
    base["meta_eval"]["panel_file"] = str(LOCAL / "panels/panels.json")
    centers = (
        {"agreement": manifest["agreement_prior"], "cost": manifest["cost_prior"]}
        if arm == "primo-prior"
        else {}
    )
    optimizer = (
        {
            "name": "mo_hyperband",
            "sampler": "uniform",
            "eta": 3,
            "mo_selector": "epsnet",
        }
        if arm == "uniform"
        else {
            "name": "primo",
            "sampler": "uniform",
            "eta": 3,
            "mo_selector": "epsnet",
            "initial_design_size": 2,
            "device": "cpu",
            "prior_centers": centers,
            "prior_confidences": {
                objective: {parameter: 0.5 for parameter in center}
                for objective, center in centers.items()
            },
        }
    )
    base["tune_judge"] = {
        "meta_eval_task": "meta-eval-lmarena-140k-en",
        "objectives": ["agreement", "cost"],
        "objective_bounds": manifest["objective_bounds"],
        "fidelity": {"battles_per_model": {"lower": 20, "upper": 180}},
        "search_space": {
            "judge": {
                "model": manifest["models"],
                "prompt_preset": manifest["presets"],
                "swap_mode": ["fixed", "both"],
            }
        },
        "optimizer": optimizer,
        "neps": {"total_fidelities_to_spend": 1440, "ignore_errors": False},
        "price_per_million_tokens": manifest["prices"],
        "skip_test": True,
        "seed": seed,
        "run_dir": str(root / "tuning"),
    }
    cfg = RunConfig(**base)
    trials, acquisitions, proposals = [], [], []

    def fake_execute(path):
        c = load_config(path)
        quota = c.meta_eval.battles_per_model
        model_index = manifest["models"].index(c.judge.model)
        prompt_index = manifest["presets"].index(c.judge.prompt_preset)
        agreement = (
            0.41
            + model_index * 0.017
            + prompt_index * 0.002
            + 0.004 * (c.judge.swap_mode == "both")
        )
        folder = path.parent / "synthetic"
        folder.mkdir()
        n = len(panels[str(quota)])
        (folder / "results.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "meta_eval_agreement": {
                            "all": {
                                "accuracy_attempted": agreement,
                                "cohen_kappa": 0.2,
                                "coverage": 1.0,
                                "n_attempted": n,
                            }
                        }
                    }
                }
            )
        )
        usage = json.dumps(
            {
                "stage": "judging",
                "model": c.judge.model,
                "input_tokens": 4000,
                "output_tokens": 800 + prompt_index * 100,
            }
        )
        orientations = 2 if c.judge.swap_mode == "both" else 1
        pd.DataFrame(
            {
                "battle_id": list(range(n)) * orientations,
                "usage_json": [usage] * n * orientations,
                "error": [None] * n * orientations,
            }
        ).to_parquet(folder / "annotations.parquet", index=False)
        trials.append(
            {
                "quota": quota,
                "n_battles": n,
                "model": c.judge.model,
                "preset": c.judge.prompt_preset,
                "order": c.judge.swap_mode,
            }
        )

    native_fit, native_bo = (
        primo_module.fit_and_acquire_from_gp,
        primo_module.PriMO.sample_using_bo,
    )

    def observe_fit(*args, **kwargs):
        result = native_fit(*args, **kwargs)
        acquisitions.append(
            {
                "prior_nonnull": kwargs.get("prior") is not None,
                "original_gp_returned": True,
            }
        )
        return result

    def observe_bo(self, *args, **kwargs):
        result = native_bo(self, *args, **kwargs)
        proposals.append(
            {"trial_index": len(trials), "priors_nonnull": self.priors is not None}
        )
        return result

    primo_module.fit_and_acquire_from_gp, primo_module.PriMO.sample_using_bo = (
        observe_fit,
        observe_bo,
    )
    try:
        run_tune_judge(cfg, None, execute_trial=fake_execute)
    finally:
        primo_module.fit_and_acquire_from_gp, primo_module.PriMO.sample_using_bo = (
            native_fit,
            native_bo,
        )
    if arm == "primo-none":
        assert acquisitions and not any(a["prior_nonnull"] for a in acquisitions)
    elif arm == "primo-prior":
        # Native epsilon=.25 deliberately omits the prior for some BO proposals.
        assert acquisitions and any(a["prior_nonnull"] for a in acquisitions)
    return {
        "arm": arm,
        "seed": seed,
        "schedule": dict(Counter(t["quota"] for t in trials)),
        "actual_fidelity_units": sum(t["quota"] for t in trials),
        "actual_config_battles": sum(t["n_battles"] for t in trials),
        "trials": trials,
        "bo_proposals": proposals,
        "gp_acquisitions": acquisitions,
        "no_model_inference": True,
    }


if __name__ == "__main__":
    results = []
    for arm in ["uniform", "primo-none", "primo-prior"]:
        for seed in [0]:
            results.append(check(arm, seed))
            (LOCAL / "analysis/schedules.json").write_text(
                json.dumps(results, indent=2)
            )
            print(
                arm,
                seed,
                results[-1]["schedule"],
                results[-1]["actual_config_battles"],
                "BO",
                len(results[-1]["bo_proposals"]),
                flush=True,
            )
