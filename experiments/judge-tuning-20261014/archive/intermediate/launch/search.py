"""Run one approved arm; observe original NePS methods without changing them."""

import argparse, json, sys
from pathlib import Path
import neps.optimizers.primo as module
from judgearena.config import load_config
from judgearena.tuning.runner import run_tune_judge, run_trial_subprocess

ROOT = Path("/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014")
p = argparse.ArgumentParser()
p.add_argument("arm", choices=["uniform", "primo-none", "primo-prior"])
args = p.parse_args()
cfg = load_config(ROOT / "configs" / f"{args.arm}.yaml")
records = {
    "arm": args.arm,
    "execution_attempts": [],
    "bo_proposals": [],
    "gp_acquisitions": [],
}
out = ROOT / "analysis" / f"{args.arm}-optimizer-observations.json"
if out.exists():
    records = json.loads(out.read_text())
    assert records["arm"] == args.arm


def save():
    out.write_text(json.dumps(records, indent=2))


original_fit = module.fit_and_acquire_from_gp
original_bo = module.PriMO.sample_using_bo


def observe_fit(*a, **kw):
    result = original_fit(*a, **kw)
    records["gp_acquisitions"].append({"prior_nonnull": kw.get("prior") is not None})
    save()
    return result


def observe_bo(self, *a, **kw):
    result = original_bo(self, *a, **kw)
    records["bo_proposals"].append(
        {
            "execution_index": len(records["execution_attempts"]),
            "priors_nonnull": self.priors is not None,
        }
    )
    save()
    return result


def execute(path):
    trial = load_config(path)
    assert trial.judge.model.startswith("VLLM/"), "Hosted execution forbidden"
    record = {
        "model": trial.judge.model,
        "preset": trial.judge.prompt_preset,
        "order": trial.judge.swap_mode,
        "fidelity": trial.meta_eval.battles_per_model,
        "config_path": str(path),
        "status": "started",
    }
    records["execution_attempts"].append(record)
    save()
    try:
        run_trial_subprocess(path)
        record["status"] = "finished"
    except Exception:
        record["status"] = "failed"
        raise
    finally:
        save()


module.fit_and_acquire_from_gp = observe_fit
module.PriMO.sample_using_bo = observe_bo
try:
    run_tune_judge(cfg, None, execute_trial=execute)
finally:
    module.fit_and_acquire_from_gp = original_fit
    module.PriMO.sample_using_bo = original_bo
    save()
