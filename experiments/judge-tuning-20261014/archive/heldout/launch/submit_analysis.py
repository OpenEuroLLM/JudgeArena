"""Submit an approved bounded phase with neutral names."""

import argparse, json, secrets
from pathlib import Path
from slurmpilot import JobCreationInfo, SlurmPilot

p = argparse.ArgumentParser()
p.add_argument("phase", choices=["check", "qualify"])
p.add_argument("--dependency")
a = p.parse_args()
root = Path("/private/tmp/judgearena-tuning-heldout-3000-20261015")
gpus = 0 if a.phase == "check" else 4
args = "--exclude=dlc2gpu10,dlc2gpu12,lmbgpu09,lmbgpu10"
if gpus:
    args += " --mem-per-gpu=24G"
if a.dependency:
    args += f" --dependency=afterok:{a.dependency}"
j = JobCreationInfo(
    jobname="r" + secrets.token_hex(3),
    entrypoint="worker_analyze.sh",
    cluster="kislurm",
    src_dir=str(root / "launch"),
    remote_dir="/work/dlclarge1/lushtake-hiwi/slurmpilot",
    partition="mldlc2_gpu-l40s",
    nodes=1,
    n_gpus=gpus,
    n_cpus=4 if not gpus else 16,
    mem=12288 if not gpus else None,
    max_runtime_minutes=30,
    sbatch_arguments=args,
)
result = SlurmPilot().schedule_job(j)
(root / "launch" / "analysis-submission.json").write_text(
    json.dumps(
        {
            "job_id": str(result),
            "jobname": j.jobname,
            "gpus": gpus,
            "max_minutes": 30,
            "cap": 32,
        },
        indent=2,
    )
)
print("JOB", result, "NAME", j.jobname, flush=True)
