"""Bounded two-lane graph: three approved searches, no held-out jobs."""

import json, secrets
from pathlib import Path
from slurmpilot import JobCreationInfo, SlurmPilot

root = Path("/private/tmp/judgearena-tuning-intermediate-20261014")
qual = json.loads((root / "launch/qualify-submission.json").read_text())["job_id"]
pilot = SlurmPilot()
jobs = []
for arm in ["uniform", "primo-prior", "primo-none"]:
    dependency = None if arm != "primo-none" else jobs[0]["job_id"]
    job = JobCreationInfo(
        jobname="r" + secrets.token_hex(3),
        entrypoint=f"worker_{arm}.sh",
        cluster="kislurm",
        src_dir=str(root / "launch"),
        remote_dir="/work/dlclarge1/lushtake-hiwi/slurmpilot",
        partition="mldlc2_gpu-l40s",
        nodes=1,
        n_gpus=4,
        n_cpus=16,
        mem=None,
        max_runtime_minutes=534,
        sbatch_arguments="--mem-per-gpu=24G --exclude=dlc2gpu10,dlc2gpu12,lmbgpu09,lmbgpu10" + (f" --dependency=afterok:{dependency}" if dependency else ""),
    )
    result = pilot.schedule_job(job)
    jobs.append(
        {
            "arm": arm,
            "job_id": str(result),
            "jobname": job.jobname,
            "dependency": dependency,
            "gpus": 4,
            "wall_minutes": 534,
        }
    )
    (root / "launch/search-submissions.json").write_text(json.dumps(jobs, indent=2))
    print(jobs[-1], flush=True)
