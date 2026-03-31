from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify


CLUSTER = "kislurm"
REMOTE_PROJECT_ROOT = Path("/work/dlclarge1/lushtake-hiwi/JudgeArena")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BINARY = REMOTE_PROJECT_ROOT / ".venv" / "bin" / "python"
ENTRYPOINT = "generate_and_evaluate.py"
SRC_DIR = str(LOCAL_PROJECT_ROOT / "judgearena")

# Use L40S partitions from the all_dlc / ml_dlc families.
# For this cluster/account, mldlc2 and testdlc2 are available.
PARTITION_ALL_DLC_L40S = "testdlc2_gpu-l40s"
PARTITION_ML_DLC_L40S = "mldlc2_gpu-l40s"


def submit_smoke_jobs(
    partition: str = PARTITION_ALL_DLC_L40S,
) -> list[tuple[str, str, int]]:
    slurm = SlurmPilot(clusters=[CLUSTER])

    datasets = [
        "arena-hard-v0.1",
        "arena-hard-v2.0",
    ]
    submitted: list[tuple[str, str, int]] = []

    for dataset in datasets:
        jobname = unify(f"arena-hard-smoke/{dataset}", method="date")
        job_info = JobCreationInfo(
            cluster=CLUSTER,
            partition=partition,
            jobname=jobname,
            entrypoint=ENTRYPOINT,
            python_binary=str(PYTHON_BINARY),
            python_args={
                "dataset": dataset,
                "model_A": "Dummy/no_answer",
                "model_B": "Dummy/open_answer",
                "judge_model": "Dummy/scoreA:0scoreB:10",
                "n_instructions": 1,
            },
            src_dir=SRC_DIR,
            n_cpus=1,
            max_runtime_minutes=15,
            env={
                # Data is pre-staged on login node; force offline execution on compute nodes.
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HOME": "/work/dlclarge1/lushtake-hiwi/.cache/huggingface",
                "OPENJURY_DATA": "/work/dlclarge1/lushtake-hiwi/judgearena-data",
                "JUDGEARENA_DATA": "/work/dlclarge1/lushtake-hiwi/judgearena-data",
            },
        )
        job_id = slurm.schedule_job(job_info)
        submitted.append((dataset, job_info.jobname, job_id))
        print(f"Submitted {dataset}: jobname={job_info.jobname}, job_id={job_id}")

    return submitted


if __name__ == "__main__":
    # Switch to PARTITION_ML_DLC_L40S if needed.
    print(f"Using LOCAL_PROJECT_ROOT={LOCAL_PROJECT_ROOT}")
    print(f"Using REMOTE_PROJECT_ROOT={REMOTE_PROJECT_ROOT}")
    print(f"Using PYTHON_BINARY={PYTHON_BINARY}")
    submit_smoke_jobs(partition=PARTITION_ALL_DLC_L40S)
