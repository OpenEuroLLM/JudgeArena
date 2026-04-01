from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify

CLUSTER = "kislurm"
REMOTE_PROJECT_ROOT = Path("/work/dlclarge1/lushtake-hiwi/JudgeArena")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_BINARY = REMOTE_PROJECT_ROOT / ".venv" / "bin" / "python"
ENTRYPOINT = "generate_and_evaluate.py"
SRC_DIR = str(LOCAL_PROJECT_ROOT / "judgearena")

# Use L40S partitions from the all_dlc / ml_dlc families.
PARTITION_ALL_DLC_L40S = "testdlc2_gpu-l40s"
PARTITION_ML_DLC_L40S = "mldlc2_gpu-l40s"

# Same weights as `VLLM/Qwen/Qwen3.5-27B-FP8`; repo-id loading fails offline in vLLM
# without a resolved revision — point at the HF hub snapshot dir under `HF_HOME`.
QWEN35_27B_FP8_SNAPSHOT = (
    "/work/dlclarge1/lushtake-hiwi/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-27B-FP8/snapshots/"
    "2e1b21350ce589fcaafbb3c7d7eac526a7aed582"
)
JUDGE_MODEL = f"VLLM//{QWEN35_27B_FP8_SNAPSHOT.lstrip('/')}"


def submit_smoke_job(partition: str = PARTITION_ALL_DLC_L40S) -> tuple[str, str, int]:
    slurm = SlurmPilot(clusters=[CLUSTER])
    dataset = "alpaca-eval"
    jobname = unify("qwen3.5-smoke/judgearena-canonical", method="date")

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
            "judge_model": JUDGE_MODEL,
            "n_instructions": 1,
            "max_out_tokens_judge": 64,
        },
        src_dir=SRC_DIR,
        n_cpus=1,
        max_runtime_minutes=20,
        env={
            "HF_HUB_OFFLINE": "1",
            # Ensure Hugging Face uses the shared cache location that
            # already contains the Qwen3.5 FP8 checkpoint.
            "HF_HOME": "/work/dlclarge1/lushtake-hiwi",
            "JUDGEARENA_DATA": "/work/dlclarge1/lushtake-hiwi/judgearena-data",
        },
    )
    job_id = slurm.schedule_job(job_info)
    print(f"Submitted {dataset}: jobname={job_info.jobname}, job_id={job_id}")
    return dataset, job_info.jobname, job_id


if __name__ == "__main__":
    print(f"Using LOCAL_PROJECT_ROOT={LOCAL_PROJECT_ROOT}")
    print(f"Using REMOTE_PROJECT_ROOT={REMOTE_PROJECT_ROOT}")
    print(f"Using PYTHON_BINARY={PYTHON_BINARY}")
    submit_smoke_job(partition=PARTITION_ALL_DLC_L40S)
