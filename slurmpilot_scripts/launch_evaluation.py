from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify

cluster = "kislurm"
slurm = SlurmPilot(clusters=[cluster])
job_info = JobCreationInfo(
    cluster=cluster,
    partition="alldlc2_gpu-l40s",
    jobname=unify("oellmjudge/test-vllm", method="date"),
    entrypoint="evaluate.py",
    python_binary="/work/dlclarge1/salinasd-llm-judge/llm-judge-eval/.venv/bin/python",
    python_args={
        "dataset": "alpaca-eval",
        "method_A": "gpt4_1106_preview",
        # TODO change to different model to have a better example
        "method_B": "gpt4_1106_preview",
        "judge_model": "VLLM/meta-llama/Meta-Llama-3-8B-instruct",
        "n_instructions": 10,
        # "ignore_cache": None,
    },
    src_dir=str(Path(__file__).parent.parent / "judgearena/"),
    n_cpus=1,
    max_runtime_minutes=60,
    env={"HF_HUB_OFFLINE": "1"},
)

# Launch the job
job_id = slurm.schedule_job(job_info)

slurm.wait_completion(job_info.jobname)

print(f"Job {job_id} scheduled on {job_info.cluster}")
