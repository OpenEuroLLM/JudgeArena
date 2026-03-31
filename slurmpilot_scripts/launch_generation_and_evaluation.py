from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify

cluster = "kislurm"
slurm = SlurmPilot(clusters=[cluster])


qwen_models = [
    "VLLM/Qwen/Qwen2.5-0.5B",
    "VLLM/Qwen/Qwen2.5-1.5B",
    "VLLM/Qwen/Qwen2.5-3B",
    "VLLM/Qwen/Qwen2.5-7B",
    "VLLM/Qwen/Qwen2.5-14B",
]

for language in [
    "italian",
    "spanish",
    "german",
    "french",
    "swedish",
    "finnish",
]:
    if language == "spanish":
        baseline = "VLLM/HPLT/hplt2c_spa_checkpoints"
        multisynt_models = [
            "VLLM/MultiSynt/nemotron-cc-spanish-tower9b",
            "VLLM/MultiSynt/nemotron-cc-spanish-run1-tower72b",
            "VLLM/MultiSynt/nemotron-cc-translated-100B-opus-mt-spa",
        ]
    elif language == "german":
        baseline = "VLLM/HPLT/hplt2c_deu_checkpoints"
        multisynt_models = [
            "VLLM/MultiSynt/nemotron-cc-translated-100B-opus-mt-deu",
            "VLLM/MultiSynt/nemotron-cc-german-9b",
            "VLLM/MultiSynt/nemotron-cc-german-tower72b",
        ]
    elif language == "swedish":
        baseline = "VLLM/HPLT/hplt2c_swe_checkpoints"
        multisynt_models = [
            "VLLM/MultiSynt/nemotron-cc-swedish-opus",
            "VLLM/MultiSynt/nemotron-cc-swedish-tower9b",
            "VLLM/MultiSynt/nemotron-cc-swedish-tower72b",
        ]
    elif language == "finnish":
        baseline = "VLLM/HPLT/hplt2c_fin_checkpoints"
        multisynt_models = [
            "VLLM/MultiSynt/nemotron-cc-finnish-opus",
            "VLLM/MultiSynt/nemotron-cc-finnish-tower9b",
            "VLLM/MultiSynt/nemotron-cc-finnish-tower72b",
        ]
    elif language == "italian":
        baseline = "VLLM/HPLT/hplt2c_ita_checkpoints"
        multisynt_models = [
            "VLLM/MultiSynt/nemotron-cc-italian-opus",
            "VLLM/MultiSynt/nemotron-cc-italian-tower72b",
        ]
    elif language == "french":
        baseline = "VLLM/HPLT/hplt2c_fra_checkpoints"
        multisynt_models = []
    else:
        raise ValueError(language)

    job_info = JobCreationInfo(
        cluster=cluster,
        partition="alldlc2_gpu-h200",
        jobname=unify(f"oellmjudge-v3/{language}-eval", method="date"),
        entrypoint="generate_and_evaluate.py",
        python_binary="/work/dlclarge1/salinasd-llm-judge/check/llm-judge-eval/.venv/bin/python",
        python_args=[
            {
                "dataset": f"{language}-contexts",
                "model_A": baseline,
                "model_B": model,
                "judge_model": "VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
                "n_instructions": 100,
                # "ignore_cache": None,
            }
            for model in multisynt_models + qwen_models
        ],
        src_dir=str(Path(__file__).parent.parent / "judgearena/"),
        python_libraries=[str(Path(__file__).parent.parent / "data/")],
        n_cpus=1,
        max_runtime_minutes=30,
        env={
            "HF_HUB_OFFLINE": "1",
            # "LLM_JUDGE_EVAL_DATA": "/work/dlclarge1/salinasd-llm-judge/check/llm-judge-eval/here",
        },
    )

    # Launch the job
    job_id = slurm.schedule_job(job_info)
    print(f"Job {job_id} scheduled on {job_info.cluster}")
    # slurm.wait_completion(job_info.jobname)
