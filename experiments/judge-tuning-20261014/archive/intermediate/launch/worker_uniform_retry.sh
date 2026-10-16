set -euo pipefail
root=/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014
venv=/work/dlclarge1/lushtake-hiwi/JudgeArena-tune-judge-neps17/.venv
export PATH="$venv/bin:$PATH" PYTHONPATH="$root/source"
export HF_HOME=/work/dlclarge1/lushtake-hiwi/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export TIKTOKEN_CACHE_DIR=/work/dlclarge1/lushtake-hiwi/.cache/tiktoken
export OPENROUTER_API_KEY="" OPENAI_API_KEY=""
export VLLM_USE_FLASHINFER_SAMPLER=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ja-${SLURM_JOB_ID}"
printf 'job=%s host=%s stage=0 allocation_cap=120L40Sh\n' "$SLURM_JOB_ID" "$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$root/source"
"$venv/bin/python" "$root/launch/recover_uniform.py"
"$venv/bin/python" "$root/launch/search.py" uniform
