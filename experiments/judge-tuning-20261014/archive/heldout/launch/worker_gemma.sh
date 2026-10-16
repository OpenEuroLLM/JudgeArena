set -euo pipefail
root=/work/dlclarge1/lushtake-hiwi/judge-tuning-heldout-3000-20261015
venv=/work/dlclarge1/lushtake-hiwi/JudgeArena-tune-judge-neps17/.venv
export PATH="$venv/bin:$PATH" PYTHONPATH="/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014/source"
export HF_HOME=/work/dlclarge1/lushtake-hiwi/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export TIKTOKEN_CACHE_DIR=/work/dlclarge1/lushtake-hiwi/.cache/tiktoken
export OPENROUTER_API_KEY="" OPENAI_API_KEY=""
export VLLM_USE_FLASHINFER_SAMPLER=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/ja-${SLURM_JOB_ID}"
printf 'job=%s host=%s stage=0 allocation_cap=32L40Sh\n' "$SLURM_JOB_ID" "$(hostname)"
cd /work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014/source
"$venv/bin/python" "$root/launch/evaluate_test.py" gemma
