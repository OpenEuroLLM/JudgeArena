# Running JudgeArena on LUMI

The low-friction setup for LUMI (CSC, MI250X / ROCm): keep vLLM in a fixed Singularity image, and run JudgeArena from an editable venv beside it. vLLM is the heavy, stable part; JudgeArena is light and changes often, so baking it into the image on every change is wasteful. With this layout, updating JudgeArena is just `git pull` — no image rebuild.

Build the image and the venv on a **login node** (it has internet); execute on a **compute node** (offline). The `.sif` and the venv are portable across both.

## Paths

The vLLM `.sif` lives in the shared containers directory:

```
/scratch/project_465002530/containers/vllm-openai-rocm.sif
```

Set the two paths that are yours — your JudgeArena checkout and where your venv should live:

```bash
export SIF=/scratch/project_465002530/containers/vllm-openai-rocm.sif
export REPO=/path/to/your/JudgeArena          # <-- set this: your git checkout
export VENV=/path/to/your/venvs/judgearena    # <-- set this: on /scratch or /flash, NOT /projappl
export SINGULARITY_BIND=/pfs,/scratch,/projappl,/flash
```

Put `$VENV` on `/scratch` or `/flash` — `/projappl` has a small inode cap and a venv is thousands of small files. `/pfs` in the bind list is mandatory; the friendly names (`/scratch`, `/flash`, `/projappl`) are symlinks into it.

## Build the vLLM image (if it doesn't exist, or you want a different version)

If `$SIF` is missing, or you want a different vLLM/ROCm version, build it on the **login node** by pulling the official image. Point tmp/cache at `/scratch` — the image is ~20–30 GB.

```bash
export SINGULARITY_TMPDIR=/scratch/project_465002530/users/$USER/sing-tmp
export SINGULARITY_CACHEDIR=/scratch/project_465002530/users/$USER/sing-cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
singularity build "$SIF" docker://vllm/vllm-openai-rocm:latest
```

- Use the official `vllm/vllm-openai-rocm` image, not the deprecated `rocm/vllm` (frozen at an old vLLM).
- For a different version, change the tag — e.g. `:nightly`, or a pinned `vllm/vllm-openai-rocm:<version>`.


## One-time setup (login node)

Build the venv against the container's own Python so it sees the container's vLLM/torch, then install JudgeArena editable:

```bash
singularity exec "$SIF" python -m venv --system-site-packages "$VENV"
singularity exec "$SIF" "$VENV/bin/pip" install -e "$REPO"
```

- Use the container's `python`, not `uv venv` or host Python — only the container's interpreter sees the container's vLLM under `--system-site-packages`.
- Do not install the `[vllm]` extra; the image already provides vLLM/torch/transformers.

## Updating JudgeArena

Pull the latest code; the editable install reflects it immediately, no rebuild:

```bash
git -C "$REPO" checkout main && git -C "$REPO" pull
```

Re-run `pip install -e "$REPO"` only if dependencies in `pyproject.toml` changed.

## Pre-download models (login node)

Compute nodes have no internet, so cache anything you need first, then run offline:

```bash
HF_HUB_OFFLINE=0 singularity exec "$SIF" "$VENV/bin/python" -c \
  "from huggingface_hub import snapshot_download; print(snapshot_download('<org/model>'))"
```

If a checkpoint is stored as subfolders (e.g. `iter_0124800/` with its own `config.json`), point `--model` at that subfolder.

## Running (compute node)

Set the offline/runtime flags per run rather than baking them in:

```bash
srun --account=project_465002530 --partition=dev-g --gpus=1 --time=01:00:00 \
  singularity exec \
    --env HF_HUB_OFFLINE=1 \
    "$SIF" "$VENV/bin/judgearena" --config_path "$REPO/configs/<run>.yaml"
```

`HF_HUB_OFFLINE=1` is required because compute nodes are offline. `VLLM_ENABLE_V1_MULTIPROCESSING=0` avoids a ZMQ engine-IPC crash on this stack.

## Gotchas

- **Run WITHOUT `singularity --rocm`.** That flag injects host ROCm libs built against a newer glibc than the image has. The image is self-contained; the default `/dev` mount plus `srun --gpus=1` provide GPU access.
- **`/scratch` auto-purges.** Move anything you want to keep (the `.sif`, generated outputs) to `/flash` or off-cluster.
- **If a bound path is empty inside the container**, resolve it with `readlink -f` and bind the real `/pfs/...` target.
