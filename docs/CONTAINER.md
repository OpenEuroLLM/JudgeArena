# JudgeArena container

This image installs JudgeArena on top of a vLLM base image, so `judgearena` runs inside the container against a local vLLM model and a local vLLM judge. It is also the image the [oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) integration runs.

For LUMI-specific build/run notes (unprivileged `proot`, compute-node builds, the no-`--rocm` gotcha), see [`lumi-setup.md`](lumi-setup.md).

## Definition

`docker/judgearena-rocm.def` (AMD/ROCm). For NVIDIA, swap the base image to `vllm/vllm-openai:latest`.

```
Bootstrap: docker
From: vllm/vllm-openai-rocm:latest
%post
    pip install --no-cache-dir "judgearena @ git+https://github.com/OpenEuroLLM/JudgeArena@main"
%runscript
    exec judgearena "$@"
```

## Build

Building only needs userspace, not a GPU — a `pip install` has no compile step that touches the card.

```bash
singularity build judgearena-rocm.sif docker/judgearena-rocm.def
```

On clusters where users lack root, build unprivileged with `proot`; see `lumi-setup.md` for the LUMI recipe.

## Where to pull

The published image lives at [`kbora/judgearena-container`](https://huggingface.co/datasets/kbora/judgearena-container):

```bash
huggingface-cli download kbora/judgearena-container judgearena-rocm.sif --repo-type dataset --local-dir .
```

## Prefetch datasets

Compute nodes are usually offline, so the task datasets must be fetched ahead of time on a machine with internet. The `judgearena-download` entry point pulls task datasets into the data root — everything, or just the tasks you name:

```bash
singularity exec judgearena-rocm.sif judgearena-download                    # all tasks
singularity exec judgearena-rocm.sif judgearena-download alpaca-eval mt-bench   # only these
```

The data root is resolved in this order:

1. `JUDGEARENA_DATA` (or the legacy `OPENJURY_DATA`) if set;
2. `$HF_HOME/judgearena-data` if `HF_HOME` is set — convenient inside containers (e.g. under oellm-eval), where `HF_HOME` is already a bound, writable path and no extra env var is needed;
3. `~/judgearena-data` otherwise.

Point whichever variable you use at a path the compute job can also read.

## Run

```bash
singularity exec --bind <host paths> judgearena-rocm.sif \
  env HF_HUB_OFFLINE=1 JUDGEARENA_DATA=<data dir> \
  judgearena --config_path <judge.yaml> \
    --task alpaca-eval --model.name VLLM/<model> \
    --run.result_folder <out> --generation.n_instructions 5
```

The judge config carries the judge model and engine settings; `--task`, `--model.name`, and `--run.result_folder` select what to evaluate. See the repo README for task names and config options.
