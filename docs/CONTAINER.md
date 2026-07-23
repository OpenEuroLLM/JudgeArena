# JudgeArena container

This image installs JudgeArena on top of a vLLM base image, so `judgearena` runs inside the container against a local vLLM model and a local vLLM judge.

## Definition

Save the following as `judgearena-rocm.def` (AMD/ROCm). For NVIDIA, swap the base image to `vllm/vllm-openai:latest`.

```
Bootstrap: docker
From: vllm/vllm-openai-rocm:latest
%post
    pip install --no-cache-dir "judgearena @ git+https://github.com/OpenEuroLLM/JudgeArena@main"
%runscript
    exec judgearena "$@"
```

## Build

Building only needs userspace, not a GPU — a `pip install` has no compile step that touches the card. With root:

```bash
singularity build judgearena-rocm.sif judgearena-rocm.def
```

### On a cluster without root

Most HPC users can't `sudo`, so build unprivileged with `proot`. Two things that bite on ROCm clusters:

- Use a **static** `proot` — a distro/module `proot` often links a newer glibc than the vLLM image ships, and the build fails with a `GLIBC_x.yz not found` error.
- The image is ~10 GB, so point `SINGULARITY_TMPDIR` at real disk (not RAM-backed `/tmp`), and run the build on a **compute node** if the login node is memory-limited.

```bash
wget -O ~/bin/proot https://proot.gitlab.io/proot/bin/proot   # static build
chmod +x ~/bin/proot && export PATH="$HOME/bin:$PATH"
export SINGULARITY_TMPDIR=<scratch>/sing-tmp
mkdir -p "$SINGULARITY_TMPDIR"
singularity build judgearena-rocm.sif judgearena-rocm.def
```

## Pull

The published image lives at [`kbora/judgearena-container`](https://huggingface.co/datasets/kbora/judgearena-container):

```bash
huggingface-cli download kbora/judgearena-container judgearena-rocm.sif --repo-type dataset --local-dir .
```

## Run

Run the ROCm image **without `--rocm`** — that flag injects host ROCm libs built against a newer glibc than the container has, so `torch` import fails. The image is self-contained; `--gpus`/`--device` plus the default `/dev` mount provide GPU access.

Each task ships a ready-to-use base config, so you don't need to write one. Pass its name to `--config_path` and supply your model with `--model.name`:

```bash
singularity exec --bind <host paths> judgearena-rocm.sif \
  judgearena --config_path alpaca-eval --model.name VLLM/<your-model>
```

`VLLM/` selects JudgeArena's local vLLM backend for the model under test (other backends exist, e.g. `OpenRouter/<model>` for an API model). On an offline GPU node only the local backend works.

Available base configs: `alpaca-eval`, `arena-hard-v2.0`, `mt-bench` (win-rate vs the task's native baseline), and `elo-lmarena-100k` (ELO rating against an arena). Each sets the task and a local vLLM judge (`google/gemma-4-31b-it`, sharing the GPU with the candidate). `--config_path` also accepts a path to your own YAML file. Override any field from the CLI — no need to edit the config:

```bash
singularity exec --bind <host paths> judgearena-rocm.sif \
  judgearena --config_path alpaca-eval --model.name VLLM/<your-model> \
    --judge.model VLLM/<other-judge> \
    --generation.n_instructions 10 \      # quick run: only 10 prompts
    --run.result_folder <out>
```

Results (annotations, win-rate report, the resolved `config.yaml`) land in `run.result_folder`.

## Datasets

`judgearena-download` fetches task datasets — win-rate task data *and* the ELO arena battles — so a task runs once its data is present. Get everything, or just the tasks you name:

```bash
singularity exec judgearena-rocm.sif judgearena-download                             # all tasks
singularity exec judgearena-rocm.sif judgearena-download alpaca-eval elo-lmarena-100k  # only these
```

They land in the data root, resolved in this order:

1. `JUDGEARENA_DATA` (or the legacy `OPENJURY_DATA`) if set;
2. `$HF_HOME/judgearena-data` if `HF_HOME` is set — convenient in containers, where `HF_HOME` is already a writable path, so no extra env var is needed;
3. `~/judgearena-data` otherwise.

**Offline clusters:** compute nodes usually have no internet. Prefetch models and datasets on a node that does (`judgearena-download`, `huggingface-cli download <model>`), then run with `HF_HUB_OFFLINE=1`.
