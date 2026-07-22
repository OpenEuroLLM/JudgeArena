# JudgeArena container

This image installs JudgeArena on top of a vLLM base image, so `judgearena` runs inside the container against a local vLLM model and a local vLLM judge. It is also the image the [oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) integration runs.

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

Available base configs: `alpaca-eval`, `arena-hard-v2.0`, `mt-bench` (win-rate vs the task's native baseline), and `elo-lmarena-100k` (ELO rating against an arena). Each sets the task and a local vLLM judge (`google/gemma-4-12b-it`, sharing the GPU with the candidate). `--config_path` also accepts a path to your own YAML file. Override any field from the CLI — no need to edit the config:

```bash
singularity exec --bind <host paths> judgearena-rocm.sif \
  judgearena --config_path alpaca-eval --model.name VLLM/<your-model> \
    --judge.model VLLM/<other-judge> \
    --generation.n_instructions 10 \      # quick run: only 10 prompts
    --run.result_folder <out>
```

Results (annotations, win-rate report, the resolved `config.yaml`) land in `run.result_folder`.

## Datasets

`judgearena-download` fetches task datasets — everything, or just the tasks you name:

```bash
singularity exec judgearena-rocm.sif judgearena-download                       # all tasks
singularity exec judgearena-rocm.sif judgearena-download alpaca-eval mt-bench  # only these
```

They land in the data root, resolved in this order:

1. `JUDGEARENA_DATA` (or the legacy `OPENJURY_DATA`) if set;
2. `$HF_HOME/judgearena-data` if `HF_HOME` is set — convenient in containers, where `HF_HOME` is already a writable path, so no extra env var is needed;
3. `~/judgearena-data` otherwise.

**Offline clusters:** compute nodes usually have no internet. Prefetch models and datasets on a node that does (`judgearena-download`, `huggingface-cli download <model>`), then run with `HF_HUB_OFFLINE=1`.

## Running via oellm-eval

[oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) can schedule JudgeArena as an eval suite across clusters. JudgeArena ships the image; oellm-eval runs `judgearena` inside `EVAL_CONTAINER_IMAGE`.

Prerequisites:
- The image reachable at `$EVAL_BASE_DIR/<EVAL_CONTAINER_IMAGE>`.
- A judge config (`JUDGEARENA_CONFIG`) carrying the judge model and engine settings.
- Prefetched datasets on a bound path (`JUDGEARENA_DATA`, or under `HF_HOME`).

```bash
export EVAL_CONTAINER_IMAGE=judgearena-rocm.sif   # placed under $EVAL_BASE_DIR
export SINGULARITY_ARGS=""                         # AMD/ROCm image: no --rocm (glibc)
export JUDGEARENA_CONFIG=<judge.yaml>
export JUDGEARENA_DATA=<prefetched data dir>
export JUDGEARENA_EXTRA_BINDS=<extra host paths, if config/data live outside the default binds>
export HF_HOME=<model cache>; export HF_HUB_OFFLINE=1

oellm-eval schedule --models "<hf-model-id>" --task_groups judgearena-suite \
  --slurm_template_var '{"GPUS_PER_NODE":"2"}'
oellm-eval collect --results_dir "$EVAL_BASE_DIR" --output_csv results.csv
```

Notes:
- Pass `--models` **without** a `VLLM/` prefix — the suite prepends it.
- Task groups: `judgearena-alpaca` (single task) or `judgearena-suite` (alpaca-eval, arena-hard-v2.0, mt-bench).
- The judge config carries everything except `--task`, `--model.name`, and `--run.result_folder`, which oellm-eval injects.
- oellm-eval binds only `EVAL_BASE_DIR`, `HF_HOME`, and `HF_DATASETS_CACHE`, so the config and data must live under one of those or be added via `JUDGEARENA_EXTRA_BINDS`.
