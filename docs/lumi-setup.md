# JudgeArena on LUMI (AMD MI250X / ROCm)

LUMI-specific notes for the baked JudgeArena image. For the image itself (definition, generic build, pull, `judgearena-download`), see [`CONTAINER.md`](CONTAINER.md).

LUMI facts that shape everything below:
- Login nodes have internet; compute nodes are offline — prefetch models and datasets on a login node, then run with `HF_HUB_OFFLINE=1`.
- `/flash`, `/projappl`, `/scratch` are symlinks into `/pfs/lustre*`, so `--bind /pfs` is mandatory (binding only the friendly name leaves the real target unbound and the path is empty in the container).
- `/projappl` has a maxed inode cap — keep the repo there, but put the `.sif`, HF cache, and outputs on `/flash` or `/scratch`. `/scratch` auto-purges, so move keepers to `/flash`.
- GPU partitions: `dev-g` (fast, 3 h max), `small-g` (backfills quickly), `standard-g` (production).

`<proj>` below is your project (e.g. Slurm `--account`); substitute your own paths.

## Build on LUMI

LUMI users have no root, so `singularity build` uses unprivileged `proot`. Two LUMI gotchas: LUMI's own `proot` links a newer glibc than the vLLM image ships (use a *static* `proot`), and login nodes OOM-kill the build (run it on a compute node). The image is ~10 GB.

```bash
module load CrayEnv systools                                   # provides proot
wget -O ~/bin/proot https://proot.gitlab.io/proot/bin/proot    # STATIC proot (avoids the glibc mismatch)
chmod +x ~/bin/proot && export PATH="$HOME/bin:$PATH"
export SINGULARITY_TMPDIR=/scratch/<proj>/users/$USER/sing-tmp   # NOT /tmp (RAM-backed)
mkdir -p "$SINGULARITY_TMPDIR"
env -u SINGULARITY_BIND singularity build judgearena-rocm.sif docker/judgearena-rocm.def
```

## Prefetch models and datasets (login node)

```bash
export SIF=/scratch/<proj>/containers/judgearena-rocm.sif
export HF_HOME=/flash/<proj>/users/$USER/hf-cache; mkdir -p "$HF_HOME"
# each model in your config (candidate, judge, any baseline):
singularity exec "$SIF" python -c "from huggingface_hub import snapshot_download; snapshot_download('<org/model>')"
# task datasets (see CONTAINER.md for the data-root precedence):
HF_HUB_OFFLINE=0 singularity exec --bind /pfs,/projappl,/scratch,/flash "$SIF" judgearena-download
```

If a checkpoint is stored as subfolders (e.g. `iter_0124800/` with its own `config.json`), point `--model` at that subfolder.

## Run

Run **without `--rocm`** — that flag injects host ROCm libs built against a newer glibc than the container has, so `torch` import fails. The image is self-contained; `srun --gpus=N` plus the default `/dev` mount provide GPU access.

```bash
export SIF=/scratch/<proj>/containers/judgearena-rocm.sif
export HF_HOME=/flash/<proj>/users/$USER/hf-cache
srun --account=<proj> --partition=dev-g --gpus=2 --time=00:30:00 \
  singularity exec --bind /pfs,/projappl,/scratch,/flash "$SIF" \
    env HF_HUB_OFFLINE=1 JUDGEARENA_DATA=<data dir> \
    judgearena --config_path <judge.yaml> \
      --task alpaca-eval --model.name VLLM/<model> \
      --run.result_folder <out> --generation.n_instructions 5
```

## Running via oellm-eval

[oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) can schedule JudgeArena as an eval suite. JudgeArena ships the image; oellm-eval runs `judgearena` inside `EVAL_CONTAINER_IMAGE`. Only the paths and the no-`--rocm` note are LUMI-specific.

Prerequisites:
- The image reachable at `$EVAL_BASE_DIR/<EVAL_CONTAINER_IMAGE>`.
- A judge config (`JUDGEARENA_CONFIG`) carrying the judge model and engine settings.
- Prefetched datasets on a bound path (`JUDGEARENA_DATA`, or under `HF_HOME`).

```bash
export EVAL_CONTAINER_IMAGE=judgearena-rocm.sif   # placed under $EVAL_BASE_DIR
export SINGULARITY_ARGS=""                         # LUMI: no --rocm (glibc)
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
- Task groups: `judgearena-alpaca` (single task) or `judgearena-suite` (alpaca-eval, arena-hard-v2.0, mt-bench). Each task uses its native baseline; the metric is a win-rate.
- The judge config carries everything except `--task`, `--model.name`, and `--run.result_folder`, which oellm-eval injects.
- oellm-eval binds only `EVAL_BASE_DIR`, `HF_HOME`, and `HF_DATASETS_CACHE`, so the config and data must live under one of those or be added via `JUDGEARENA_EXTRA_BINDS`.
- If a scheduled job reports the wrong image (e.g. the cluster's default `eval_env.sif`), a login profile is overriding `EVAL_CONTAINER_IMAGE`/`SINGULARITY_ARGS` inside the job — set them with `${VAR:-default}` in the profile so submit-time values survive.
