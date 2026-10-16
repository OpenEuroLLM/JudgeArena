# Intermediate judge tuning and 3,000-battle held-out evaluation

Experiment run 14–16 October 2026. This branch starts from the exact execution
base, `ab7cc3f5c22527ba29ed2d337c92e2f689bc25b4`, and adds the eight experiment
prompt templates and their registry entries. The underlying parsers, metric,
cache and optimizer implementations are unchanged from that base.

## Start here

- [Intermediate results](archive/intermediate/analysis/report.md)
- [Held-out results](archive/heldout/analysis/report.md)
- [Frozen intermediate configs](archive/intermediate/configs/)
- [Frozen test candidates](archive/heldout/configs/manifest.json)
- [Original scripts](archive/): exact executed/preparation/recovery scripts;
  historical absolute paths are preserved for provenance. Do not run archived
  submission scripts blindly: they reference the original account and job roots.
- [Portable entrypoint](reproduce.py): creates a separate working directory,
  relocates operational paths and refreshes relocated candidate config hashes.
  It leaves the archived originals unchanged.
- [Execution environment](runtime-packages.json): package versions captured from
  the read-only cluster runtime. Key versions: Python 3.12, NePS 0.17.0,
  vLLM 0.25.1, transformers 5.10.1, torch 2.11.0, openai 2.54.0.

The original source snapshot differs from this branch's base only in the prompt
registry and eight `mid-*` template files. `default` and
`default_with_explanation` retain their original bytes. `off/on` in preset names
means written explanation absent/present, not evaluation guidance disabled.
Thinking remains enabled in both cases. `mid` identifies the intermediate study.
The verdict presets also change output format; they are not a pure wording ablation.

## Design and interpretation

Three searches, one optimizer seed each: native uniform MO-Hyperband, native
PriMO without priors, and native PriMO with priors. Search space: 3 judge models
× 10 enumerated prompts (5 base strategies × explanation off/on) × 2 order modes
= 60 configurations. Fidelity keys 20/60/180 refer to explicit panels of
400/1,200/3,548 battles, not literal per-model sample sizes. Panel seed 0,
judge seed 42, temperature 0, thinking enabled, total output 8,192 tokens,
context 65,536, completion truncation 8,192 characters, TP4.

The common operational horizon is 28,540 attempted configuration-battles and
includes the original timed-out uniform panel. The native arms finish their
current schedule beyond that horizon. The report also retains logical-budget
sensitivity. Uniform recovery used the official dead-owner reset; observer
wrappers delegate unchanged to native NePS. Do not reset live trials.

Five candidates were frozen from full-fidelity validation results before test
inference. All use the identical 3,000 test battle IDs / 2,981 prompt clusters.
The exclusion audit records five old reserved rows removed due to normalized
prompt overlap before outcomes were observed. No candidate or prompt was changed
using test results. One optimizer seed does not establish optimizer robustness.

Agreement is exact A/tie/B accuracy with unparseable outputs counted as failures.
The tie interval is [0.49,0.51]; the paper uses exactly 0.5, which changes none of
our 15,000 test classifications. Cohen's kappa excludes unparsed judgments and
must be read with coverage. Unlike the paper's tunable implementation, parsing
failures are not converted to ties. Cost is native input/output tokens at frozen
reference prices, normalized to 1,000 battles, not GPU rental charges. Paired
intervals use 10,000 seed-0 prompt-cluster bootstrap resamples. All primary
paired intervals include zero. No confirmatory prior-benefit claim is made.

Intermediate GPU allocation: 108.8189 L40S-hours, including failed work/retry.
Held-out stage: 17.711111 L40S-hours, including the failed Qwen startup. API spend
was zero. The original manifest's inherited 16-hour field is stale; approved
intermediate cap was 120 hours, separately from the 32-hour test-stage cap.
Historical launch-file hash exceptions remain documented in the archived audit.

## Data and cache publication

Native content-addressed cache folders are in the existing dataset repository:
<https://huggingface.co/datasets/judge-arena/judge-arena-cache>.
They retain the standard `judgements/meta-eval-lmarena-140k-en/...` layout.
The publication is scoped to canonical input hashes and models referenced by
this study's qualification, search and held-out annotations. Unrelated historical
rows and descriptors are excluded. Native descriptors/input hashes govern reuse.

Publication uses only the repository's `judgearena-cache` CLI and the standard
cache layout. No replay bundle, optimizer-state archive or per-battle result
archive is published to Hugging Face.

Published revision: `b36a31a9de6679fab7e15fa82bc1285f349dd949`. The three databases contain
92,869 unique judgments (Qwen-9B: 25,408; Qwen-27B: 15,879; Gemma-31B: 51,582).
Remote database SHA256 digests and sizes match the scoped local snapshots, and
all three descriptor metadata files match exactly. The native fetch CLI reads
current main; `PUBLICATION.json` records this verified snapshot for provenance. The Git branch supplies scripts, frozen
panels/configs and aggregate reports. `PUBLICATION.json` records upload status
and the published revision when available.

## Environment setup

Use a separate Python 3.12 environment. For exact Linux/CUDA package versions,
consult `runtime-packages.json`; `runtime-requirements.txt` captures installed
versions, excluding the editable JudgeArena project. Slurmpilot is only needed
for the historical submission scripts, not the portable entrypoint. CUDA driver,
GPU topology and model files are external prerequisites, not bundled wheels.

```bash
uv venv --python 3.12 .venv-experiment
uv pip install --python .venv-experiment/bin/python -r experiments/judge-tuning-20261014/runtime-requirements.txt
uv pip install --python .venv-experiment/bin/python --no-deps -e .
```

On a laptop, install the ordinary analysis dependencies instead of the GPU
runtime (`uv sync --extra tune`); that is an analysis environment, not the original
Linux inference environment. Data loaders use the dataset revisions pinned in
this branch's task definitions. Download those dataset snapshots and the three
model checkpoints before enabling offline mode on compute nodes. Cache descriptors
record resolved model settings; these runs did not separately pin every model
weight repository to a recorded Hub commit, so a future fresh download of mutable
model weights is not guaranteed to be byte-identical.

## Fetch the published inference cache

Use the existing cache CLI:

```bash
judgearena-cache --action fetch --store_root /path/to/new-run/cache \
  --hf_repo judge-arena/judge-arena-cache \
  --kind judgements --task meta-eval-lmarena-140k-en
```

Run `prepare` below before fetching. Then run the frozen configurations using
that cache. Exact matches reuse stored outputs and token usage; missing inputs
still require inference and a suitable allocation. Cache replay does not
reproduce historical runtime or reconstruct optimizer state. Aggregate reports
are already in `archive/*/analysis/`; no separate result archive is required.

The relocated working manifest hashes operationally relocated configs; the Git
archive retains original execution hashes. Do not compare relocated bytes against
original absolute-path hashes and call that corruption.

## Fresh inference rerun (requires a GPU allocation)

Use a new empty workdir. `prepare` copies frozen panels and
configs; it does not resample or inspect test labels. Run workers within an
allocation; the helper does not request GPUs, change accounts or enforce a global
budget. Each inference worker needs 4 L40S GPUs. Use at most two concurrent workers
(8 GPUs); account/user limits still apply. Do not run all three searches together.

```bash
PY=.venv-experiment/bin/python
REPRO=experiments/judge-tuning-20261014/reproduce.py
$PY "$REPRO" prepare --workdir /path/to/new-run
# In a 4-GPU compute allocation, one arm per worker:
$PY "$REPRO" search --workdir /path/to/new-run --arm uniform
$PY "$REPRO" search --workdir /path/to/new-run --arm primo-none
$PY "$REPRO" search --workdir /path/to/new-run --arm primo-prior
# Frozen test candidates; each group in its own 4-GPU allocation:
$PY "$REPRO" test --workdir /path/to/new-run --group gemma
$PY "$REPRO" test --workdir /path/to/new-run --group qwen
```

A warm-cache rerun measures cache replay, not original inference runtime. To
measure cold runtime, use an empty cache and budget all new computation. Queue
noise and stochastic GPU kernels mean a fresh search need not reproduce the
same trajectory bit for bit. Original intermediate qualification/recovery and
Slurmpilot workers remain archived for inspection, not automatically resubmitted.
Do not rerun `archive/intermediate/launch/prepare.py` on this checkout: the prompt
registry has already been generated, and the frozen panels supersede regeneration.

## Publication review

Before publication scope was narrowed to native cache CLI only, the portable
`prepare` and private local-bundle replay completed with API keys
blank and HF offline. Recomputed test agreement, coverage, kappa, cost, individual
intervals and all paired intervals match the original results exactly. All 50
intermediate trials were recomputed and search accounting matched. This validated
stored-artifact replay in the existing local analysis environment, not a fresh
Linux/CUDA install or fresh GPU run. See `REPLAY-REVIEW.json`. No new tests or
inference were run for publication.

Once all searches and test workers have produced their result records, regenerate
the analyses on a CPU compute node with:

```bash
$PY "$REPRO" analyze --workdir /path/to/new-run
```
