# 🏛️ JudgeArena: LLM Evaluation with Swappable Judges

JudgeArena makes it easy to benchmark language models against each other while giving you complete control over the evaluation process.
Whether you're comparing proprietary models or testing your own fine-tuned creations, JudgeArena lets you choose your judge.

## ✨ Key Features

🎯 **Flexible Benchmarking** – Evaluate models on `Alpaca-Eval`, `Arena-Hard`, `m-Arena-Hard` and others

🔄 **Swappable Judges** – Switch between self-hosted (`vLLM`) or remote judges (`OpenAI`, `Together AI`, `OpenRouter`)

🌍 **Multilingual Support** – Test models across multiple languages with m-Arena-Hard

🛠️ **Provider Agnostic** – Works with any model available in [LangChain](https://python.langchain.com/docs/integrations/chat/)

Compared to other libraries, here is a breakdown of features:

| Framework | MT-Bench | AlpacaEval | Arena-Hard | M-Arena-Hard | Tuned judge configuration | Support vLLM Judges |
|-----------|----------|------------|------------|--------------|---------------------------|---------------------|
| **FastChat** | ✅  | ❌  | ❌  | ❌  | ❌                         | ❌                        |
| **AlpacaEval** | ❌  | ✅  | ❌  | ❌  | ❌                         | ❌                                             |
| **Arena-Hard-Auto** | ❌  | ❌  | ✅  | ❌  | ❌                         | ❌                                            |
| **Lighteval** | ✅  | ❌  | ❌  | ❌  | ❌                         | ❌                                       |
| **Evalchemy** | ✅  | ✅  | ❌  | ❌  | ❌                         | ❌                                           |
| **JudgeArena** | ✅  | ✅  | ✅  | ✅  | ✅                         | ✅                                          |

The table has been done on Oct 2025, in case some libraries implemented missing features, please open an issue
or send a PR, we will be happy to update the information.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/OpenEuroLLM/JudgeArena
cd JudgeArena
uv sync
uv sync --extra vllm      # Optional: install vLLM support
uv sync --extra llamacpp   # Optional: install LlamaCpp support
```

### Basic Evaluation

Compare two models head-to-head:

```bash
judgearena \
  --task alpaca-eval \
  --model.name gpt4_1106_preview \
  --model.baseline VLLM/utter-project/EuroLLM-9B \
  --judge.model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --generation.n_instructions 10
```

**What happens here?**
- Loads precomputed completions for `gpt4_1106_preview` from the Alpaca-Eval dataset
- Generates completions for `model_B` when they are not already present locally or in an optional inference cache (see [Unified inference cache](#unified-inference-cache))
- Compares the two models with `deepseek-chat-v3.1`, a low-cost option on OpenRouter

It will then display the results of the battles:

```bash
============================================================
                  🏆 MODEL BATTLE RESULTS 🏆
📊 Task: alpaca-eval
🤖 Competitors: Model A: gpt4_1106_preview vs Model B: VLLM/utter-project/EuroLLM-9B
⚖️ Judge: OpenRouter/deepseek/deepseek-chat-v3.1
📈 Results Summary:
   Total Battles: 10
   Win Rate (A): 30.0%
   ✅ Wins:   3
   ❌ Losses: 6
   🤝 Ties:   1
============================================================
```

### Run from a YAML config (`--config_path`)

Every option can also come from a YAML file. The CLI flags mirror the config
structure (`--model.name`, `--judge.model`, `--generation.n_instructions`, …),
and **precedence is: CLI flags > `--config_path` YAML > built-in defaults**, so
you can keep a base file and override single fields on the command line:

```bash
judgearena --config_path configs/alpaca_eval.yaml
# override one field on top of the file:
judgearena --config_path configs/alpaca_eval.yaml --judge.model VLLM/Qwen/Qwen2.5-32B-Instruct
```

Example [`configs/alpaca_eval.yaml`](configs/alpaca_eval.yaml):

```yaml
task: alpaca-eval
model:
  name: gpt4_1106_preview
  baseline: VLLM/utter-project/EuroLLM-9B
judge:
  model: OpenRouter/deepseek/deepseek-chat-v3.1
generation:
  n_instructions: 10
```

Only the fields you want to change need to be present — everything else uses the
model defaults. Each run also writes the fully-resolved config to
`<result_folder>/<run>/config.yaml`, which you can feed straight back via
`--config_path` to reproduce the run. An ELO example is in
[`configs/elo_comparia.yaml`](configs/elo_comparia.yaml).

### Unified inference cache

Caching is **disabled by default**. Set `cache.store_root` (CLI: `--store_root` or
`--cache.store_root`) to enable the shared SQLite store. Generation, judging,
meta-eval, MT-Bench, and ELO all route through the same per-inference boundary.

Each **configuration cell** is scoped by benchmark task, model spec, and a
descriptor hash derived from resolved engine/sampling settings. Within a cell,
rows are keyed by a SHA-256 hash of the **canonical rendered input** (for judges,
the fully rendered prompt including instruction, truncated completions, and
template). Changing any of those fields is a cache miss rather than a stale hit.

**Store layout** (local paths mirror the default Hugging Face dataset
`judge-arena/judge-arena-cache`):

```text
{store_root}/inference/{task}/{provider}/{percent-encoded-model}/{descriptor_hash}/
  metadata.json    # descriptor for this cell
  inference.db     # WAL-mode SQLite (`inference` + `inference_metadata` tables)
```

**Modes** (`--cache_mode` / `--cache.cache_mode`):

- `use` (default): read hits, infer misses, insert new rows (`INSERT OR IGNORE`).
- `off`: always infer, even when `store_root` is set.
- `refresh`: replace existing rows for the same input hash (`INSERT OR REPLACE`).

**Run flags** — nested and flat aliases are equivalent (`--cache.store_root` ≡
`--store_root`, `--cache.cache_fetch` ≡ `--cache_fetch`, and so on). Hugging Face
sync is **opt-in**: `cache_fetch`, `cache_push`, and `cache_create_pr` default to
`false`. When enabled, fetch pulls remote cells lazily as each local cell opens;
push uploads only cells written during the run and **only after a successful exit**
(failures and interrupts leave local SQLite updates but skip HF push). Use
`--no-cache_fetch` on compute nodes that cannot reach the Hub.
Main-pipeline sync is best-effort and logs failures without discarding local
results; the standalone `judgearena-cache fetch` and `push` commands fail loudly.

**Stochastic sampling.** Identical canonical inputs in one batch are deduplicated to
a single inference call. Across runs, the first stored output wins locally; Hub
merges keep the newest row by `pushed_at`. For stochastic judges (for example
meta-eval PairScore at temperature `0.5`), set an explicit `--judge.seed` when you
need reproducible resampling, or use `--cache_mode refresh` / a separate
`store_root`. `--run.seed` drives battle sampling and shuffles; it is not
automatically forwarded as a backend seed.

**Descriptors and provenance.**

- **Local engines (vLLM, LlamaCpp):** descriptors capture fully resolved engine
  settings and package versions (`vllm_version`, `llama_cpp_python_version`, …).
- **Hosted APIs (OpenRouter, OpenAI, Together):** only **explicitly set** client
  sampling fields are hashed. Provider-side defaults and model aliases are
  inherently best-effort — the client library version is recorded in row
  provenance (`langchain_openai_version`, `hosted_adapter_version`, …) but is **not**
  part of the descriptor hash.
- **Secrets and unsafe settings:** API keys, headers, and other sensitive
  constructor fields are redacted. Non-JSON-safe or un-normalizable
  `engine_kwargs` yield no descriptor; inference still runs, uncached, with a
  warning.

**Single-host SQLite.** Open cells use WAL mode. Do not share one open
`inference.db` across NFS or multiple compute nodes. Synchronize **closed** cells
via Hugging Face (`judgearena-cache fetch` / `push`) or a local store copy.

**Interrupted runs.** Partial rows are persisted locally as they are written. Re-run
to hit local cache, use `refresh` to regenerate, or `judgearena-cache push` after
you verify the store.

**ELO and meta-eval.** Cached judge completions are stored as raw outputs; ELO
preferences, calibration, and leaderboard statistics are always recomputed from
those outputs (and current run settings) rather than replaying prior aggregates.
Meta-eval uses task namespace `meta-eval-{reference_arena}` under `inference/`.

#### Examples

Local cache only (no network):

```bash
judgearena \
  --task alpaca-eval \
  --model.name gpt4_1106_preview \
  --model.baseline VLLM/utter-project/EuroLLM-9B \
  --judge.model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --generation.n_instructions 10 \
  --store_root ~/judgearena-data/cache
```

Run with explicit Hugging Face fetch and push:

```bash
judgearena \
  --config_path configs/alpaca_eval.yaml \
  --cache.store_root ~/judgearena-data/cache \
  --cache.cache_hf_repo judge-arena/judge-arena-cache \
  --cache.cache_fetch \
  --cache.cache_push \
  --cache.pushed_by "$USER"
```

Offline workflow (login node → compute → login node):

```bash
# login node: prefetch cells for the benchmark
judgearena-cache fetch \
  --store_root ~/judgearena-data/cache \
  --cache_hf_repo judge-arena/judge-arena-cache \
  --task alpaca-eval

# compute node: local store only
judgearena \
  --config_path configs/alpaca_eval.yaml \
  --store_root ~/judgearena-data/cache \
  --no-cache_fetch

# login node: merge and upload
judgearena-cache push \
  --store_root ~/judgearena-data/cache \
  --cache_hf_repo judge-arena/judge-arena-cache \
  --task alpaca-eval \
  --pushed_by "$USER"
```

If that path is on NFS, stage the **closed** store to node-local storage before
opening it on a compute node, then stage it back after the job. Never open the
same cell concurrently from multiple hosts.

Filtered sync (`fetch` requires `--prefix` or at least `--task`; `push` without
filters uploads every local cell):

```bash
judgearena-cache fetch \
  --store_root ~/judgearena-data/cache \
  --prefix inference/alpaca-eval/OpenRouter

judgearena-cache push \
  --store_root ~/judgearena-data/cache \
  --task alpaca-eval \
  --provider OpenRouter \
  --model deepseek/deepseek-chat-v3.1 \
  --create_pr \
  --pushed_by "$USER"
```

Open a pull request from the main CLI instead:

```bash
judgearena \
  --config_path configs/alpaca_eval.yaml \
  --store_root ~/judgearena-data/cache \
  --cache_fetch \
  --cache_push \
  --cache_create_pr
```

#### `judgearena-cache backfill`

`judgearena-cache backfill` imports **verified hosted judge rows** from saved
generate-and-evaluate, MT-Bench, and meta-eval run folders into the unified store.
For generate-and-evaluate and meta-eval, it reconstructs the rendered judge input
from saved configuration and checks it against stored `judge_input` text.
MT-Bench stores the exact system and rendered user messages used for each pass,
which are canonicalized directly.
**Insert-only, no network.**

Imported: hosted judge outputs (`OpenRouter`, `ChatOpenAI`, `OpenAI`, `Together`,
and `Dummy` in tests). Model-generation outputs are never backfilled because old
artifacts cannot prove whether they came from inference or precomputed datasets.

Skipped: generation-only artifacts, ELO runs, local engines (vLLM / LlamaCpp),
legacy pass-level caches, identity-keyed meta-eval DBs under `cache/db`, legacy
`judgements.db` / `completions.db` cells, rows whose recomputed input does not
match saved `judge_input`, and conflicting outputs for the same key.

Dry-run first, then write:

```bash
judgearena-cache backfill results/ \
  --store_root ~/judgearena-data/cache \
  --dry_run \
  --report ~/judgearena-data/cache/backfill-report.json

judgearena-cache backfill results/alpaca-eval-2026-04-01/ \
  --store_root ~/judgearena-data/cache \
  --report ~/judgearena-data/cache/backfill-report.json
```

The JSON report lists `written`, `existing`, `rows_planned`, `runs_processed`,
and per-reason `skipped` counts.

Optional YAML (commented so defaults stay network-free):

```yaml
# cache:
#   store_root: ~/judgearena-data/cache
#   cache_mode: use
#   cache_hf_repo: judge-arena/judge-arena-cache
#   # cache_fetch: true
#   # cache_push: true
#   pushed_by: your-user
```

Standalone cache CLI: `judgearena-cache {fetch,push,backfill}` (see
`judgearena-cache --help`).

### Length and Token Parameters

The evaluation scripts expose four different length controls with different roles:
- `--generation.truncate_all_input_chars`: character-level truncation applied to prompts before model generation and before judge evaluation.
- `--model.max_out_tokens`: generation token budget for each answer from `model_A` and `model_B`.
- `--judge.max_out_tokens`: generation token budget for the judge completion (reasoning + score output).
- `--model.max_model_len`: optional vLLM context-window limit (prompt + generated tokens), applied to vLLM models; this should be greater than or equal to the two `max_out_tokens_*` values.

### Engine-Specific Configuration (`--model.engine_kwargs`)

Some providers expose additional engine-level knobs (for example, vLLM allows configuring tensor parallelism or GPU memory utilization).
JudgeArena lets you forward these options directly to the underlying engine via `--model.engine_kwargs`, which expects a JSON object.

For instance, to run vLLM with tensor parallelism across multiple GPUs:

```bash
judgearena \
  --task alpaca-eval \
  --model.name VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model.baseline VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge.model VLLM/Qwen/Qwen3.5-27B-FP8 \
  --generation.n_instructions 10 \
  --model.engine_kwargs '{"tensor_parallel_size": 2}'
```

While any key in `--model.engine_kwargs` is forwarded to the underlying engine (e.g. `vllm.LLM`, `LlamaCpp`, `ChatOpenAI`), existing dedicated flags such as `--model.max_model_len` and `--model.chat_template` have higher precedence.

## 🎨 Model Specification

Models are specified using the format: `{LangChain Backend}/{Model Path}`

**Examples:**

```bash
Together/meta-llama/Llama-3.3-70B-Instruct-Turbo
ChatOpenAI/gpt-4o
LlamaCpp/jwiggerthale_Llama-3.2-3B-Q8_0-GGUF_llama-3.2-3b-q8_0.gguf
VLLM/utter-project/EuroLLM-9B
OpenRouter/deepseek/deepseek-chat-v3.1
```

For instance, to run everything locally with vLLM:

```bash
judgearena \
  --task alpaca-eval \
  --model.name VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model.baseline VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge.model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --generation.n_instructions 10
```

### Running locally with LlamaCpp

LlamaCpp allows you to run GGUF models locally with high efficiency across various hardware, including CPUs, Apple Silicon (Metal), and NVIDIA GPUs. This is ideal for testing your setup without relying on external API keys or high-end server GPUs.

**Install the LlamaCpp extra:**

```bash
uv sync --extra llamacpp
```

**Download GGUF models** using `hf` (included via `huggingface-hub`):

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir ./models
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf --local-dir ./models
```

The `LlamaCpp` provider expects a **file path** to a `.gguf` model after the `LlamaCpp/` prefix.
For absolute paths, this results in a double slash (e.g., `LlamaCpp//home/user/models/model.gguf`).

**Chat format:** LlamaCpp falls back to a generic template if the GGUF metadata does not embed a chat template. Pass the correct format for your model via `--model.engine_kwargs`:

```bash
# Llama-3.x models
--model.engine_kwargs '{"chat_format": "llama-3"}'
# Gemma models
--model.engine_kwargs '{"chat_format": "gemma"}'
# ChatML models (SmolLM, Qwen, Mistral, etc.)
--model.engine_kwargs '{"chat_format": "chatml"}'
```

Use separate `--judge.engine_kwargs` if the judge uses a different architecture than the evaluated models.

**Mixed example** — local LlamaCpp model with a remote judge:

```bash
uv run judgearena \
  --task alpaca-eval \
  --model.name LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model.baseline OpenRouter/qwen/qwen-2.5-7b-instruct \
  --judge.model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --generation.n_instructions 10 --model.max_out_tokens 16384 \
  --model.engine_kwargs '{"chat_format": "chatml"}'
```

**Fully local example** — no API keys required (useful for verifying your setup):

```bash
uv run judgearena \
  --task alpaca-eval \
  --model.name LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model.baseline LlamaCpp/./models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --judge.model LlamaCpp/./models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --generation.n_instructions 5 --model.max_out_tokens 16384 \
  --model.engine_kwargs '{"chat_format": "chatml"}'
```

**Note:** Ensure you have the required LangChain dependencies installed for your chosen provider.
If you use remote endpoint, you would have to set your credentials.

### Chat Templates (vLLM)

When using vLLM, JudgeArena automatically picks the right inference method based on the model:

- **Instruct/chat models** (e.g. `swiss-ai/Apertus-8B-Instruct-2509`): the tokenizer already defines a chat template, so JudgeArena uses `vllm.LLM.chat()` and the template is applied automatically.
- **Base/pretrained models** (e.g. `swiss-ai/Apertus-8B-2509`): these typically don't ship a chat template. JudgeArena detects this and falls back to `vllm.LLM.generate()` (plain text, no chat formatting). A warning is printed when this happens.

If you need to force a specific chat template (for example, a base model that you know works with ChatML), pass it via `--model.chat_template`:

```bash
judgearena \
  --task alpaca-eval \
  --model.name VLLM/swiss-ai/Apertus-8B-2509 \
  --model.baseline VLLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --judge.model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --model.chat_template '{% for message in messages %}<|im_start|>{{ message["role"] }}\n{{ message["content"] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}'
```

This override applies to all vLLM models in the run. For remote providers (OpenAI, Together, OpenRouter), the flag is ignored since they handle templates server-side.

## 📊 Supported Tasks

Task names follow [LMHarness](https://github.com/EleutherAI/lm-evaluation-harness) conventions. Generate+judge tasks produce pairwise preferences between two models; ELO tasks (`elo-*`) estimate a single model's ELO rating against human-annotated arena opponents.

### Generate + judge (pairwise)

| Task                         | Description                                                                                    |
|------------------------------|------------------------------------------------------------------------------------------------|
| `alpaca-eval`                | General instruction-following benchmark                                                        |
| `arena-hard-v2.0`            | Arena-Hard v2.0 from official `lmarena-ai/arena-hard-auto` source                              |
| `arena-hard-v0.1`            | Legacy Arena-Hard v0.1 from official `lmarena-ai/arena-hard-auto` source                       |
| `m-arena-hard-v0.1`          | `CohereLabs/m-ArenaHard` (500 prompts, Google-Translate) across 23 languages                   |
| `m-arena-hard-v0.1-{lang}`   | Language-specific v0.1 slice (e.g., `ar`, `cs`, `de`, `uk`, `zh`, `pl`)                        |
| `m-arena-hard-v0.1-EU`       | All EU v0.1 languages combined                                                                 |
| `m-arena-hard-v2.0`          | `CohereLabs/m-ArenaHard-v2.0` (498 prompts, in-house translation) across 23 languages          |
| `m-arena-hard-v2.0-{lang}`   | Language-specific v2.0 slice                                                                   |
| `m-arena-hard-v2.0-EU`       | All EU v2.0 languages combined                                                                 |
| `mt-bench`                   | Multi-turn benchmark with FastChat-compatible pairwise judging                                 |
| `fluency-{lang}`             | Fluency evaluation for pretrained models (`finnish`, `french`, `german`, `spanish`, `swedish`) |
| `meta-eval`                  | Judge meta-evaluation against human-labeled arena battles (not generator ELO)                  |

For MT-Bench, the default pairwise baseline is `gpt-4`.
We diverge from FastChat's own `pairwise-baseline` default (`gpt-3.5-turbo`) to keep
a stronger reference consistent with Arena-Hard v0.1; the `gpt-4.jsonl` completions
ship in the `lmsys/mt-bench` HF Space. Override per run with `--model.baseline`.

For Arena-Hard, JudgeArena resolves baseline metadata by task version:
- `arena-hard-v0.1`: `gpt-4-0314`
- `arena-hard-v2.0`: per-question baseline routed by `category`:
  - `o3-mini-2025-01-31` for `hard_prompt`, `coding`, and `math` (500 prompts).
  - `gemini-2.0-flash-001` for `creative_writing` (250 prompts).

For m-Arena-Hard, baseline completions are tied to the benchmark release:
- `m-arena-hard-v0.1`: Aya Expanse 8B (`CohereLabs/aya-expanse-8b`), ingested
  from `CohereLabs/deja-vu-pairwise-evals` (repeat 0) via
  [`scripts/multilingual_arena_hard/ingest_deja_vu_aya_references.py`](scripts/multilingual_arena_hard/ingest_deja_vu_aya_references.py).
- `m-arena-hard-v2.0`: Gemini 2.5 Flash (`google/gemini-2.5-flash`).

### ELO rating

| Task                | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `elo-lmarena-100k`  | Battles sampled from `lmarena-ai/arena-human-preference-100k`      |
| `elo-lmarena-140k`  | Battles sampled from `lmarena-ai/arena-human-preference-140k`      |
| `elo-lmarena`       | Union of all `LMArena-*` variants                                  |
| `elo-comparia`      | Battles sampled from the ComparIA arena                            |

## Judge meta-evaluation (`meta-eval`)

`meta-eval` measures how well an LLM judge agrees with **human-labeled arena battles**.
It does **not** generate model completions and does **not** estimate a generator model's ELO rating.
Use `elo-*` tasks for generator ELO estimation instead.

The task samples battles from a reference arena (default `LMArena-140k`), asks the judge to label fixed human battles, and reports accuracy, Cohen's kappa, Bradley-Terry ranking agreement, Spearman correlation, ELO MAE, bootstrap uncertainty, and ELO-gap analyses.

```bash
judgearena \
  --task meta-eval \
  --judge.model OpenRouter/deepseek/deepseek-v3.2 \
  --meta_eval.reference_arena LMArena-140k \
  --meta_eval.prompt_mode standard \
  --meta_eval.languages '["en", "es"]' \
  --meta_eval.top_models 20 \
  --meta_eval.battles_per_model 50 \
  --meta_eval.n_bootstraps 1000 \
  --run.seed 0
```

### Prompt modes

Only named prompt modes are supported (no custom prompt files):

| `--meta_eval.prompt_mode` | Description                              |
|---------------------------|------------------------------------------|
| `standard`                | Default PairScore judge prompt           |
| `arena-hard`              | Arena-Hard Likert verdict parsing        |
| `alpaca-eval`             | Alpaca-Eval JSON ordering prompt         |
| `alpaca-eval-pair-score`  | Alpaca-Eval prompt with PairScore output |

PairScore meta-eval uses temperature `0.5` (paper setting). The standard generate+judge benchmark path keeps PairScore temperature `0.3`.

Language filters use ISO 639-1 codes such as `en es fr`.

With `--judge.swap_mode both`, overall accuracy and kappa use both judge orderings
(the reversed verdict is inverted), while language, ranking, and ELO-gap analyses
retain one forward-order row per sampled battle to match the reference
meta-evaluation methodology. `annotations.parquet` stores one row per judge pass
with an `orientation` column; swapped winners and preferences are normalized
back to the original model ordering. Cost totals include both passes.

Results are written under `[run.result_folder]/meta-eval-*` as `args.json`,
`annotations.parquet`, `results.json`, `summary.csv`, logs, and
`run-metadata.v1.json`. With `--store_root`, judge calls use the unified cache
under `{store_root}/inference/meta-eval-{reference_arena}/…` (content-addressed
by rendered judge input; see [Unified inference cache](#unified-inference-cache)).
Annotation artifacts include character-based token estimates. Equivalent
OpenRouter cost is reported only when
`[data_root]/cache/openrouter_pricing.json` already contains the judge model;
meta-eval never fetches pricing from compute nodes.

## 📈 Estimating ELO Ratings

JudgeArena can estimate the ELO rating of a model by running it against opponents sampled from a human preference arena (`LMArena-100k`, `LMArena-140k`, or `ComparIA`).
The LLM judge scores each battle, and the resulting ratings are computed using the Bradley-Terry model anchored against the human-annotated arena leaderboard.

Pass an `elo-<arena>` value to `--task` to trigger the ELO flow. ELO tasks take a single `--model.name` whose opponents are sampled from the arena (matching the pairwise CLI shape; `--model.baseline` is reserved for a future extension).

### Quick start

```bash
judgearena \
  --task elo-comparia \
  --model.name Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --judge.model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --generation.n_instructions 200
```

With `--store_root`, generation and judge calls reuse cached rows; ELO win rates,
calibration, and Bradley-Terry scores are always recomputed from the cached judge
outputs and the current run configuration (see [Unified inference cache](#unified-inference-cache)).

### Key options

| Flag | Default | Description |
|---|---|---|
| `--task elo-<arena>` | *(required)* | Arena to sample opponents from: `elo-lmarena-100k`, `elo-lmarena-140k`, `elo-lmarena`, or `elo-comparia` |
| `--model.name` | *(required)* | Model under evaluation (same format as pairwise tasks) |
| `--judge.model` | *(required)* | LLM judge (same format as pairwise tasks) |
| `--generation.n_instructions` | all | Number of arena battles to use for evaluation |
| `--elo.n_instructions_per_language` | all | Cap battles per language (useful for balanced multilingual eval) |
| `--elo.languages` | all | Restrict to specific language codes (JSON list), e.g. `'["en", "fr", "de"]'` |
| `--elo.n_bootstraps` | `20` | Bootstrap samples for ELO confidence intervals |
| `--elo.elo_random_battles` | off | Sample N arena rows uniformly at random (seeded by `--run.seed`) instead of the first N |
| `--judge.swap_mode` | `fixed` | `fixed`: single judge pass; `both`: correct for position bias |
| `--run.result_folder` | `results` | Directory where annotations and results are saved |
| `--elo.soft_elo` | `true` | Soft Bradley-Terry (continuous preferences); set `false` for hard win/loss/tie labels |
| `--elo.soft_elo_temperature` | `0.3` | Initial softmax temperature for soft-ELO; overridden if calibration succeeds |
| `--elo.calibrate_temperature` | off | MLE-calibrate the score-to-preference temperature against human arena annotations |
| `--elo.calibration_size` | all | Number of human battles to sample for calibration (requires `--elo.calibrate_temperature`) |

### Soft-ELO & temperature calibration

By default, judge scores are converted into continuous preferences via a softmax (temperature `0.3`) and fed into a soft
Bradley-Terry model. Set `--elo.soft_elo false` to fall back to hard win/loss/tie labels.

To let the data choose the best temperature automatically, add `--elo.calibrate_temperature`.
JudgeArena will run the judge on a sample of human-annotated arena battles, fit the temperature $T^*$ by MLE, and
use it for the full evaluation:

```bash
judgearena \
  --task elo-lmarena-100k \
  --model.name Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --judge.model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --generation.n_instructions 200 \
  --elo.calibrate_temperature \
  --elo.calibration_size 300
```

### Output

The script prints win/loss/tie counts, win rate, and a ranked ELO leaderboard with confidence intervals:

```
=== Results for meta-llama/Llama-3.3-70B-Instruct-Turbo ===
Battles: 200 | Wins: 112 | Losses: 71 | Ties: 17
Win rate: 60.25%

=== ELO Ratings (Bradley-Terry, 20 bootstraps) ===
  gpt-4o  (12453): 1132.4 ± 3.1
  meta-llama/Llama-3.3-70B-Instruct-Turbo  (200) <-----: 1089.7 ± 8.2
  ...
```

### Offline Setup (Slurm/Air-Gapped Environments)

Pre-download datasets and models on a login node before submitting jobs:

```bash
python -c "from judgearena.utils import download_all; download_all()"  # optional: all datasets
hf download <org/model> --local-dir ~/models/my-model                 # model weights
```

Prefetch shared inference cells when you use `--store_root`:

```bash
judgearena-cache fetch \
  --store_root ~/judgearena-data/cache \
  --task alpaca-eval
```

On compute nodes, pass `--store_root` with `--no-cache_fetch`, then push from the
login node after the job finishes (see [Unified inference cache](#unified-inference-cache)).

Datasets are stored in:
- `$JUDGEARENA_DATA` if set; otherwise `$OPENJURY_DATA` if set (legacy)
- `~/judgearena-data/` if neither variable is set

## 🛠️ Development

To maintain code quality, we use **pre-commit** hooks. Run this once to set them up:

```bash
uv run pre-commit install
```

Once installed, hooks will automatically check and format your code on every `git commit`. If a commit is blocked, simply `git add` the changes made by the hooks and commit again.

## 🤝 Contributing

We welcome contributions! Whether it's bug fixes, new features, or additional benchmark support, feel free to open an issue or submit a pull request.

## Citation

If you use this work in your research, please cite the following paper.

```bibtex
@inproceedings{
  salinas2025tuning,
  title={Tuning {LLM} Judge Design Decisions for 1/1000 of the Cost},
  author={David Salinas and Omar Swelam and Frank Hutter},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=cve4NOiyVp}
}
```

The judge configurations was tuned in this paper and a lot of code is reused in this package.

---
