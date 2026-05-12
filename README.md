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
  --model_A gpt4_1106_preview \
  --model_B VLLM/utter-project/EuroLLM-9B \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 10
```

**What happens here?**
- Use completions available for `gpt4_1106_preview` in Alpaca-Eval task
- Generates completions for `model_B` if not already cached on `vLLM`
- Compares two models using `deepseek-chat-v3.1` which the cheapest option available on `OpenRouter`

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

### Length and Token Parameters

The evaluation scripts expose four different length controls with different roles:
- `--truncate_all_input_chars`: character-level truncation applied to prompts before model generation and before judge evaluation.
- `--max_out_tokens_models`: generation token budget for each answer from `model_A` and `model_B`.
- `--max_out_tokens_judge`: generation token budget for the judge completion (reasoning + score output).
- `--max_model_len`: optional vLLM context-window limit (prompt + generated tokens), applied to vLLM models; this should be greater than or equal to the two `max_out_tokens_*` values.

### Engine-Specific Configuration (`--engine_kwargs`)

Some providers expose additional engine-level knobs (for example, vLLM allows configuring tensor parallelism or GPU memory utilization).
JudgeArena lets you forward these options directly to the underlying engine via `--engine_kwargs`, which expects a JSON object.

For instance, to run vLLM with tensor parallelism across multiple GPUs:

```bash
judgearena \
  --task alpaca-eval \
  --model_A VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model_B VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model VLLM/Qwen/Qwen3.5-27B-FP8 \
  --n_instructions 10 \
  --engine_kwargs '{"tensor_parallel_size": 2}'
```

While any key in `--engine_kwargs` is forwarded to the underlying engine (e.g. `vllm.LLM`, `LlamaCpp`, `ChatOpenAI`), existing dedicated flags such as `--max_model_len` and `--chat_template` have higher precedence.

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
  --model_A VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model_B VLLM/Qwen/Qwen2.5-1.5B-Instruct \
  --judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --n_instructions 10
```

### Running locally with LlamaCpp

LlamaCpp allows you to run GGUF models locally with high efficiency across various hardware, including CPUs, Apple Silicon (Metal), and NVIDIA GPUs. This is ideal for testing your setup without relying on external API keys or high-end server GPUs.

**Install the LlamaCpp extra:**

```bash
uv sync --extra llamacpp
```

**Download GGUF models** using `huggingface-cli` (included via `huggingface-hub`):

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir ./models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf --local-dir ./models
```

The `LlamaCpp` provider expects a **file path** to a `.gguf` model after the `LlamaCpp/` prefix.
For absolute paths, this results in a double slash (e.g., `LlamaCpp//home/user/models/model.gguf`).

**Mixed example** — local LlamaCpp model with a remote judge:

```bash
uv run judgearena \
  --task alpaca-eval \
  --model_A LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model_B OpenRouter/qwen/qwen-2.5-7b-instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 10 --max_out_tokens_models 16384
```

**Fully local example** — no API keys required (useful for verifying your setup):

```bash
uv run judgearena \
  --task alpaca-eval \
  --model_A LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model_B LlamaCpp/./models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --judge_model LlamaCpp/./models/qwen2.5-1.5b-instruct-q8_0.gguf \
  --n_instructions 5 --max_out_tokens_models 16384
```

**Note:** Ensure you have the required LangChain dependencies installed for your chosen provider.
If you use remote endpoint, you would have to set your credentials.

### Chat Templates (vLLM)

When using vLLM, JudgeArena automatically picks the right inference method based on the model:

- **Instruct/chat models** (e.g. `swiss-ai/Apertus-8B-Instruct-2509`): the tokenizer already defines a chat template, so JudgeArena uses `vllm.LLM.chat()` and the template is applied automatically.
- **Base/pretrained models** (e.g. `swiss-ai/Apertus-8B-2509`): these typically don't ship a chat template. JudgeArena detects this and falls back to `vllm.LLM.generate()` (plain text, no chat formatting). A warning is printed when this happens.

If you need to force a specific chat template (for example, a base model that you know works with ChatML), pass it via `--chat_template`:

```bash
judgearena \
  --task alpaca-eval \
  --model_A VLLM/swiss-ai/Apertus-8B-2509 \
  --model_B VLLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --chat_template '{% for message in messages %}<|im_start|>{{ message["role"] }}\n{{ message["content"] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}'
```

This override applies to all vLLM models in the run. For remote providers (OpenAI, Together, OpenRouter), the flag is ignored since they handle templates server-side.

## 📊 Supported Tasks

Task names follow [LMHarness](https://github.com/EleutherAI/lm-evaluation-harness) conventions. Generate+judge tasks produce pairwise preferences between two models; ELO tasks (`elo-*`) estimate a single model's ELO rating against human-annotated arena opponents.

### Generate + judge (pairwise)

| Task                  | Description                                                                                    |
|-----------------------|------------------------------------------------------------------------------------------------|
| `alpaca-eval`         | General instruction-following benchmark                                                        |
| `arena-hard-v2.0`     | Arena-Hard v2.0 from official `lmarena-ai/arena-hard-auto` source                             |
| `arena-hard-v0.1`     | Legacy Arena-Hard v0.1 from official `lmarena-ai/arena-hard-auto` source                      |
| `m-arena-hard`        | Translated version of Arena-Hard in 23 languages                                               |
| `m-arena-hard-{lang}` | Language-specific variants (e.g., `ar`, `cs`, `de`)                                            |
| `m-arena-hard-EU`     | All EU languages combined                                                                      |
| `mt-bench`            | Multi-turn benchmark with FastChat-compatible pairwise judging                                 |
| `fluency-{lang}`      | Fluency evaluation for pretrained models (`finnish`, `french`, `german`, `spanish`, `swedish`) |

For Arena-Hard, JudgeArena resolves baseline metadata by task version:
- `arena-hard-v0.1`: `gpt-4-0314`
- `arena-hard-v2.0`: `o3-mini-2025-01-31` (standard prompts)

### ELO rating

| Task                | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `elo-lmarena-100k`  | Battles sampled from `lmarena-ai/arena-human-preference-100k`      |
| `elo-lmarena-140k`  | Battles sampled from `lmarena-ai/arena-human-preference-140k`      |
| `elo-lmarena`       | Union of all `LMArena-*` variants                                  |
| `elo-comparia`      | Battles sampled from the ComparIA arena                            |

## 📈 Estimating ELO Ratings

JudgeArena can estimate the ELO rating of a model by running it against opponents sampled from a human preference arena (`LMArena-100k`, `LMArena-140k`, or `ComparIA`).
The LLM judge scores each battle, and the resulting ratings are computed using the Bradley-Terry model anchored against the human-annotated arena leaderboard.

Pass an `elo-<arena>` value to `--task` to trigger the ELO flow. ELO tasks take a single `--model_A` whose opponents are sampled from the arena (matching the pairwise CLI shape; `--model_B` is reserved for a future extension).

### Quick start

```bash
judgearena \
  --task elo-comparia \
  --model_A Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 200
```

### Key options

| Flag | Default | Description |
|---|---|---|
| `--task elo-<arena>` | *(required)* | Arena to sample opponents from: `elo-lmarena-100k`, `elo-lmarena-140k`, `elo-lmarena`, or `elo-comparia` |
| `--model_A` | *(required)* | Model under evaluation (same format as pairwise tasks) |
| `--judge_model` | *(required)* | LLM judge (same format as pairwise tasks) |
| `--n_instructions` | all | Number of arena battles to use for evaluation |
| `--n_instructions_per_language` | all | Cap battles per language (useful for balanced multilingual eval) |
| `--languages` | all | Restrict to specific language codes, e.g. `en fr de` |
| `--n_bootstraps` | `20` | Bootstrap samples for ELO confidence intervals |
| `--swap_mode` | `fixed` | `fixed`: single judge pass; `both`: correct for position bias |
| `--result_folder` | `results` | Directory where annotations and results are saved |
| `--soft-elo` | off | Use continuous judge preferences (soft Bradley-Terry) instead of hard win/loss/tie labels |
| `--soft-elo-temperature` | `0.3` | Initial softmax temperature for `--soft-elo`; overridden if `--calibrate-temperature` succeeds |
| `--calibrate-temperature` | off | MLE-calibrate the score-to-preference temperature against human arena annotations (requires `--soft-elo`) |
| `--calibration-size` | all | Number of human battles to sample for calibration (requires `--calibrate-temperature`) |
| `--conformal-alpha` | off | Target miscoverage for the conformal Elo interval (e.g. `0.1` for 90% coverage). Requires `--calibrate-temperature`. |
| `--conformal-min-battles-per-anchor` | `20` | Minimum judge-scored calibration battles an anchor must appear in to contribute a residual. |

### Soft-ELO & temperature calibration

By default, judge scores are discretised to hard win/loss/tie labels. Passing `--soft-elo` instead converts the raw score
difference into a continuous preference via a softmax, which is then fed into a soft Bradley-Terry model.

To let the data choose the best temperature automatically, add `--calibrate-temperature`.
JudgeArena will run the judge on a sample of human-annotated arena battles, fit the temperature $T^*$ by MLE, and
use it for the full evaluation:

```bash
judgearena \
  --task elo-lmarena-100k \
  --model_A Together/meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 200 \
  --soft-elo \
  --calibrate-temperature \
  --calibration-size 300
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

### Conformal Elo intervals

When `--conformal-alpha` is set together with `--calibrate-temperature`, JudgeArena reuses
the judge-scored human battles produced for temperature calibration as a leave-one-anchor-out
calibration set. For each anchor model with at least `--conformal-min-battles-per-anchor`
judge-scored battles and a human Elo reference, it refits Bradley–Terry on the human pool
excluding that anchor plus the anchor's judge-scored battles, and records the residual
$r_k = \mathrm{human\_elo}(m_k) - \mathrm{judge\_elo}(m_k)$. The conformal quantile
$\hat q_\alpha$ of $\{|r_k|\}$ then gives a distribution-free marginal-coverage interval
$\mathrm{judge\_elo}(\text{model\_A}) \pm \hat q_\alpha$ for the model under evaluation:

```
=== Conformal Interval (α=0.10, K=24 anchors) ===
  q̂ = 28.4 Elo
  meta-llama/Llama-3.3-70B-Instruct-Turbo: 1089.7 ∈ [1061.3, 1118.1]
```

The interval is only as reliable as the calibration set is large. As a rule of thumb,
non-trivial $\alpha = 0.1$ coverage needs at least $K \geq 19$ surviving anchors; smaller
$K$ saturates the quantile at the worst observed residual. Raise `--calibration-size` (and
correspondingly `--conformal-min-battles-per-anchor`) for tighter intervals.

### Offline Setup (Slurm/Air-Gapped Environments)

Pre-download all datasets before running jobs:

```bash
python -c "from judgearena.utils import download_all; download_all()"  # Download all datasets (optional)
```

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
