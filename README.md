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
| **JudgeArena** | 🔜  | ✅  | ✅  | ✅  | ✅                         | ✅                                          |

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
python judgearena/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A gpt4_1106_preview \
  --model_B VLLM/utter-project/EuroLLM-9B \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 10
```

**What happens here?**
- Use completions available for `gpt4_1106_preview` in Alpaca-Eval dataset
- Generates completions for `model_B` if not already cached on `vLLM`
- Compares two models using `deepseek-chat-v3.1` which the cheapest option available on `OpenRouter`

It will then display the results of the battles:

```bash
============================================================
                  🏆 MODEL BATTLE RESULTS 🏆
📊 Dataset: alpaca-eval
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
python judgearena/generate_and_evaluate.py \
  --dataset alpaca-eval \
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
python judgearena/generate_and_evaluate.py \
  --dataset alpaca-eval \
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
uv run python judgearena/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model_B OpenRouter/qwen/qwen-2.5-7b-instruct \
  --judge_model OpenRouter/deepseek/deepseek-chat-v3.1 \
  --n_instructions 10 --max_out_tokens_models 16384
```

**Fully local example** — no API keys required (useful for verifying your setup):

```bash
uv run python judgearena/generate_and_evaluate.py \
  --dataset alpaca-eval \
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
python judgearena/generate_and_evaluate.py \
  --dataset alpaca-eval \
  --model_A VLLM/swiss-ai/Apertus-8B-2509 \
  --model_B VLLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --judge_model VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
  --chat_template '{% for message in messages %}<|im_start|>{{ message["role"] }}\n{{ message["content"] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}'
```

This override applies to all vLLM models in the run. For remote providers (OpenAI, Together, OpenRouter), the flag is ignored since they handle templates server-side.

## 📊 Supported Datasets

| Dataset               | Description                                                                                    |
|-----------------------|------------------------------------------------------------------------------------------------|
| `alpaca-eval`         | General instruction-following benchmark                                                        |
| `arena-hard-v2.0` / `arena-hard`     | Arena-Hard v2.0 from official `lmarena-ai/arena-hard-auto` source                             |
| `arena-hard-v0.1`     | Legacy Arena-Hard v0.1 from official `lmarena-ai/arena-hard-auto` source                      |
| `m-arena-hard`        | Translated version of Arena-Hard in 23 languages                                               |
| `m-arena-hard-{lang}` | Language-specific variants (e.g., `ar`, `cs`, `de`)                                            |
| `m-arena-hard-EU`     | All EU languages combined                                                                      |
| `fluency-{lang}`      | Fluency evaluation for pretrained models (`finnish`, `french`, `german`, `spanish`, `swedish`) |

For Arena-Hard, JudgeArena resolves baseline metadata by dataset version:
- `arena-hard-v0.1`: `gpt-4-0314`
- `arena-hard-v2.0` / `arena-hard`: `o3-mini-2025-01-31` (standard prompts)

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
