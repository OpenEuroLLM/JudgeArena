# Models and backends

Use `Backend/model` for a model that should generate answers or judgments. A saved-output identifier such as `gpt4_1106_preview` selects answers already supplied by a dataset instead.

| Backend | Example | Setup |
|---|---|---|
| OpenRouter | `OpenRouter/google/gemma-4-31b-it` | `OPENROUTER_API_KEY` |
| OpenAI chat | `ChatOpenAI/gpt-4o` | `OPENAI_API_KEY` |
| Together | `Together/meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| vLLM | `VLLM/Qwen/Qwen2.5-0.5B-Instruct` | `uv sync --extra vllm` |
| LlamaCpp | `LlamaCpp/./models/model.gguf` | `uv sync --extra llamacpp` |

Choose a model available to your provider account. Candidate and judge backends can differ. Tasks such as AlpacaEval require judge token logprobs, so check that the selected backend and model support the task's protocol.

## Local vLLM

```bash
uv sync --extra vllm
uv run judgearena \
  --task arena-hard-v0.1 \
  --model.name VLLM/Qwen/Qwen2.5-0.5B-Instruct \
  --model.max_out_tokens 4096 \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --generation.n_instructions 10
```

For chat models, vLLM applies the tokenizer's chat template. Without a template, the wrapper falls back to plain-text generation. Use the model's intended template when evaluating instruction-tuned checkpoints.

`model.chat_template` overrides the candidate template. The baseline inherits it unless `model.baseline_chat_template` is set. The judge also uses it as a fallback unless `judge.chat_template` is set. Hosted providers handle their own chat formatting.

Forward engine options with a mapping:

```yaml
model:
  engine_kwargs:
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9
judge:
  engine_kwargs:
    tensor_parallel_size: 1
```

The CLI equivalent is `--model.engine_kwargs '{"tensor_parallel_size":2}'`. These are backend-specific options; vLLM engine settings do not apply to hosted APIs. Dedicated fields such as `model.max_model_len` take precedence over the same engine option. Use `model.temperature`, `model.top_p` and the corresponding judge fields for supported sampling settings.

## Local GGUF models

```bash
uv sync --extra llamacpp
uv run hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  qwen2.5-0.5b-instruct-q8_0.gguf --local-dir ./models

uv run judgearena \
  --task arena-hard-v0.1 \
  --model.name LlamaCpp/./models/qwen2.5-0.5b-instruct-q8_0.gguf \
  --model.engine_kwargs '{"chat_format":"chatml"}' \
  --model.max_out_tokens 1024 \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --generation.n_instructions 10
```

The part after `LlamaCpp/` is a local file path. Absolute paths have two slashes, for example `LlamaCpp//home/user/models/model.gguf`. Set `chat_format` to match the checkpoint when its GGUF metadata does not supply the intended template. A different judge can use its own `judge.engine_kwargs`.

For compute nodes without internet access, see [offline preparation](../judgearena/tasks/README.md#prepare-an-offline-run). Dataset preparation does not download model weights.
