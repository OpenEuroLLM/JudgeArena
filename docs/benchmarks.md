# Benchmarks

Inspect the installed definition before choosing a model and judge:

```bash
uv run judgearena tasks list
uv run judgearena tasks show arena-hard-v0.1 --resolved
```

| Task | Protocol and baseline |
|---|---|
| `alpaca-eval` | AlpacaEval 2.0 prompt, logprob-weighted preferences and length-controlled scoring; archived model answers |
| `arena-hard-v0.1` | Both answer orders and weighted verdict scoring against `gpt-4-0314` |
| `arena-hard-v2.0` | Category-specific prompts and baselines: `o3-mini-2025-01-31` for hard questions, `gemini-2.0-flash-001` for creative writing |
| `m-arena-hard-v0.1` | 500 questions per language across 23 languages; Aya Expanse 8B baseline |
| `m-arena-hard-v2.0` | 498 questions per language across 23 languages; Gemini 2.5 Flash baseline |
| `mt-bench` | Multi-turn pairwise judging; saved `gpt-3.5-turbo` baseline |
| `mt-bench-101` | Supplied conversation histories, independent scores and dialogue-level aggregation |
| `wildbench` | Checklist-guided WB-Reward against a selected reference |
| `wildbench-reward-mix` | Mean WB-Reward against three saved references |
| `wildbench-score` | Independent answer scores |
| `fluency` | Sentence continuations across 43 languages; select the comparison baseline |

Saved reference availability depends on the task's pinned data. A baseline name identifies the intended reference; it does not guarantee that every upstream language has saved answers. Inspect preparation failures before launching a full run.

## Choose a baseline

For tasks that require a reference model, set `model.baseline`. For example, compare against WildBench's saved Llama 2 answers:

```bash
uv run judgearena \
  --task wildbench \
  --model.name OpenRouter/google/gemma-4-31b-it \
  --model.baseline Llama-2-7b-chat-hf \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --run.sampling.limit 10
```

A provider-prefixed baseline such as `VLLM/Qwen/Qwen2.5-0.5B-Instruct` can generate reference answers when the task supports a runtime baseline. Tasks with multiple-reference or category-based baselines use their declared policy. To compare against multiple named models, set `protocol.baseline: {strategy: all_references, references: [model-a, model-b]}` using your reference model IDs. Sample the prompts once with `run.sampling`; `pairwise_win_rate` with `breakdown_by: [baseline]` reports overall and per-reference win rates.

## Multilingual evaluation

Select languages through metadata filters:

```bash
uv run judgearena \
  --task m-arena-hard-v2.0 \
  --model.name OpenRouter/google/gemma-4-31b-it \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --run.filters '{"lang":["en","de"]}' \
  --run.sampling.limit 10
```

The instruction limit caps the selected run; it is not a per-language quota. Omit it for the full filtered set. Language-specific tasks such as `m-arena-hard-v2.0-de` are also available. Fluency uses named variants such as `fluency-french`; its prompts are sentence continuations, so check the generation format for the model you are evaluating.

## Model Elo estimation

`elo-lmarena-100k`, `elo-lmarena-140k`, `elo-lmarena` and `elo-comparia` select arena data. Opponents come from the arena, so these tasks do not use `model.baseline`.

```bash
uv run judgearena \
  --task elo-lmarena-140k \
  --model.name OpenRouter/google/gemma-4-31b-it \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --run.filters '{"lang":["en","de"]}' \
  --run.sampling.group_by lang \
  --run.sampling.per_group 100 \
  --protocol.scoring.metrics '[{"metric":"pairwise_win_rate"},{"metric":"bradley_terry","parameters":{"fit_mode":"frozen","soft":true,"n_bootstraps":200}}]'
```

The judge compares candidate answers with sampled opponents. Frozen Elo fits reference ratings from the arena’s supplied preferences, then rates the candidate against those fixed opponents. The example requests both win rate and Elo because the metrics list replaces the task’s defaults.

Set the Bradley–Terry metric parameter `soft` to `false` for hard win/loss/tie labels. For score-pair judges, `protocol.judge.preference_temperature` controls how score differences become preferences. To fit the temperature against reference labels, set `protocol.judge.calibrate_temperature` to `true` and use `protocol.judge.calibration_size` to limit the extra judging requests.

Ratings depend on the reference pool and evaluation settings. Compare runs using the same data revision, judge, prompt and sampling settings.

## Judge meta-evaluation

Meta-evaluation judges existing answer pairs and compares the resulting preferences with human labels. It does not generate a new candidate's answers.

```bash
uv run judgearena \
  --task meta-eval-lmarena-140k \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --meta_eval.languages '["en","de"]' \
  --meta_eval.top_models 20 \
  --meta_eval.battles_per_model 50
```

Other packaged tasks are `meta-eval-lmarena-100k` and `meta-eval-comparia`. Their metrics report preference agreement and ranking comparisons. See [task definitions](../judgearena/tasks/README.md) for custom metrics and the reference-label contract.

To customize an existing benchmark, use [config overrides](configuration.md). For independent scoring and custom annotations, reuse the [single-answer workflow](../judgearena/tasks/README.md#independent-evaluation).
