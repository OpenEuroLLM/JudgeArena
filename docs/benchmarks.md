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
  --generation.n_instructions 10
```

A provider-prefixed baseline such as `VLLM/Qwen/Qwen2.5-0.5B-Instruct` can generate reference answers when the task supports a runtime baseline. Tasks with fixed multiple-reference or category-based baselines use their declared policy.

## Multilingual evaluation

Select languages through metadata filters:

```bash
uv run judgearena \
  --task m-arena-hard-v2.0 \
  --model.name OpenRouter/google/gemma-4-31b-it \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --run.filters '{"lang":["en","de"]}' \
  --generation.n_instructions 10
```

The instruction limit caps the selected run; it is not a per-language quota. Omit it for the full filtered set. Language-specific tasks such as `m-arena-hard-v2.0-de` are also available. Fluency uses named variants such as `fluency-french`; its prompts are sentence continuations, so check the generation format for the model you are evaluating.

## Model Elo estimation

`elo-lmarena-100k`, `elo-lmarena-140k`, `elo-lmarena` and `elo-comparia` select arena data. Opponents come from the arena, so these tasks do not use `model.baseline`.

```bash
uv run judgearena \
  --task elo-lmarena-140k \
  --model.name OpenRouter/google/gemma-4-31b-it \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --elo.languages '["en","de"]' \
  --elo.n_instructions_per_language 100 \
  --elo.n_bootstraps 200
```

The judge compares candidate answers with sampled arena opponents. Bradley-Terry scoring combines those judgments with the dataset's reference preferences. These reference labels are supplied by the arena; they are not new judgments from the selected judge.

`elo.soft_elo` selects continuous preferences and defaults to `true`. Set it to `false` for hard win/loss/tie labels. With a score-pair parser, `elo.soft_elo_temperature` controls score-to-preference conversion. Optional `elo.calibrate_temperature: true` fits that temperature against human-labeled battles; `elo.calibration_size` limits the calibration sample. Calibration adds judging requests.

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
