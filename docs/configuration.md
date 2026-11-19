# Configuration

A task defines the benchmark defaults. A run config selects models and changes the settings for one experiment:

```yaml
# run.yaml
task: arena-hard-v0.1
model:
  name: OpenRouter/google/gemma-4-31b-it
  max_out_tokens: 4096
judge:
  model: OpenRouter/google/gemma-4-31b-it
generation:
  n_instructions: 10
run:
  seed: 0
  result_folder: results
```

```bash
uv run judgearena --config_path run.yaml
uv run judgearena --config_path run.yaml --generation.n_instructions 100
uv run judgearena --help
```

Settings resolve in this order: **CLI > run config > task > code defaults**. Inspect a task with `uv run judgearena tasks show arena-hard-v0.1 --resolved`. Each run saves its effective settings in `config.yaml`.

## Inheritance and replacement

Nested mappings merge and lists replace. With the same runner or adapter, omitted fields retain the task's values. Changing `protocol.runner`, `dataset.adapter`, generation `mode`, baseline `strategy` or a dataset source's `type` replaces that section. Supply the fields the new implementation needs.

An explicit `{}` clears a nested mapping; `null` clears a nullable field. For example, this replaces the metric list but keeps the task's generation and judging settings:

```yaml
task: arena-hard-v0.1
protocol:
  scoring:
    metrics:
      - metric: pairwise_win_rate
        breakdown_by: [category]
```

Metric `parameters` are constructor arguments. Each `breakdown_by` field produces a separate breakdown, not a combination of fields. The chosen parser must return the values your metrics need.

A dataset revision can be overridden without repeating its adapter or field mappings:

```yaml
dataset:
  sources:
    examples:
      revision: "<dataset-commit-sha>"
```

Use the source name declared by your task; not every adapter calls its source `examples`.

## Prompts

Select a registered preset or a [YAML prompt bundle](../judgearena/prompts/README.md):

```yaml
judge:
  prompt:
    file: prompts/judge.yaml
```

Paths resolve beside the YAML file that declares them. Explicit CLI paths resolve from the working directory. `judge.prompt` overrides the task's main prompt and category routing.

Selecting a new `preset` or `file` replaces the prompt selection. Within a custom prompt, changing `user` or `user_file` preserves its parser and judging mode; changing an inline template clears the corresponding file field, and vice versa. Replacing a preset with custom text requires a compatible parser and judging mode.

A configurable preset accepts its settings through `options`:

```yaml
judge:
  prompt:
    preset: criteria-score
    options:
      criteria_file: rubric.yaml
```

See the [task guide](../judgearena/tasks/README.md#prompts-and-metrics) for independent scoring, custom metrics and rubric requirements.

## Filtering and length limits

Filter available metadata fields, for example:

```bash
uv run judgearena --config_path run.yaml --run.filters '{"lang":["en","de"]}'
```

This requires a task with a `lang` column. Task-level and runtime filters intersect.

| Setting | Controls |
|---|---|
| `generation.n_instructions` | Number of selected examples; omitted means the full task, subject to filters |
| `generation.truncate_all_input_chars` | Character cap before answer generation |
| `generation.truncate_judge_input_chars` | Character cap on judge-side inputs |
| `model.max_out_tokens` | Output-token budget for generated answers |
| `judge.max_out_tokens` | Output-token budget for the judge |
| `model.max_model_len` / `judge.max_model_len` | vLLM context-window limits, covering input and output together |

Tasks may set their own truncation defaults. Use `null` to disable a nullable character cap. Backend setup and engine options are in [Models](models.md).
