# Task definitions

Task YAML describes **what benchmark is being run**. A run config describes
**how one experiment runs it**.

- Keep stable benchmark behavior in the task: pinned data, field mappings,
  baseline policy, judge protocol, and scoring definition.
- Keep experiment choices in `RunConfig`: candidate and judge models, sampling
  parameters, instruction limits, caching, and supported overrides.
- Publish a new task version when changing benchmark defaults. Experiments can
  replace `protocol` or `dataset` in `RunConfig` without editing the task; these
  use the same schemas and replace the whole section before defaults resolve.

## Layout

```text
tasks/
├── definitions/       # Packaged task YAML, grouped by benchmark family
├── schema/            # Typed YAML contracts, separated by responsibility
├── registry.py        # Loading, discovery, task lookup, and validation
└── cli.py             # `judgearena tasks ...` commands
```

Executable behavior lives outside the definitions:

- `judgearena/datasets/` downloads and normalizes upstream data.
- `judgearena/benchmarks/` runs the selected evaluation workflow.
- `judgearena/prompts/` owns reusable judge prompt presets.

## Add a task using existing components

Create a public YAML file under `definitions/<family>/`. Files beginning with
`_` are private bases and are not runnable tasks.

```yaml
schema_version: 1
task: example-pairwise
task_version: 1
description: Pairwise evaluation on the Example instruction set.
tags: [pairwise, instruction-following]

dataset:
  adapter: judgearena_tables
  sources:
    examples:
      type: huggingface_dataset
      repo_id: organization/example
      revision: "0123456789abcdef0123456789abcdef01234567"
      allow_patterns:
        - instructions/example-pairwise.csv
        - model_outputs/example-pairwise.csv.zip
  fields:
    id: instruction_index
    instruction: instruction

protocol:
  runner: pairwise
  generation:
    mode: single_turn_chat
  baseline:
    strategy: task_default
    reference_id: example-baseline
  judge:
    default_prompt_preset: default
  scoring:
    metrics:
      - metric: pairwise_win_rate
      - metric: length_controlled_winrate
```

Reuse a private `_base.yaml` with `extends: _base.yaml` when several versions
share meaningful settings. Avoid inheritance merely to save a few lines.

Validate the result before running it:

```bash
judgearena tasks validate example-pairwise
judgearena tasks show example-pairwise --resolved
```

The resolved view includes inherited values and hashes of every YAML resource.

## Add a dataset adapter

An adapter loads data and returns `CompletionData` or `PreferenceData`.
For example, this adapter reads local Parquet files:

```python
import pandas as pd

from judgearena.datasets.data import CompletionData
from judgearena.datasets.registry import DatasetAdapter, register_dataset


@register_dataset("local_questions")
class LocalQuestions(DatasetAdapter):
    def load_data(self) -> CompletionData:
        questions = pd.read_parquet(self.task.spec.dataset.options["path"])
        return CompletionData.from_frames(
            questions.rename(columns={"id": "instruction_index", "prompt": "instruction"}),
        )
```

Replace the example task's `dataset` section with:

```yaml
dataset:
  adapter: local_questions
  options:
    path: questions.parquet
```

The adapter decides which `sources`, `fields` and `options` it needs.
Pin data revisions when publishing a reproducible benchmark.

Import the adapter before loading a task that uses it. Registration lasts for
that Python process. To run `my_tasks/example/example-pairwise.yaml`:

```python
from pathlib import Path

from judgearena.benchmarks.runner import run_benchmark
from judgearena.config import RunConfig
from judgearena.tasks import load_tasks

task = load_tasks(Path("my_tasks"))["example-pairwise"]
cfg = RunConfig(
    task=task.task,
    resolved_task=task,
    model={"name": "Dummy/candidate", "baseline": "Dummy/baseline"},
    judge={"model": "Dummy/score_A: 8 score_B: 6"},
)
run_benchmark(cfg)
```

Use Python for external adapters and task roots; the CLI loads installed tasks.

### Run without a task YAML

Supply validated data directly, or declare `dataset` with a registered adapter:

```python
import pandas as pd
from judgearena import CompletionData, RunConfig, run_benchmark

data = CompletionData(
    instructions=pd.DataFrame({"instruction": ["Describe the weather."]}),
    model_outputs=pd.DataFrame(
        [(0, "candidate", "It is raining.")],
        columns=["instruction_index", "model", "completion"],
    ),
)
cfg = RunConfig(
    evaluation_name="grammar-example",
    protocol={
        "runner": "single_answer",
        "generation": {"mode": "single_turn_chat"},
        "judge": {"default_prompt_preset": "single-score"},
        "scoring": {"metrics": [{"metric": "mean_score"}]},
    },
    model={"name": "candidate"},
    judge={"model": 'Dummy/{"score": 8}'},
    run={"store_root": None},
)
annotations = run_benchmark(cfg, data=data)
print(annotations[["model", "score"]])
```

The dummy response checks execution, not answer quality. `evaluation_name`
names taskless artifacts and caches. Saved configs contain the effective task
and resolved settings; reloading freezes those settings for replay. Supplied
DataFrames and custom Python implementations must be supplied again. Custom
user/system templates and criteria files are copied into each run directory;
its saved config uses relative paths to those copies. YAML judge asset paths
resolve beside the YAML file. Explicit CLI paths and direct Python paths resolve
from the working directory. For `configs/run.yaml` to use a project-root
`prompts/rubric.txt`, set `user_file: ../prompts/rubric.txt`. Effective task hashes are recomputed, with a warning if a saved
definition no longer matches its recorded hash.

### Dataset contracts

| Workflow | Adapter output |
|---|---|
| Text pairwise | `CompletionData`: indexed `instruction` rows; optional outputs with `instruction_index`, `model`, and `completion`. Missing answers are generated. |
| Multi-turn pairwise | `CompletionData` with `user_turns` (a list of user prompts) per instruction. Saved outputs also carry `messages`, a list of role/content dictionaries ending in an assistant answer. |
| Supplied-history generation | `CompletionData` with input `messages` per row; optional `instruction_id` sets cache identity. Each row produces one answer. |
| Anchored Elo | `PreferenceData`: canonical comparisons with observed `reference_pref` values supplying rating anchors. |
| Meta-evaluation | `PreferenceData` with observed `reference_pref` values. Current meta metrics require discrete references (`0`, `0.5`, `1`). Existing battle IDs are preserved; otherwise `PreferenceData` assigns row IDs. Adapters should supply IDs stable across reloads. |

Instruction IDs must be unique and nonmissing; output IDs must match their types.
`PreferenceData` requires `instruction_index`, `instruction`, `model_a`,
`completion_a`, `model_b`, and `completion_b`. Reference labels and provenance
are optional. Adapters map source labels to `reference_pref`: `0` favors A, `1`
favors B, and `0.5` is a tie. New judge results use `pref`. Reference labels belong
to the original answer pair and are not inherited when new comparisons are built.
See [data.py](../datasets/data.py) for the canonical columns and validation.
Meta-eval metrics receive judged outcomes plus unjudged reference rows (`sampled=false`).
Generic win rate ignores missing preferences, but its missing count includes
unjudged rows. Custom metrics can select `rows.loc[rows["sampled"]]` when needed.
Reference metrics combine answer orders per comparison; `minimum_score_by` is
unsupported here because it discards the original reference labels.

`from_battles()` derives outputs from the A/B columns, including `messages_a`
and `messages_b` when supplied. The built-in arena adapter exposes only the first
user/assistant turn as instruction and answer text. Source system prompts stay
in metadata and are not added to the judge input. It logs and excludes malformed
or mismatched source pairs. Multi-turn evaluation needs complete histories.

Output metadata is preserved in judging rows; pairwise rows suffix it with `_a`
or `_b`. Saved output `messages` contain the completed conversation. Independent
judging retains instruction-side `messages` as `input_messages`; other overlapping
metadata names must be renamed in the adapter. Exports JSON-encode nested and
mixed-type columns, including mixed integer/string IDs; use `json.loads` on
those columns to restore original values before joining.

Standard prompt fields `completion` (or `completion_A`/`completion_B`) render
complete conversations when available. Use `completion_text` (or
`completion_A_text`/`completion_B_text`) for the answer text alone, for example
when your template already includes the history. Both respect thinking removal
and truncation settings.

The MT-Bench-101 adapter handles a fixed source format and rejects
`dataset.fields`. For another source layout, normalize it in your own adapter.

Extra columns remain metadata. Use `--run.filters '{"category": ["coding"]}'`
to select any available metadata field. Task variants and runtime filters are
intersected; empty selections raise an error.

## Multi-turn pairwise judging

Use the shared pairwise runner with these protocol settings:

```yaml
generation:
  mode: multi_turn_chat
judge:
  default_prompt_preset: default
  turns: all
```

Omitting `turns` judges each prepared example intact. `turns` accepts `all`,
`last`, or numbered turns such as `[1, 3]`. Numbered/all
comparisons require equally many answers from both candidates. `last` compares
complete histories, including histories of unequal length. Generation proceeds
one turn at a time using each candidate's own previous answers.
`generation.system_prompt` optionally sets the generation system message;
`generation.fallback_temperature` applies to categories absent from
`category_temperatures`. Generic defaults are no system message and `0.0`; the
MT-Bench task explicitly selects its original system message and `0.7` fallback.
An explicit model temperature overrides category routing.

Select `fastchat-pairwise` to use MT-Bench's renderer with any compatible dataset.
Its one/two-turn prompts require matching user questions across candidates.
Set reference routing in `judge.prompt_options.reference_categories`; see the
[packaged task](definitions/mt_bench/mt-bench.yaml) for the benchmark defaults.
FastChat combines swapped verdicts by agreement; ordinary presets keep per-pass
preferences.

MT-Bench templates accept `{question}`, `{answer_a}` and `{answer_b}` for one
turn, numbered fields such as `{question_1}` and `{answer_a_2}` for two turns,
and `{ref_answer_1}` and `{ref_answer_2}` for references. This format does not
support parsers requiring token logprobs.

## Multiple references

A pairwise task can compare each candidate answer against every listed reference:

```yaml
baseline:
  strategy: all_references
  references: [reference-a, reference-b, reference-c]
```

The runner prepares each candidate answer once, then loads or generates the
references. Comparison rows keep `source_instruction_index` and have unique IDs.
The candidate must differ from the reference models. Change the list in the task
definition; `model.baseline` cannot override it.

## Independent evaluation

Use `runner: single_answer` to annotate one answer per instruction. Dataset
adapters return the same `CompletionData` contract: instructions plus optional
saved model outputs. Select the saved model through `model.name`; missing
answers are generated. `provided_messages` uses supplied conversation history.

```yaml
protocol:
  runner: single_answer
  generation:
    mode: provided_messages
  judge:
    default_prompt_preset: single-score
  scoring:
    metrics:
      - metric: mean_score
        breakdown_by: [category]
```

The standard renderer supplies `{user_prompt}`, `{completion}`, their `_json`
forms, and dataset metadata such as `{checklist}` or `{history}`. Required
metadata is checked after saved answers are loaded and before generation. Each preset pairs its template with an
`OutputParser`: `parse_result()` returns a dataclass, a JSON-compatible value,
or `None` for an unparseable response. Parsed evidence is retained in `parsed`;
a `score` field is also exposed as the `score` column. This supports correctness
and criterion annotations without requiring scalar scores.

Metrics receive these annotations through the same `calculate_metrics` used by
pairwise evaluation. Omit `scoring` or use `metrics: []` to save annotations
without metrics. The runner returns the annotation dataframe. A baseline and
swapped judging are not used.

## Independent scoring

A single-answer score preset grades each candidate separately before comparing
its scores. MT-Bench-101 configures it as follows:

```yaml
limit_by: dialogue_uid
generation:
  mode: provided_messages
judge:
  default_prompt_preset: mt-bench-101
  minimum_score_by: [dialogue_uid, dialogue_id, task, ability, domain]
```

`limit_by` makes `n_instructions` select complete groups. `minimum_score_by`
compares each model's minimum score per group; omit it to compare individual
rows. The `absolute_score` metric accepts `macro_by` to weight groups equally.
MT-Bench-101's adapter supplies the reference histories used by its prompt.

`judge.score_temperature` controls conversion of scalar score differences into
preferences (default `0.3`): `sigmoid(score_temperature * (score_B - score_A))`.
Choose it for your score scale; larger values give stronger preferences. Raw
scores are unchanged.

## Prompts and metrics

A prompt preset defines the judge's output format and parser. Metrics consume
annotation or battle dataframes. Select them in task YAML, with optional
parameters and breakdowns:

```yaml
scoring:
  metrics:
    - metric: pairwise_win_rate
      breakdown_by: [category]
```

Each `breakdown_by` field produces a separate breakdown. For example,
`[category, turn]` gives results per category and per turn, not per combination.

Task loading validates metric names and parameters. Runtime metric failures
produce warnings and an `error` entry in results; failed breakdowns appear in
`breakdown_errors`. Annotations are still saved. Meta-evaluation also accepts
custom metrics and breakdowns; metrics must account for its reference-pool rows.

Standard pairwise prompt templates can reference adapter metadata columns such
as `{history}` or `{checklist}`. Saved answers load before field validation,
so their metadata is available before generation starts.
The shared renderer supplies `{user_prompt}`, `{completion_A}`, `{completion_B}`
and their `_json` variants; these fields take precedence over source metadata
and follow the actual answer order.

Independent templates receive `{user_prompt}`, `{completion}`, their `_json`
variants, and adapter metadata. Within a `single_answer` protocol, use:

```yaml
judge:
  default_prompt_preset: single-score
scoring:
  metrics:
    - metric: mean_score
```

`single-score` gives one overall score from 1 to 10. `criteria-score` uses the
same rubric definitions as the pairwise `criteria` preset. Both use a packaged
default rubric, or a YAML file supplied through `judge.prompt.options.criteria_file`. It retains each criterion's
1–10 score in `parsed.details.criteria` and sets `score` to their equal-weight
mean. Missing or invalid criteria make that annotation unparsed.

`mean_score` averages available scores without normalization and reports sample
and missing counts. For custom prompt files using another numeric scale, choose
the `json-score` parser: it accepts a finite numeric `score` field and preserves
the remaining JSON evidence. Benchmark tasks keep their configured defaults;
these components can be selected in other task definitions.

Omit `judge.prompt` to inherit the task's default, or select a bundle:

```yaml
judge:
  prompt:
    preset: criteria-score
    options:
      criteria_file: grammar.yaml  # Optional custom rubric
```

Use `judge.prompt.user_file` with `judge.prompt.parser` for a custom template,
instead of `preset`. The parser defaults to `score` for custom files.
`judge.prompt.system_file` is optional. Register an inline bundle with
`register_prompt(JudgePromptPreset(name=..., inline_user=..., parser=...,
judging_mode=...))`, and a metric class with `@register_metric("my_metric")`.
Configurable presets can supply `configure(prompt, **options)`, which returns a
resolved prompt with matching templates and parser. The function owns option
validation; normal keyword arguments reject unknown options. Declare file option
names with `option_files=("rubric",)` so config-relative paths and saved-run
snapshots work for custom bundles too.

Metrics implement `calculate(rows)` and `render(result)`; optional
`required_columns` are checked before calculation. Public entrypoints are available
from `judgearena`, including `register_dataset`, `register_prompt`,
`register_metric`, and `register_parser(name, parser)`. Duplicate names raise an
error. Import registrations before loading tasks or replaying saved configs. See
[prompt registration](../prompts/registry.py), [parsers](../prompts/parsing.py)
and [metrics](../benchmarks/scoring.py).

Metrics can request runtime inputs by declaring keyword parameters, for example
`calculate(self, rows, *, rng, references)`. Shared execution supplies accepted
arguments; other metrics need no signature changes. Each metric gets its own
RNG stream, which continues through its breakdowns. `references` contains the
pairwise baselines, Elo anchor models, or meta-eval's selected model pool.

Independent scalar parsers return `ParsedScore` or `None`; other independent
annotations may use a custom serializable result. Parsers declare `result_type`,
and presets declare `judging_mode` as `pairwise` or `single_answer`.

## Add a workflow

If a benchmark needs a genuinely different evaluation algorithm:

1. Add its typed protocol under `schema/`.
2. Add that protocol to `ProtocolSpec` in `schema/task.py`.
3. Implement and register its benchmark runner.
4. Keep algorithm-specific fields in that protocol, as Elo does in `EloProtocol`.

Protocol fields are available in task YAML and `RunConfig.protocol` through
the same schema. Runners must consume their meaning; a second config field is
not needed. Effective settings follow this precedence:

```text
CLI > run config YAML > task YAML > code default
```

For a small change to a packaged dataset, copy its declaration with
`cfg.resolved_task.spec.dataset.model_dump()`, edit the source revision, and
pass it as `dataset` in the new config. The packaged task stays unchanged.
