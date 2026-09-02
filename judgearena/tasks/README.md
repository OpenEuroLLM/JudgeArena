# Task definitions

Task YAML describes **what benchmark is being run**. A run config describes
**how one experiment runs it**.

- Keep stable benchmark behavior in the task: pinned data, field mappings,
  baseline policy, judge protocol, and scoring definition.
- Keep experiment choices in `RunConfig`: candidate and judge models, sampling
  parameters, instruction limits, caching, and supported overrides.
- Prefer a new task version when changing data or evaluation behavior. Changing
  only the model or runtime settings does not create a new task.

## Layout

```text
tasks/
├── definitions/       # Packaged task YAML, grouped by benchmark family
├── schema/            # Typed YAML contracts, separated by responsibility
├── loader.py          # Inheritance, strict YAML loading, and stable hashes
├── registry.py        # Discovery, task lookup, and component validation
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

## Add new behavior

Most new tasks should reuse the existing `pairwise` protocol and need no new
runner code.

Keep task YAML boring: declarative facts belong in YAML, while downloading,
format conversion, and scoring algorithms belong in Python.

The judge prompt preset owns the expected output format and its parser. Scoring
metrics consume battle dataframes and return result dictionaries. Task YAML
selects metrics in order and may provide parameters or grouped breakdowns:

```yaml
scoring:
  metrics:
    - metric: pairwise_win_rate
      group_by: [category]
```

Each metric owns its calculation and rendering. Runners only build battle data
and invoke the shared metric executor.

If an upstream dataset has a new format, implement a dataset adapter under
`judgearena/datasets/` and register it in the dataset registry. Task validation
derives the available adapter IDs from that registry. The adapter must return
JudgeArena's canonical columns; the common runner should never check a task
name.

If a benchmark needs a genuinely different evaluation algorithm:

1. Add its typed protocol under `schema/`.
2. Add that protocol to `ProtocolSpec` in `schema/task.py`.
3. Implement and register its benchmark runner.
4. Keep algorithm-specific fields in that protocol, as MT-Bench and ELO do in
   `MTBenchProtocol` and `EloProtocol`.

Adding a schema field makes it valid in task YAML; it does not automatically
make it a run-config or CLI option. Runtime overrides must be explicitly added
to `RunConfig` and resolved with this precedence:

```text
CLI > run config YAML > task YAML > code default
```

Dataset revisions and evaluation definitions should normally remain
task-owned. Expose overrides only for settings users are expected to change
between experiments.
