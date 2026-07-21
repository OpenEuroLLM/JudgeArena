# Task definitions

Task YAML describes **what benchmark is being run**. A run config describes
**how one experiment runs it**.

See [`docs/task-architecture.md`](../../docs/task-architecture.md) for the
runtime flow, component ownership, and design rules.

## Files you usually touch

| Change | Location |
| --- | --- |
| Add or update a packaged task | `definitions/<family>/*.yaml` |
| Normalize a new upstream data format | `judgearena/datasets/` |
| Add a reusable judge prompt and parser | `judgearena/prompts/` |
| Add a genuinely different evaluation workflow | `schema/` and `judgearena/benchmarks/` |

Most new single-turn pairwise tasks require only a task YAML file. They should
reuse registered dataset, prompt, scoring, and runner components.

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
    default_prompt: default
  scoring:
    adapter: pairwise_win_rate
```

The task definition owns stable benchmark facts: pinned sources, canonical
field mappings, generation mode, baseline policy, judge defaults, and scoring
adapter. Model IDs, sampling parameters, instruction limits, caching, and
output paths belong in `RunConfig`.

Validate and inspect the resolved task before running it:

```bash
judgearena tasks validate example-pairwise
judgearena tasks show example-pairwise --resolved
```

The resolved view includes inherited values and hashes of every YAML resource.

## Share settings across a task family

Use a private `_base.yaml` when versions or variants share meaningful benchmark
policy:

```yaml
# _base.yaml
schema_version: 1
task_version: 1
tags: [pairwise]

dataset:
  adapter: example
  fields:
    id: id
    instruction: prompt

protocol:
  runner: pairwise
  generation:
    mode: single_turn_chat
  judge:
    default_prompt: default
  scoring:
    adapter: pairwise_win_rate
```

```yaml
# example-v2.yaml
extends: _base.yaml
task: example-v2
description: Example benchmark version 2.

dataset:
  sources:
    examples:
      type: huggingface_dataset
      repo_id: organization/example-v2
      revision: "0123456789abcdef0123456789abcdef01234567"

protocol:
  baseline:
    strategy: runtime_required
```

Maps merge recursively, lists replace inherited lists, and `null` removes an
inherited key. Avoid inheritance when it only saves a few lines.

Use `variants` when suffixes select views of the same underlying definition,
such as language selections. Use separate public task YAML files when the
source data or benchmark policy differs.

## Add new behavior

Keep task YAML declarative. Downloading, format conversion, prompt parsing, and
scoring algorithms remain Python components selected by ID.

For a new upstream data format:

1. Implement an instruction or battle dataset adapter in
   `judgearena/datasets/`.
2. Register it once in `judgearena/datasets/registry.py`.
3. Return JudgeArena's canonical columns so runners never branch on task names.
4. Select the adapter from the task YAML.

For a genuinely different evaluation algorithm:

1. Add its typed protocol schema under `schema/`.
2. Add the protocol to `ProtocolSpec` in `schema/task.py`.
3. Implement and register its runner under `judgearena/benchmarks/`.
4. Keep algorithm-specific task fields in that protocol schema.

`MTBenchProtocol` and `EloProtocol` are examples of specialized protocols.
Single-turn pairwise datasets should continue to use `PairwiseProtocol` and the
shared pairwise runner.

Adding a schema field only makes it valid in task YAML. If users should be able
to override it per experiment, add an explicit `RunConfig` field and resolution
logic. The effective precedence is:

```text
CLI > run config YAML > task YAML default > code default
```

## Change or remove a task

- Change only the run config for experiment-specific choices.
- Create a new task or increment `task_version` when stable data or evaluation
  behavior changes.
- Remove a task by deleting its public YAML. Remove Python components only when
  no remaining task references them.
- Run `judgearena tasks validate` to validate every packaged definition.
