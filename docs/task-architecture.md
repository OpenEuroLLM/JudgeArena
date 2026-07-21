# Task architecture

JudgeArena packages benchmark identity and stable evaluation policy as typed
YAML task definitions. This makes adding, changing, or removing a benchmark a
local operation instead of extending a central function with more task-name
branches.

This document explains how the task system is divided. For the shortest path to
adding a task, see [`judgearena/tasks/README.md`](../judgearena/tasks/README.md).

## Runtime flow

Every packaged task enters through one dispatcher. The resolved task chooses a
compatible runner; it does not force benchmarks with different algorithms into
one function.

```text
CLI flags + run config YAML
            |
            v
        RunConfig
            |
            | cfg.task
            v
        TaskRegistry
            |
            | load, inherit, validate, fingerprint
            v
     ResolvedTaskSpec
            |
            | protocol.runner
            v
     Benchmark registry
       |       |         |       |
       v       v         v       v
   pairwise  mt_bench  wildbench  elo
       |       |         |       |
       +-------+---------+-------+
               |
               v
    dataset + prompt + scorer
               |
               v
      artifacts and metadata
```

At the top level, `judgearena/benchmarks/runner.py` performs only resolution and
dispatch:

```python
resolved = resolve_benchmark(cfg.task)
return resolved.adapter.runner(cfg, resolved.task)
```

The selected runner receives the already resolved task definition. It should
not rediscover benchmark identity from prefixes or duplicate task maps.

## Ownership

| Area | Main files | Owns |
| --- | --- | --- |
| Task definitions | `judgearena/tasks/definitions/` | Stable sources, fields, protocol, baseline policy, judge defaults, and scorer selection |
| Task schemas | `judgearena/tasks/schema/` | Strict contracts for valid YAML and protocol-specific fields |
| Loading | `judgearena/tasks/loader.py` | YAML safety, inheritance, normalization, and hashes |
| Task registry | `judgearena/tasks/registry.py` | Discovery, task and variant lookup, and referenced-component validation |
| Dispatch | `judgearena/benchmarks/registry.py`, `judgearena/benchmarks/runner.py` | Mapping `protocol.runner` to executable workflows |
| Benchmark runners | `judgearena/benchmarks/` | Evaluation control flow for pairwise, MT-Bench, WildBench, and ELO |
| Dataset adapters | `judgearena/datasets/` | Downloading upstream resources and normalizing their data shape |
| Prompt presets | `judgearena/prompts/` | Judge prompts, output format, and parsing policy |
| Scoring adapters | `judgearena/benchmarks/*/scoring.py` | Metric calculation and result summaries |
| Runtime config | `judgearena/config.py` | Models, sampling, limits, cache behavior, output paths, and supported overrides |
| Run metadata | `judgearena/artifacts/metadata.py` | Reproducibility hashes, resolved task provenance, and produced artifacts |

The main boundary is simple: YAML selects registered behavior; Python
implements it.

## Task definition and run config

A task definition answers questions that should remain stable across repeated
runs:

- Which upstream resources and revisions define the benchmark?
- Which columns map to JudgeArena's canonical fields?
- Is generation single-turn, base completion, or multi-turn?
- How is the baseline selected?
- Which judge prompt and scoring algorithm define the evaluation?
- Which runner can execute this protocol?

A run config answers experiment-specific questions:

- Which candidate, baseline override, and judge models are used?
- Which generation and judge sampling parameters are used?
- How many examples are evaluated?
- Is the cache ignored, and where are artifacts written?
- Which ELO filters, sampling limits, or bootstrap settings are used?

Task defaults are applied only when the run config does not set a supported
override:

```text
CLI > run config YAML > task YAML default > code default
```

Not every task field should be overridable. Dataset revisions and benchmark
identity normally remain task-owned. A new run config field is appropriate only
when users are expected to change that value between experiments.

## Resolution and validation

`TaskLoader` reads a public YAML definition, resolves any private `_*.yaml`
base, validates the normalized result as a `TaskSpec`, and records provenance.

Inheritance follows three rules:

- mappings merge recursively;
- lists replace inherited lists;
- `null` removes an inherited key.

`TaskRegistry` then validates every selected component against its owning
registry. A task cannot select an unknown runner, dataset adapter, prompt
preset, or scorer.

The result is a `ResolvedTaskSpec` containing:

- the validated task definition;
- the invocation ID and any selected variant;
- the source path and source hash;
- a normalized resolved hash;
- hashes for all inherited YAML resources.

The resolved hash and resource hashes are written to run metadata. Judge prompt
and evaluated-input hashes are recorded separately, so changes to task policy,
prompt content, or run inputs remain visible without copying those full values
into metadata.

## Shared and specialized runners

The runner boundary follows the evaluation algorithm, not the dataset name.

### Pairwise

The shared pairwise runner covers tasks that follow the same workflow:

```text
load instructions
      |
      v
generate or load candidate and baseline outputs
      |
      v
judge pairwise preferences
      |
      v
summarize with the selected scorer
```

AlpacaEval instruction-set evaluation, Arena-Hard, m-ArenaHard, and fluency
tasks can share this runner because their dataset adapters normalize them to
the same interface. The runner must not contain `if task == ...` branches.

### MT-Bench

MT-Bench keeps a specialized runner because it owns multi-turn generation,
turn-specific judging, reference answers for selected categories, and
FastChat-compatible prompt behavior. Its stable policy still lives in task
YAML and it still enters through the common dispatcher.

### ELO

ELO keeps a specialized runner because it samples human arena battles,
generates one focal model's responses, combines LLM judgments with human
anchors, and fits Bradley-Terry ratings. Arena identity, battle sources, judge
defaults, and scoring defaults live in ELO task YAML; sampling, filtering,
calibration, and bootstrapping remain runtime options.

### WildBench

WildBench keeps a specialized runner because WB-Score and WB-Reward use checklist-aware official prompts, conversation inputs, official reference outputs, and benchmark-specific aggregation. Its two public YAML definitions share a family base and select their mode, baseline policy, prompt, and versioned scorer without adding task-name branches to shared execution code.

Use the shared pairwise runner by default. Add a specialized protocol and
runner only when the data shape or evaluation algorithm genuinely changes.

## Dataset capabilities

Dataset adapters hide upstream layout from benchmark runners. Two capabilities
are currently explicit:

```text
InstructionDatasetAdapter
  - download(task, path)
  - load_instructions(task, path)
  - load_model_outputs(task, path)

BattleDatasetAdapter
  - download(task, path)
  - load_battles(task, path)
```

Instruction adapters serve pairwise and MT-Bench workflows. Battle adapters
serve ELO workflows. `TaskRegistry` validates that each protocol selects a
compatible adapter kind, so an ELO task cannot accidentally select an
instruction-only adapter.

An adapter may contain source-specific conversion logic. A runner should see
only JudgeArena's canonical fields and should not know which upstream file
format produced them.

## Task families and variants

Use a private `_base.yaml` for stable policy shared by multiple public task
definitions, such as Arena-Hard versions or ELO arenas. Each public YAML keeps
its own task ID, description, source revisions, and differences from the base.

Use suffix variants when one definition exposes several views of the same
source and policy. For example, an m-ArenaHard task can resolve a language or a
declared language group without copying the complete task definition.

```text
m-arena-hard-v2.0
        |
        +-- m-arena-hard-v2.0-en
        +-- m-arena-hard-v2.0-fr
        +-- m-arena-hard-v2.0-EU
```

The selected suffix becomes a validated `TaskSelection` passed to the dataset
adapter. It is also recorded in metadata.

## Extending the system

Choose the smallest extension that matches the change:

| Need | Add |
| --- | --- |
| Existing data shape and pairwise behavior | One task YAML |
| New upstream data format | Task YAML plus one dataset adapter |
| New reusable judge format | Prompt preset and parser, then select it in YAML |
| New scoring algorithm for an existing protocol | Scoring adapter, then select it in YAML |
| New evaluation algorithm | Typed protocol, runner, and registrations |

After a change, validate definitions and inspect the resolved task:

```bash
judgearena tasks validate
judgearena tasks show <task-id> --resolved
```

Changing runtime models or sample counts does not require a new task. Changing
pinned data or stable evaluation policy should create a new task version or a
clearly versioned task definition. Removing a benchmark should normally require
deleting its public YAML; shared Python components stay until no task uses them.

This architecture makes benchmark packages independently understandable while
keeping reusable execution logic centralized.
