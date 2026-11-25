# Changelog

## Unreleased

- Elo CLI flags and YAML keys are deprecated. Use `run.filters`, `run.sampling`, judge settings and scoring metrics instead. Existing saved runs still replay.

## 0.2.0

- Define benchmarks in task YAML and customize runs with config files or CLI overrides.
- Keep benchmark adapters, metrics, parsers and prompt bundles beside their tasks. Register reusable names or reference Python components directly.
- Merge partial config overrides with task defaults and save resolved settings, templates and external components for replay.
- Prepare effective task data and scoring assets for offline runs.
- Organize the README and add guides for configuration, models, benchmarks and caching.

### Updating from 0.1.0

Use the `judgearena` command with `--task`, `--model.name`, `--model.baseline` and `--judge.model` instead of the old script arguments. Use `--config_path run.yaml` for YAML configs. Task names and defaults are listed by `judgearena tasks list` and `judgearena tasks show <task> --resolved`.

Python 3.12 or later is required. Local inference remains optional through the `vllm` and `llamacpp` extras.
