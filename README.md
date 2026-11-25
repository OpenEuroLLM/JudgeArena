# 🏛️ JudgeArena: LLM Evaluation with Swappable Judges

JudgeArena lets you compare language models and study the judges that evaluate them. Choose a benchmark, a model and a judge, then inspect the answers, judgments and scores from the run.

You can run a packaged benchmark or bring your own dataset, prompt, parser and metrics. Models and judges can run locally or through an API. Multilingual tasks let you evaluate across languages using the same setup.

[Benchmarks](docs/benchmarks.md) · [Configuration](docs/configuration.md) · [Models](docs/models.md) · [Cache and results](docs/cache.md) · [Custom tasks](judgearena/tasks/README.md)

## ✨ Key Features

- 🎯 **Run established benchmarks** including AlpacaEval, Arena-Hard, MT-Bench and WildBench.
- 🔄 **Choose your judge** from local models or hosted APIs, independently of the model you evaluate.
- 🌍 **Evaluate across languages** with m-ArenaHard, multilingual fluency and arena datasets.
- 🛠️ **Bring your own components**: datasets, prompt bundles, parsers and metrics.
- 📁 **Inspect and reuse results** with saved annotations, resolved configs and inference caching.

## 🚀 Quick start

Install with Python 3.12 or later:

```bash
pip install judgearena
export OPENROUTER_API_KEY="your-api-key"

judgearena \
  --task arena-hard-v0.1 \
  --model.name OpenRouter/google/gemma-4-31b-it \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --generation.n_instructions 10
```

The task data downloads automatically on the first run. This example uses paid OpenRouter inference.

This generates ten candidate answers and compares them with the task's saved GPT-4 baseline answers. The judge evaluates both answer orders. The run saves its annotations, scores and resolved configuration under `results/`.

The example uses the same model for generation and judging; choose a different `--judge.model` to evaluate with another judge. Ten questions are useful for checking your setup. Omit the limit to evaluate the full task.

For local inference, install `pip install "judgearena[vllm]"`, `pip install "judgearena[sglang]"` or `pip install "judgearena[llamacpp]"`. See [model setup](docs/models.md) for backend options and chat templates.

To work from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/OpenEuroLLM/JudgeArena
cd JudgeArena
uv sync
export OPENROUTER_API_KEY="your-api-key"

uv run judgearena \
  --task arena-hard-v0.1 \
  --model.name OpenRouter/google/gemma-4-31b-it \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --generation.n_instructions 10
```

## ⚙️ Configure an experiment

Save your settings in `run.yaml`:

```yaml
task: arena-hard-v0.1
model:
  name: OpenRouter/google/gemma-4-31b-it
judge:
  model: OpenRouter/google/gemma-4-31b-it
generation:
  n_instructions: 10
```

Run the file, or override individual fields:

```bash
judgearena --config_path run.yaml
judgearena --config_path run.yaml --generation.n_instructions 100
```

Use `--model.sampling_kwargs '{"repetition_penalty":1.1}'` to apply a repetition penalty with vLLM or OpenRouter. Keep shared options such as temperature in their existing fields (e.g. `--model.temperature`). Set judge options with `--judge.sampling_kwargs` and baseline options with `--model.baseline_sampling_kwargs`.

Tasks define benchmark defaults. Experiment configs override selected settings, and CLI flags override the config. Omitted fields inherit task settings when the implementation stays the same; switching the runner or adapter replaces that section. See [configuration](docs/configuration.md) for prompt, metric and dataset overrides.

## 📊 Supported benchmarks

| Benchmark | What it evaluates | Evaluation mode |
|---|---|---|
| AlpacaEval | Instruction-following answers, with length-controlled scoring | Pairwise |
| Arena-Hard v0.1 and v2.0 | Challenging questions, including category-specific judging in v2.0 | Pairwise |
| m-ArenaHard v0.1 and v2.0 | Translated Arena-Hard questions across 23 languages | Pairwise |
| MT-Bench | Multi-turn conversations | Pairwise |
| MT-Bench-101 | Answers with supplied conversation histories | Independent scoring, then score comparison |
| WildBench | Conversations with question-specific checklists | Pairwise reward or independent score |
| Multilingual fluency | Sentence continuations across 43 languages | Pairwise |
| LMArena and ComparIA | Arena battles with human preference labels | Model Elo estimation or judge meta-evaluation |

List the installed tasks and inspect their defaults:

```bash
judgearena tasks list
judgearena tasks show arena-hard-v0.1 --resolved
```

The [benchmark guide](docs/benchmarks.md) lists task names, language selection, baselines and examples. Changing a benchmark's judge or rubric changes the evaluation protocol; record those choices when comparing results.

## 🛠️ Use your own data and judgments

A task can keep its dataset adapter, prompt bundles, parsers and metrics in one folder. Reference a Python component directly from YAML, or register a name in the task family's `register.py` and use it from other configs.

Prompt bundles keep the template, parser selection and judging mode together. Parsing logic stays in Python. You can also pass data directly through the Python API without creating a task file.

- [Create a task or use the Python API](judgearena/tasks/README.md)
- [Write a prompt bundle](judgearena/prompts/README.md)
- [Use decision judges: Choice, Noul and Score](judgearena/prompts/README.md#decision-judging)
- [Reuse cached inference and replay a saved run](docs/cache.md)

## 🤝 Contributing

Bug reports, new benchmarks and documentation fixes are welcome. Open an issue to discuss a change, or send a pull request.

Install the checks before making code changes:

```bash
uv run pre-commit install
```

## 📚 Citation

If you use JudgeArena in your research, please cite:

```bibtex
@article{lushtaku2026judgearena,
  title = {JudgeArena: A Unified Framework for Reproducible LLM-Judge Evaluation},
  author = {Lushtaku, Erlis and Kargi, Bora and Elganzory, Ali and Ferreira, Fabio and Salamanca, Alejandro R. and Kreutzer, Julia and Salinas, David},
  journal = {arXiv preprint arXiv:2608.02620},
  year = {2026},
}
```

See [additional citations](docs/citations.md) for Soft-Elo and the tuned judge configurations.
