# Versioned leaderboards

A release fixes the prompts, opponents (for pairwise tasks), judge and metrics. Each submission uses the task runner to evaluate a candidate against that release. You can vary the candidate's generation settings, including temperature. Create a new release version when you change the benchmark.

## Create a release

Save this as `leaderboard.yaml` to compare models on the same 100 Arena-Hard prompts. Creation saves the selected instructions and the task's baseline answers.

```yaml
task: arena-hard-v0.1
judge: {model: OpenRouter/google/gemma-4-31b-it}
run: {sampling: {limit: 100}}
protocol:
  scoring:
    metrics: [{metric: pairwise_win_rate}]
leaderboard: {version: arena-hard-v1}
```

```sh
release=releases/arena-hard-v1
collection=leaderboards/arena-hard-v1
judgearena leaderboard create --config leaderboard.yaml --output "$release"
```

## Submit a model

Set `MODEL` to the candidate's provider and model ID. Repeat these commands for each checkpoint.

```sh
result="results/arena-hard-v1/$MODEL"
judgearena leaderboard evaluate --leaderboard "$release" \
  --model.name "$MODEL" --model.temperature 0 --output "$result"
judgearena leaderboard add --leaderboard "$release" \
  --submission "$result" --collection "$collection"
```

To generate on one machine and judge later, run `evaluate` with `--stage generate`. Keep the generation directory together. On the judging machine, pass `--answers path/to/generation/answers.parquet` to `evaluate` with the same model settings. Both stages use the [standard inference cache](cache.md).

## Read results

```sh
judgearena leaderboard table --leaderboard "$release" --collection "$collection" --overall
```

Use `--columns` to choose metric fields and `--sort-by` to order the rows. If the metric was configured with `breakdown_by: [lang]`, use `--breakdown lang` for per-language scores. Keep the submission directory to inspect its annotations and metrics. Adding the same result twice produces one collection entry.

## Other release configurations

- [Independent criteria scores](../configs/leaderboard/arena-hard-quality-v1.yaml). The judge scores each answer against a saved rubric. The table can display the overall score and individual fields such as `criteria.clarity.score`.
- [Multilingual battle panel](../configs/leaderboard/lmarena-140k-v1.yaml). With `leaderboard.labels: human`, the dataset's votes set the reference ratings. With `labels: judge`, the configured judge labels the sampled reference battles before the ratings are frozen. The same panel and judge are used for every candidate. `max_reference_comparisons` caps the reference battles per language, and `comparisons_per_opponent` sets the candidate's panel size per opponent and language.

For Elo and Soft-Elo, candidate intervals are conditional on the frozen reference ratings. Set each rating metric's `n_bootstraps` and the run's `run.seed` when creating the release. Every submission uses those values.
