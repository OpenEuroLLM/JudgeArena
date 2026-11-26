# Versioned leaderboards

A release freezes evaluation inputs, baseline answers or a battle panel, and the
judge/prompt/scoring configuration. Each submission saves its results separately;
it never changes the release or earlier submissions.

Evaluation delegates to the existing task runner. Custom adapters, prompt bundles,
parsers and metrics use the same task/config references as a normal run.
`evaluation.json` records the model, metrics and checksummed artifact paths;
collection reads that manifest rather than assuming runner filenames. Frozen Elo
and its bootstrap intervals are calculated by the rating metric.

## Baseline win rate

```yaml
task: arena-hard-v0.1
judge:
  model: OpenRouter/google/gemma-4-31b-it
run:
  sampling: {limit: 100}
protocol:
  scoring:
    metrics:
      - metric: pairwise_win_rate
leaderboard:
  version: arena-hard-winrate-v1
```

```bash
judgearena leaderboard create \
  --config configs/leaderboard/arena-hard-winrate-v1.yaml \
  --output releases/arena-hard-winrate-v1

judgearena leaderboard evaluate \
  --leaderboard releases/arena-hard-winrate-v1 \
  --config configs/leaderboard/submission.yaml \
  --output results/my-model
```

Creation uses the task's baseline assignment and saves its answers, loading
available answers and generating missing ones. Instructions follow the usual
`run.sampling.limit`, task `limit_by` and `run.filters` settings. Category
baselines and all-reference assignments use the ordinary pairwise preparation.
No candidate answers are generated during creation. Submit each initial model
and each later model with the same evaluate command. `results.json` contains the
configured metrics and breakdowns. Keep the task's scoring declaration instead
of overriding it when you want its benchmark-specific metric.

## Independent criteria scores

Use `single_answer` to score each model on the same frozen instructions without
a baseline. The example rubric measures clarity and completeness; `mean_score`
reports their individual means and counts alongside the overall score.

```bash
uv run judgearena leaderboard create \
  --config configs/leaderboard/arena-hard-quality-v1.yaml \
  --output releases/arena-hard-quality-v1

uv run judgearena leaderboard evaluate \
  --leaderboard releases/arena-hard-quality-v1 \
  --model.name OpenRouter/qwen/qwen3.5-9b \
  --output results/arena-hard-quality-v1/qwen3.5-9b

uv run judgearena leaderboard add \
  --leaderboard releases/arena-hard-quality-v1 \
  --submission results/arena-hard-quality-v1/qwen3.5-9b \
  --collection leaderboards/arena-hard-quality-v1

uv run judgearena leaderboard table \
  --leaderboard releases/arena-hard-quality-v1 \
  --collection leaderboards/arena-hard-quality-v1
```

Repeat evaluate/add with another model. Nested metric values appear as columns
such as `criteria.clarity.score`; all raw annotations remain in each submission.
The release snapshots the rubric, so editing the original file does not change it.
If every response fails parsing, the overall missing count is reported; criterion
names cannot be recovered from those failed annotations.

## Battle panels and frozen ratings

Create a release directly from a task and one config:

```bash
judgearena leaderboard create \
  --config configs/leaderboard/lmarena-140k-v1.yaml \
  --output results/leaderboards/lmarena-human-v1
```

Creation accepts the same CLI overrides as a normal run, plus `--leaderboard.*`.
Values resolve as task defaults → config → CLI. Both `--config` and
`--config_path` work; omit the file to supply everything through flags:

```bash
judgearena leaderboard create \
  --task arena-hard-v0.1 \
  --judge.model OpenRouter/google/gemma-4-31b-it \
  --run.sampling.limit 100 \
  --protocol.scoring.metrics '[{"metric":"pairwise_win_rate"}]' \
  --leaderboard.version arena-hard-winrate-v1 \
  --output releases/arena-hard-winrate-v1
```

For a multilingual graph, a config can be narrowed with
`--run.filters '{"lang":["en","de"]}'` and
`--leaderboard.sampling.comparisons_per_opponent 10`. Evaluation keeps the
release fixed and accepts only candidate model overrides.

The config uses the usual task, dataset overrides, judge and runtime settings.
Use existing metric requests; `leaderboard` contains only
release and sampling settings:

```yaml
run:
  filters: {lang: [en, pl, de, fr, es]}
protocol:
  scoring:
    metrics:
      - metric: bradley_terry
        name: elo
        parameters: {soft: false, baseline_model: gemma-3-27b-it, n_bootstraps: 200}
      - metric: bradley_terry
        name: soft_elo
        parameters: {soft: true, baseline_model: gemma-3-27b-it, n_bootstraps: 200}
leaderboard:
  version: lmarena-human-v1
  labels: human
  sampling:
    validation_fraction: 0.1
    minimum_model_comparisons: 50
    comparisons_per_opponent: 10
    # Optional explicit selection:
    # opponents: [gemma-3-27b-it, gemini-2.5-pro]
    # Optional per-language reference-row cap, keeping whole prompt groups:
    # max_reference_comparisons: 1000
```

Omit `opponents` to select the models that meet the minimum support and connected-graph
checks in every selected language when rating metrics are configured. Win-rate-only
releases apply the support threshold without requiring a connected graph. The chosen names are saved with the release.
An explicit list restricts the panel to those models. Each must have enough distinct
prompt groups for its panel quota; otherwise creation fails.

Creation loads through the task's
adapter, splits by session/prompt group, checks graph support and selects equal
panel quotas per opponent and language. Missing session IDs fall back to grouping
identical prompt text. Task/runtime filters, including language selection, apply before grouping
and the held-out split.
Changing bootstrap count does not change the sampled panel.

Choose the source of reference ratings explicitly:

| Release setting | Reference graph | New candidate comparisons |
|---|---|---|
| `labels: human` | Dataset's human preferences; no reference judging | Configured judge |
| `labels: judge` | Configured judge annotates the sampled graph | Same judge and prompt |

Human-reference evaluation mixes human reference ratings with judge-based candidate
comparisons. Judge-reference evaluation uses the same labeling procedure for both.
The examples below use the same config and sampling limits:

```bash
uv run judgearena leaderboard create \
  --config configs/leaderboard/multilingual-judge-v1.yaml \
  --leaderboard.labels human --leaderboard.version multilingual-human-v1 \
  --output releases/multilingual-human-v1

uv run judgearena leaderboard create \
  --config configs/leaderboard/multilingual-judge-v1.yaml \
  --output releases/multilingual-judge-v1
```

The judge version annotates the sampled references, not all LMArena battles.
`max_reference_comparisons` caps reference rows per language; panel quotas control
candidate comparisons separately. Selection may differ because human releases can
check connectivity using human labels, while judge releases select before judging.

Each graph release contains reference battles, held-out battles, the fixed submission
panel, metric results and (when requested) per-language ratings with grouped bootstrap intervals,
a short report, and checksummed execution settings. Reference bootstrap fits
must retain every model and connectivity; at least 90% of draws must be valid.
The baseline is anchored at 1000. Review the sampled prompts before publication.

To reuse curated files instead of loading/sampling the task, add
`--candidate path/to/candidate`. This reads `reference_battles.parquet`,
`panel.parquet` and, when present, `heldout_battles.parquet` (or the older
`validation_battles.parquet`); sampling options
are not applied. The manifest records `sampling: null` and the actual languages
for curated inputs.

Evaluate a new model against the release:

```bash
judgearena leaderboard evaluate \
  --leaderboard results/leaderboards/lmarena-human-v1 \
  --config configs/leaderboard/submission.yaml \
  --store-root ./cache/leaderboard \
  --n-bootstraps 200 \
  --output results/leaderboards/my-submission
```

`--config` (also `--config_path`) is optional. Use the usual `--model.*` flags
alone or to override YAML values:

```bash
judgearena leaderboard evaluate \
  --leaderboard releases/arena-hard-winrate-v1 \
  --model.name VLLM/Qwen/Qwen3-8B \
  --model.temperature 0 \
  --output results/qwen3-8b
```

Submission YAML contains only `model:` settings. Baseline settings and non-model
CLI overrides are rejected; omitted candidate settings inherit release defaults. The release supplies the task,
prompt, judge and generation preparation. To reuse saved answers, pass
`--answers path/to/answers.parquet`. These rows contain `instruction_index`
(the original instruction ID for baseline releases, or panel battle ID for graph releases), `model`, `completion` and optional output metadata.
Evaluations save this same format for reuse.

The metric requests choose hard (`soft: false`) or fractional preferences
(`soft: true`); their `name` values identify result rows. Add `pairwise_win_rate` to report win rate as well, or use it alone without
fitting ratings. Remove a request to omit a metric. Metric parameters can override the shared `baseline_model` and
`n_bootstraps` defaults. Reference bootstrap draws use the metric settings;
`evaluate --n-bootstraps` controls the candidate's conditional intervals.
PairScore uses `elo.soft_elo_temperature`, resolved from the task or config before
freezing. Human reference labels produce identical reference ratings for both
methods. The candidate's 95% intervals resample prompt groups, keeping both
answer orders together, **conditional on frozen reference ratings**. Perfect
wins or losses produce infinite estimates and need more informative comparisons.

Rerunning reuses inference caches. `--store-root` relocates both generation and
judging caches without modifying the release. Evaluation prints the actual
output directory, including rerun suffixes. Failed creation can retry the same
output path; completed releases cannot be overwritten. Changes to the graph,
panel, judge or rating policy belong in a new version and directory.

Missing judge preferences are excluded from reference scoring. The release keeps
all annotations and reports valid/missing counts by language in `results.json`
and `report.md`. Frozen ratings still require every reference model and a
connected graph after exclusion.

Graph `results.json` stores historical rates under `observed_reference_metrics`.
These describe each model's observed opponents; they are not matched-panel rates
and should not be ranked alongside submission panel win rates. Baseline releases
compare submissions against the same frozen answers and instructions.

### Collect submissions

Evaluation saves a result folder. Add it to a collection for the same release:

```bash
judgearena leaderboard add \
  --leaderboard releases/multilingual-v2 \
  --submission results/multilingual-v2/my-model \
  --collection leaderboards/multilingual-v2

judgearena leaderboard table \
  --leaderboard releases/multilingual-v2 \
  --collection leaderboards/multilingual-v2
```

The collection stores scores and a link to the complete run. It checks the release
and saved artifact checksums before accepting a submission. Adding the same result
again does nothing; conflicting results or bootstrap settings for the same model name are rejected.
Use matching settings for replay, a separate collection for alternative estimates,
and separate checkpoint names for different models. Older result folders without
artifact checksums need a cache replay before collection.

The table shows saved reference Elo/Soft-Elo alongside submissions, grouped by
language and method. `entry_type` distinguishes `reference` from `submission`.
Reference intervals resample the reference graph; submission intervals hold those
reference ratings fixed. Win-rate rows contain submissions only.
Reference ratings stay frozen. Historical graph win rates are not mixed with
submission panel win rates, and adding entries never refits the graph. Keep the
saved run folders if you want to inspect answers and annotations later.

Choose displayed fields and sorting explicitly. Without `--sort-by`, the table
only groups rows by language, metric and model; it does not assign a ranking.

```bash
judgearena leaderboard table \
  --leaderboard releases/arena-hard-quality-v1 \
  --collection leaderboards/arena-hard-quality-v1 \
  --columns model method score criteria.clarity.score criteria.completeness.score \
  --sort-by score --descending
```

Any scalar or nested metric value can be displayed this way. Metrics remain
stored in full, including breakdowns; selecting columns does not change scores.
Use `--overall` to show overall results only, or `--breakdown lang` to show
results grouped by language. The selected breakdown column is included even when
`--columns` omits it. Any stored breakdown field can be selected. Without
either option, the table includes overall results and all breakdowns.

### Ordinary Elo runs

Pairwise execution prepares the comparison graph and reuses answer generation,
judging, metrics and artifacts. Configure sampling through `graph`, preference
conversion through `protocol.judge`, and ratings through metric parameters.

Ordinary LMArena Elo fits references from the supplied `reference_pref` labels
(human labels in the packaged dataset), then judges the candidate comparisons.
It does not rejudge the reference graph. By default, reference ratings stay fixed
while fitting the submitted model. It uses the same candidate fit and conditional
bootstrap intervals as a leaderboard. A leaderboard also pins the panel, reference
ratings and evaluation settings across submissions.

Set `fit_mode: joint` in the Bradley–Terry metric parameters to refit references and the candidate together, as in
previous versions. Cached answers and judgments remain reusable; rescoring them
with the new default can change ratings and confidence intervals.

### Comparison graphs and metrics

A pairwise task can use graph sampling by specifying `leaderboard.sampling`, or
load an existing panel with `--candidate`. Without those settings it keeps its
normal baseline preparation. Use `baseline: {strategy: none}` when the panel
supplies the opponents. Graph submissions use the same answer generation,
judging and metric execution as other pairwise evaluations.

Choose ratings through `protocol.scoring.metrics`. Frozen reference ratings are
inputs to the Bradley–Terry metric; they do not change how answers are judged.
For PairScore prompts, `protocol.judge.preference_temperature` explicitly controls
score-to-preference conversion. If omitted, the parser's conversion is retained.
Soft-Elo uses continuous preferences and does not require calibration. For PairScore
prompts, optional calibration fits the conversion temperature against reference
labels; it does not fit reference ratings. Verdict/probability parsers can supply
preferences directly. The Arena-Hard examples above work without calibration.

### Graph preparation without Elo settings

Use `run.sampling` to sample saved comparisons and `run.filters` to select languages.
Preference conversion belongs to the judge; changing metrics does not change it.
For example, a config can inherit the LMArena dataset and use ordinary pairwise execution:

```yaml
task: elo-lmarena-140k
run:
  sampling: {group_by: lang, per_group: 100}
  seed: 42
  filters: {lang: [en, de]}
protocol:
  runner: pairwise
  generation: {mode: single_turn_chat}
  baseline: {strategy: recorded_opponents}
  judge:
    default_prompt_preset: default
    preference_temperature: 0.3
  scoring:
    metrics:
      - metric: pairwise_win_rate
      - metric: bradley_terry
        parameters: {soft: true, fit_mode: frozen, n_bootstraps: 200}
```

Supply `model.name` and `judge.model` as usual. Both metrics use the same judged
comparisons. The rating metric also consumes the available reference labels.
Win rate in this pairwise graph workflow works without reference labels.
Graph runs save a `ComparisonReport`.

For PairScore calibration, set `protocol.judge.calibrate_temperature: true` and
optionally `calibration_size` and `calibration_min_battles`. Calibration requires
reference comparisons. If using calibration, fit it before creating a release and
save the resulting `preference_temperature`; leave `calibrate_temperature: false`
in the release config.
Existing `elo` sampling and conversion fields remain supported for replay;
explicit graph and judge settings take precedence.

Legacy joint-Elo uses a 500-appearance minimum for reference models. Set
`reference_min_battles: 500` on a directly configured Bradley–Terry metric to
reproduce that subset; this does not change the candidate comparisons.

### Generate answers before judging

Use `--stage generate` to generate candidate answers without loading a judge or
calculating metrics. Model settings use the same config and CLI overrides as an
ordinary submission; release instructions remain frozen.

```bash
judgearena leaderboard evaluate --leaderboard releases/my-benchmark \
  --config model.yaml --stage generate --output results/answers

judgearena leaderboard evaluate --leaderboard releases/my-benchmark \
  --config model.yaml --answers results/answers/answers.parquet \
  --output results/evaluated
```

Keep the generation settings in `model.yaml` the same between these commands.
The answer folder contains `answers.parquet`, `instructions.parquet`, the effective
`config.yaml` and a `generation.json` manifest. Judging verifies file checksums,
instructions, model settings and release identity. Missing candidate answers are
rejected rather than silently generated. Only scored results can be added to a
leaderboard. Generation can be repeated using the completion cache.

Ordinary runs also support `--stage generate`, without a judge or baseline:

```bash
judgearena --task arena-hard-v0.1 --stage generate \
  --model.name VLLM/Qwen/Qwen3.5-9B --model.max_out_tokens 4096 \
  --run.store_root ./cache --run.result_folder results/generation
```

To judge later, load the saved `config.yaml`, set `--stage evaluate`, supply
`--judge.model`, `--answers /path/to/answers.parquet` and a baseline if the task
requires one. Keep the answer folder together when moving it between machines.
Finish reasons are not currently retained by the inference cache.
