# Cache and saved results

The inference cache stores raw completions and judge responses. Result directories store a run's annotations, metrics and resolved configuration. Keep both if you want to inspect results and reuse inference later.

## Reuse inference

The default cache is `${XDG_CACHE_HOME:-~/.cache}/judgearena`. Set another location in your config:

```yaml
run:
  store_root: ./cache
  result_folder: results
```

Cache entries depend on the rendered input and resolved model settings, within their role and task. Identical requests can reuse responses. Changing a prompt, answer order, model setting or task can require new inference. `run.store_root: null` disables runtime caching.

Cache read/write failures are logged and inference can continue; malformed model descriptors still raise errors. Completion and judgment caches are stored separately.

## Replay a run

Use the `config.yaml` inside a completed run directory:

```bash
uv run judgearena --config_path results/<run>/config.yaml
```

Replay uses the saved effective settings and available cache entries. It writes another result directory rather than overwriting the original. Missing cache entries may trigger model calls; replay is not an offline-only mode.

Saved configs include expanded prompt bundles and copies of referenced templates, rubrics and external Python components. Keep these files beside the config when moving a run. Imported packages and other dependencies of custom components must still be installed. Data supplied directly through Python must be supplied again.

## Inspect outputs

| Artifact | Contents |
|---|---|
| `config.yaml` | Resolved task and experiment settings |
| `*-annotations.csv` for pairwise/independent runs; `annotations.parquet` for Elo/meta-eval | Judge evidence and parsed annotations |
| `battles.parquet` in Elo/meta-eval runs | Rows used for scoring |
| `results-*.json` for pairwise/independent runs | Metric results and report metadata |
| `run-metadata.v1.json` | Run metadata and reported inference usage |

Additional report files depend on the workflow. Pairwise `pref` follows the original answer order: `0` favors A, `1` favors B and `0.5` is a tie. `swapped` records reversed judge input. Parsed scores follow the judge's input order. Independent annotations include the answer and parsed result, with a `score` column when the parser returns one.

Nested and mixed-value columns are JSON-encoded in exports. See the [data contract](../judgearena/tasks/README.md#dataset-contracts) before joining exported metadata.

Token and cost totals cover successful responses returned by completed inference calls. They do not include unobserved provider retries or responses discarded by a failed call. Costs are available only when the provider reports them; cached responses do not represent new API spending.

## Share a cache

Cache synchronization is explicit. The default Hugging Face dataset is `judge-arena/judge-arena-cache`; pass `--hf_repo` to use another repository.

```bash
uv run judgearena-cache --action fetch --store_root ./cache \
  --kind completions --task alpaca-eval

uv run judgearena-cache --action push --store_root ./cache \
  --kind judgements --task alpaca-eval --model OpenRouter/org/model
```

`--kind`, `--task` and `--model` filter the transfer. Use `--all` explicitly to transfer without filters. Uploading shares the selected cached inputs and responses with the destination repository.

For dataset downloads and offline execution, see [offline preparation](../judgearena/tasks/README.md#prepare-an-offline-run).

Before writing judgments without model attribution to an older cache, stop cache writers and run `python scripts/migrate/001_nullable_judgement_models.py /path/to/cache` from the checkout. The script keeps a `.bak` copy; new caches need no migration.
