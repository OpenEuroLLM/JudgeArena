"""Run a submission with the release's frozen inputs and evaluation settings."""

import json
from pathlib import Path

import pandas as pd

from judgearena.artifacts.evaluation import load_evaluation
from judgearena.benchmarks.runner import run_benchmark
from judgearena.config import RunConfig, load_config
from judgearena.leaderboard.release import checksum, load_release, release_inputs
from judgearena.leaderboard.submissions import result_rows


def evaluate_release(
    release,
    model,
    output,
    *,
    model_outputs=None,
    n_bootstraps=200,
    seed=42,
    store_root=None,
):
    """Vary candidate settings; delegate generation, judging and metrics to the runner."""
    release = Path(release)
    manifest, ratings = load_release(release)
    if not model.name or model.name in set(manifest.get("model_names", ratings.model)):
        raise ValueError("Submission needs a new model/checkpoint name")
    cfg = load_config(release / "execution/config.yaml")
    if any(key.startswith("baseline") for key in model.model_fields_set):
        raise ValueError("Submission cannot override the frozen baseline")
    values = cfg.model_dump()
    values["model"] = {
        "baseline": cfg.model.baseline,
        **model.model_dump(exclude_unset=True),
    }
    cfg = RunConfig(**values)
    if store_root is not None:
        cfg.run.store_root = str(Path(store_root).resolve())
    if model_outputs is not None:
        model_outputs = model_outputs.loc[model_outputs.model == model.name]
    data, options = release_inputs(release, manifest, ratings, model_outputs)
    interval_settings = {}
    required_ratings = []
    if "frozen" in options:
        from judgearena.benchmarks.elo.scoring import frozen_metric_inputs

        frozen = options.pop("frozen")
        required_ratings = [method["name"] for method in frozen["methods"]]
        metrics, runtime = frozen_metric_inputs(
            cfg.resolved_task.spec.protocol.scoring.metrics,
            ratings=frozen["ratings"],
            methods=frozen["methods"],
            score_metrics=frozen["score_metrics"],
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        options.update(
            opponent_panel=frozen["panel"],
            metrics=metrics,
            metric_runtime=runtime,
            sampling_metadata={
                "sampling_mode": "fixed",
                "intervals": "conditional on frozen references",
                "bootstrap_seed": seed,
                "n_bootstraps": n_bootstraps,
            },
        )
        if frozen["methods"]:
            interval_settings = {"bootstrap_seed": seed, "n_bootstraps": n_bootstraps}
    evaluation_path = run_benchmark(cfg, data=data, output_dir=output, **options)
    evaluation = load_evaluation(evaluation_path)
    for name in required_ratings:
        error = evaluation["metrics"][name].get("error")
        if error:
            raise ValueError(f"Cannot submit {name}: {error}")
    output = evaluation_path.parent
    (output / "release.json").write_text(
        json.dumps(
            dict(
                version=manifest["version"],
                files=manifest["files"],
                submission_files={
                    **evaluation["files"],
                    evaluation_path.name: checksum(evaluation_path),
                },
                **interval_settings,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Results written to {output.resolve()}")
    return pd.DataFrame(result_rows(evaluation))
