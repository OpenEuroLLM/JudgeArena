"""Run a submission with the release's frozen inputs and evaluation settings."""

import json
from pathlib import Path

import pandas as pd

from judgearena.artifacts.evaluation import load_evaluation
from judgearena.benchmarks.runner import run_benchmark
from judgearena.config import RunConfig, load_config
from judgearena.leaderboard.release import (
    bootstrap_settings,
    checksum,
    load_release,
    release_inputs,
)
from judgearena.leaderboard.submissions import result_rows


def evaluate_release(
    release,
    model,
    output,
    *,
    model_outputs=None,
    stage="evaluate",
    answers=None,
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
    values.update(stage=stage, answers=answers)
    cfg = RunConfig(**values)
    if store_root is not None:
        cfg.run.store_root = str(Path(store_root).resolve())
    if model_outputs is not None:
        model_outputs = model_outputs.loc[model_outputs.model == model.name]
    identity = {key: manifest[key] for key in ("version", "files")}
    if answers is not None:
        generated_manifest = Path(answers).parent / "generation.json"
        if generated_manifest.exists():
            generated = json.loads(generated_manifest.read_text())
            if generated.get("release") != identity:
                raise ValueError(
                    "Saved answers belong to a different leaderboard release"
                )
    data, options = release_inputs(release, manifest, ratings, model_outputs)
    if stage == "generate":
        generated_path = run_benchmark(
            cfg,
            data=data,
            output_dir=output,
            opponent_panel=options.get("frozen", {}).get("panel"),
        )
        generated = json.loads(generated_path.read_text())
        generated["release"] = identity
        generated_path.write_text(json.dumps(generated, indent=2) + "\n")
        return generated_path

    interval_settings = bootstrap_settings(manifest)
    required_ratings = []
    if "frozen" in options:
        from judgearena.benchmarks.elo.scoring import frozen_metric_inputs

        frozen = options.pop("frozen")
        required_ratings = [method["name"] for method in frozen["methods"]]
        metrics, runtime = frozen_metric_inputs(
            cfg.resolved_task.spec.protocol.scoring.metrics,
            ratings=frozen["ratings"],
            methods=frozen["methods"],
        )
        options.update(
            opponent_panel=frozen["panel"],
            metrics=metrics,
            metric_runtime=runtime,
            sampling_metadata={
                "sampling_mode": "fixed",
                "intervals": "conditional on frozen references",
                **interval_settings,
            },
        )
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
