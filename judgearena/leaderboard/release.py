"""Write and verify immutable leaderboard releases."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa

from judgearena.artifacts import to_jsonable
from judgearena.artifacts.evaluation import file_checksum as checksum
from judgearena.artifacts.evaluation import verify_files
from judgearena.artifacts.run import prepare_run_directory, write_annotations
from judgearena.battles import hard_preferences
from judgearena.benchmarks.meta_eval.annotate import prepare_reference_comparisons
from judgearena.benchmarks.pairwise.graph import validate_graph_task
from judgearena.leaderboard.prepare import (
    judge_comparisons,
    reference_intervals,
    reference_pool,
)
from judgearena.leaderboard.scoring import score_metrics
from judgearena.leaderboard.settings import rating_metrics


def restore_nested_columns(frame):
    """Restore Python lists/dicts after pandas reads nested Parquet input columns."""
    frame = frame.copy()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    for field in table.schema:
        if pa.types.is_nested(field.type):
            frame[field.name] = pd.Series(
                table[field.name].to_pylist(), index=frame.index
            )
    return frame


def load_release(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema_version"] != 2:
        raise ValueError("Unsupported leaderboard schema version")
    verify_files(path, manifest["files"])
    ratings = (
        pd.read_csv(path / "ratings.csv")
        if "ratings.csv" in manifest["files"]
        else pd.DataFrame(columns=["lang", "method", "model", "rating"])
    )
    return manifest, ratings


@contextmanager
def release_directory(cfg, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete releases; failed builds leave reusable inference caches.
    with TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as staging:
        output = Path(staging) / "release"
        output.mkdir()
        prepare_run_directory(cfg, output / "execution", attach_log=False)
        yield output
        output.rename(destination)


def write_manifest(cfg, output, **metadata):
    manifest = dict(
        schema_version=2,
        **metadata,
        metrics=[
            m.model_dump(mode="json")
            for m in cfg.resolved_task.spec.protocol.scoring.metrics
        ],
        files={
            str(p.relative_to(output)): checksum(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def create_completion_release(cfg, data, output, *, version):
    """Freeze selected instructions and, for pairwise runs, baseline answers."""
    from judgearena.benchmarks.pairwise.baselines import resolve_baseline_plan
    from judgearena.benchmarks.pairwise.comparisons import (
        preload_model_outputs,
        prepare_answers,
        select_instructions,
    )

    if rating_metrics(cfg):
        raise ValueError(
            "Frozen ratings require a reference graph, not baseline answers"
        )
    runner = cfg.resolved_task.spec.protocol.runner
    if runner not in {"pairwise", "single_answer"}:
        raise ValueError("Completion releases require a pairwise or single_answer task")
    instructions = select_instructions(cfg, data.instructions)
    if instructions.empty:
        raise ValueError("A release requires at least one instruction")
    plan = None
    if runner == "single_answer":
        from judgearena.benchmarks.evaluation import prepare_independent_judge

        prepare_independent_judge(cfg, data, instructions)
    else:
        plan = resolve_baseline_plan(
            task_id=cfg.task,
            task=cfg.resolved_task,
            runtime_baseline=cfg.model.baseline,
            instructions=instructions,
        )
        data = preload_model_outputs(data, plan.unique_models)
    with release_directory(cfg, output) as stage:
        instructions.to_parquet(stage / "instructions.parquet")
        if plan is not None:
            outputs = []
            for model in plan.unique_models:
                ids = plan.baseline_by_index.index[plan.baseline_by_index == model]
                answers = prepare_answers(
                    cfg, data, instructions.loc[ids], model, role="B"
                )
                if answers.completion.isna().any():
                    raise ValueError(f"Missing baseline answers for {model}")
                outputs.append(answers.rename_axis("instruction_index").reset_index())
            pd.concat(outputs, ignore_index=True).to_parquet(
                stage / "model_outputs.parquet", index=False
            )
            plan.baseline_by_index.rename("model").rename_axis(
                "instruction_index"
            ).reset_index().to_parquet(stage / "baselines.parquet", index=False)
        (stage / "results.json").write_text("{}\n")
        return write_manifest(
            cfg,
            stage,
            version=version,
            input="baseline" if plan is not None else "independent",
            labels="judge",
            model_names=plan.unique_models if plan is not None else [],
            rating_metrics=[],
        )


def create_release(
    cfg,
    battles,
    panel,
    output,
    *,
    version,
    labels="human",
    heldout=None,
    preparation=None,
):
    """Freeze a curated graph and panel, using human labels or the configured judge.

    Frames use the adapter comparison columns plus lang/split_group. The panel
    also identifies opponent_model/opponent_slot. Curation decides the sample;
    publication never silently prunes unsupported models or changes the panel.
    """
    validate_graph_task(cfg.resolved_task.spec.protocol)
    if labels not in {"human", "judge"}:
        raise ValueError("labels must be human or judge")
    if not version or battles.empty or panel.empty:
        raise ValueError("A release requires a version, reference battles and a panel")
    required = {
        "battle_id",
        "instruction",
        "model_a",
        "model_b",
        "completion_a",
        "completion_b",
        "lang",
        "split_group",
    }
    for name, frame in (("references", battles), ("panel", panel)):
        missing = required - set(frame)
        if missing or frame[list(required)].isna().any().any():
            raise ValueError(f"Invalid {name} columns: {sorted(missing)}")
        if frame.battle_id.duplicated().any():
            raise ValueError(f"Duplicate battle IDs in {name}")
    if not panel.opponent_slot.isin(["a", "b"]).all():
        raise ValueError("Panel opponent_slot must be a or b")
    for row in panel.itertuples():
        if row.opponent_model != getattr(row, f"model_{row.opponent_slot}"):
            raise ValueError("Panel opponent does not match its selected slot")
    if labels == "human" and not battles.reference_pref.isin([0, 0.5, 1]).all():
        raise ValueError("Human releases require complete reference_pref labels")
    cfg = cfg.model_copy(deep=True)
    if cfg.elo is not None:
        cfg.elo = cfg.elo.resolve(cfg.resolved_task.spec.protocol.scoring)
    if cfg.preference_settings.calibrate_temperature or (
        cfg.elo is not None and cfg.elo.calibrate_temperature
    ):
        raise ValueError(
            "Set a fixed soft_elo_temperature or protocol.judge.preference_temperature before creating the release"
        )
    configured = rating_metrics(cfg)
    with release_directory(cfg, output) as output:
        if labels == "human":
            outcomes = battles.assign(pref=battles.reference_pref)
        else:
            annotations, outcomes, _ = judge_comparisons(
                cfg,
                prepare_reference_comparisons(battles),
                preference_temperature=cfg.preference_temperature,
            )
            write_annotations(annotations, output / "annotations.parquet")
        counts = outcomes.groupby("lang").pref.agg(total="size", valid="count")
        counts["missing"] = counts.total - counts.valid
        counts = counts.reindex(
            sorted(set(outcomes.lang) | set(panel.lang)), fill_value=0
        )
        valid_outcomes = outcomes.dropna(subset=["pref"])
        records = []
        for lang, frame in outcomes.groupby("lang", sort=True):
            expected_models = set(frame.model_a) | set(frame.model_b)
            frame = frame.dropna(subset=["pref"])
            names = set(frame.model_a) | set(frame.model_b)
            if frame.empty or names != expected_models:
                raise ValueError(
                    f"{lang}: insufficient valid judgments for reference models"
                )
            for method, metric in configured:
                baseline = metric.baseline_model
                bootstraps = metric.n_bootstraps
                scoring = frame.assign(
                    reference_pref=frame.pref
                    if metric.soft
                    else hard_preferences(frame.pref)
                )
                connected = reference_pool(scoring, 1)
                if len(connected) != len(scoring) or baseline not in names:
                    raise ValueError(
                        f"{lang}/{method}: reference graph must be connected with baseline present"
                    )
                values = replace(metric, n_bootstraps=0).calculate(
                    frame[["model_a", "model_b", "pref"]].assign(
                        pref_hard=hard_preferences(frame.pref)
                    )
                )["ratings"]
                distribution = (
                    reference_intervals(
                        scoring,
                        baseline=baseline,
                        bootstraps=bootstraps,
                        seed=cfg.run.seed,
                    )
                    if bootstraps
                    else None
                )
                records.extend(
                    dict(
                        lang=lang,
                        method=method,
                        model=model,
                        rating=value,
                        **(
                            {
                                "ci_low": float(distribution[model].quantile(0.025)),
                                "ci_high": float(distribution[model].quantile(0.975)),
                                "valid_bootstraps": len(distribution),
                            }
                            if distribution is not None
                            else {}
                        ),
                    )
                    for model, value in values.items()
                )
        ratings = pd.DataFrame(
            records,
            columns=[
                "lang",
                "method",
                "model",
                "rating",
                "ci_low",
                "ci_high",
                "valid_bootstraps",
            ],
        )
        model_names = sorted(set(outcomes.model_a) | set(outcomes.model_b))
        observed = {
            model: score_metrics(cfg, valid_outcomes, model, human=labels == "human")
            for model in model_names
        }
        (output / "results.json").write_text(
            json.dumps(
                to_jsonable(
                    {
                        "observed_reference_metrics": observed,
                        "judgment_counts": counts.to_dict(orient="index"),
                    }
                ),
                indent=2,
            )
            + "\n"
        )
        for row in panel.itertuples():
            if (
                configured
                and ratings[
                    (ratings.lang == row.lang) & (ratings.model == row.opponent_model)
                ].empty
            ):
                raise ValueError(
                    f"Panel opponent absent from ratings: {row.lang}/{row.opponent_model}"
                )
        battles.to_parquet(output / "reference_battles.parquet", index=False)
        panel.to_parquet(output / "panel.parquet", index=False)
        write_annotations(outcomes, output / "outcomes.parquet")
        if configured:
            ratings.to_csv(output / "ratings.csv", index=False)
        if heldout is not None:
            heldout.to_parquet(output / "heldout_battles.parquet", index=False)
        lines = [
            "# Leaderboard release",
            "",
            f"Version: {version}. Labels: {labels}.",
            "",
            "| Language | Reference comparisons | Valid judgments | Missing judgments | Panel comparisons |",
            "|---|---:|---:|---:|---:|",
        ]
        for lang in sorted(panel.lang.unique()):
            lines.append(
                f"| {lang} | {(battles.lang == lang).sum()} | {counts.loc[lang, 'valid']} | {counts.loc[lang, 'missing']} | {(panel.lang == lang).sum()} |"
            )
        if configured:
            lines += [
                "",
                "Reference ratings and grouped bootstrap intervals are in ratings.csv.",
                "The baseline is fixed at 1000; its zero-width interval reflects anchoring.",
            ]
        (output / "report.md").write_text("\n".join(lines) + "\n")
        return write_manifest(
            cfg,
            output,
            version=version,
            input="graph",
            labels=labels,
            model_names=model_names,
            rating_metrics=[
                {
                    "name": name,
                    "soft": metric.soft,
                    "baseline_model": metric.baseline_model,
                    "n_bootstraps": metric.n_bootstraps,
                }
                for name, metric in configured
            ],
            preparation=preparation,
            bootstrap_seed=cfg.run.seed,
            languages=sorted(panel.lang.unique().tolist()),
        )


def release_inputs(release, manifest, ratings, model_outputs=None):
    """Restore frozen data and preparation inputs for the ordinary task runner."""
    from judgearena.benchmarks.pairwise.baselines import BaselinePlan
    from judgearena.datasets.data import CompletionData

    release = Path(release)
    if model_outputs is not None:
        model_outputs = restore_nested_columns(model_outputs)
    if manifest.get("input") in {"independent", "baseline"}:
        instructions = restore_nested_columns(
            pd.read_parquet(release / "instructions.parquet")
        )
        options = {}
        if manifest["input"] == "baseline":
            baselines = restore_nested_columns(
                pd.read_parquet(release / "model_outputs.parquet")
            )
            model_outputs = pd.concat(
                [baselines] + ([model_outputs] if model_outputs is not None else []),
                ignore_index=True,
            )
            assignments = pd.read_parquet(release / "baselines.parquet").set_index(
                "instruction_index"
            )["model"]
            options["baseline_plan"] = BaselinePlan.per_row(assignments)
    else:
        panel = restore_nested_columns(
            pd.read_parquet(release / "panel.parquet")
        ).set_index("battle_id", drop=False)
        instructions = panel.drop(
            columns=[c for c in panel if c.endswith(("_a", "_b"))]
            + ["reference_pref", "opponent_model", "opponent_slot"],
            errors="ignore",
        )
        instructions["instruction_id"] = instructions.index
        options = {
            "frozen": {
                "panel": panel,
                "ratings": ratings,
                "methods": manifest["rating_metrics"],
            }
        }
    return CompletionData(instructions, model_outputs), options
