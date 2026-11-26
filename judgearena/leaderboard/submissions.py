"""Collect compatible saved results without changing reference ratings."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from judgearena.artifacts.evaluation import (
    load_evaluation,
    metric_data_checksum,
    verify_files,
)
from judgearena.leaderboard.release import bootstrap_settings, load_release


def _check_bootstrap(record, expected):
    if expected:
        counts = record.get("n_bootstraps")
        if isinstance(counts, int):
            counts = dict.fromkeys(expected["n_bootstraps"], counts)
        if (
            record.get("bootstrap_seed") != expected["bootstrap_seed"]
            or counts != expected["n_bootstraps"]
        ):
            raise ValueError(
                "Bootstrap settings differ from the release; replay the submission"
            )


def _entries(collection, manifest):
    entries = []
    for path in sorted(Path(collection).glob("*.json")):
        entry = json.loads(path.read_text())
        if any(entry.get(key) != manifest[key] for key in ("version", "files")):
            raise ValueError("Collection contains a different release")
        _check_bootstrap(entry["bootstrap"], bootstrap_settings(manifest))
        entries.append(entry)
    return entries


def _result_metrics(result):
    metrics = deepcopy(result["metrics"])
    for metric in metrics.values():
        for groups in metric.get("groups", {}).values():
            groups.sort(key=lambda group: json.dumps(group["group"], sort_keys=True))
    return metrics


def add_submission(release, submission, collection):
    manifest, _ = load_release(Path(release))
    submission, collection = Path(submission), Path(collection)
    receipt = json.loads((submission / "release.json").read_text())
    if any(receipt.get(key) != manifest[key] for key in ("version", "files")):
        raise ValueError("Submission belongs to a different release")
    intervals = bootstrap_settings(manifest)
    _check_bootstrap(receipt, intervals)
    files = receipt.get("submission_files", {})
    if not files:
        raise ValueError("Submission has no artifact checksums; replay it first")
    verify_files(submission, files)
    if "evaluation.json" not in files:
        raise ValueError("Submission has no evaluation manifest; replay it first")
    evaluation = load_evaluation(submission / "evaluation.json")
    result = {"model": evaluation["model"], "metrics": evaluation["metrics"]}
    config_path = evaluation["artifacts"]["config"]
    outcomes_hash = metric_data_checksum(
        submission / evaluation["artifacts"]["metric_data"]
    )
    model = result["model"]
    config = yaml.safe_load((submission / config_path).read_text())
    if model != config["model"]["name"] or model in manifest.get("model_names", []):
        raise ValueError(
            "Submission model does not match its config or is a reference model"
        )
    entry = {
        "bootstrap": intervals,
        "version": manifest["version"],
        "files": manifest["files"],
        "result": result,
        "submission": str(submission.resolve()),
        "submission_files": files,
        "outcomes_sha256": outcomes_hash,
    }
    _entries(collection, manifest)
    collection.mkdir(parents=True, exist_ok=True)
    target = collection / (
        hashlib.sha256(result["model"].encode()).hexdigest() + ".json"
    )
    if target.exists():
        previous = json.loads(target.read_text())
        # Replays can have different config paths while retaining identical outcomes.
        if (
            all(previous[key] == entry[key] for key in ("version", "files"))
            and previous["result"]["model"] == result["model"]
            and _result_metrics(previous["result"]) == _result_metrics(result)
        ):
            if previous["outcomes_sha256"] == entry["outcomes_sha256"]:
                return target
        raise ValueError(
            "Conflicting submission for this model: outcomes, scores or bootstrap settings differ; use matching settings or a separate collection"
        )
    with target.open("x") as stream:
        json.dump(entry, stream, indent=2)
        stream.write("\n")
    return target


def _metric_columns(values):
    # Flatten nested metric outputs for display; keep grouped results as separate rows.
    flat = pd.json_normalize(
        {k: v for k, v in values.items() if k != "groups"}
    ).to_dict("records")[0]
    return {k: v for k, v in flat.items() if not isinstance(v, (dict, list))}


def _grouped_rows(result):
    for name, metric in result["metrics"].items():
        identity = {"model": result["model"], "method": name}
        values = _metric_columns(metric)
        if values:
            yield None, {**values, **identity}
        for field, groups in metric.get("groups", {}).items():
            for group in groups:
                yield (
                    field,
                    {
                        **_metric_columns(group["values"]),
                        **identity,
                        field: group["group"],
                    },
                )


def result_rows(result, *, overall=False, breakdown=None):
    return [
        row
        for field, row in _grouped_rows(result)
        if (not overall or field is None) and (breakdown is None or field == breakdown)
    ]


def submission_table(
    release,
    collection,
    *,
    columns=None,
    sort_by=None,
    ascending=True,
    overall=False,
    breakdown=None,
):
    manifest, references = load_release(Path(release))
    if overall and breakdown is not None:
        raise ValueError("Select overall or a breakdown, not both")
    entries = _entries(collection, manifest)
    if breakdown is not None:
        available = {
            field
            for entry in entries
            for metric in entry["result"]["metrics"].values()
            for field in metric.get("groups", {})
        }
        if not references.empty:
            available.add("lang")
        if breakdown not in available:
            raise ValueError(
                f"Unknown breakdown: {breakdown}; available: {sorted(available)}"
            )
    rows = []
    fields = {}
    for entry in entries:
        for field, row in _grouped_rows(entry["result"]):
            fields.update(dict.fromkeys(row))
            if (overall and field is not None) or (
                breakdown is not None and field != breakdown
            ):
                continue
            rows.append(row)
    table = pd.DataFrame(rows, columns=list(fields)).assign(entry_type="submission")
    if not references.empty:
        reference_rows = references.assign(entry_type="reference")
        if overall or breakdown not in (None, "lang"):
            reference_rows = reference_rows.iloc[:0]
        table = pd.concat([reference_rows, table], ignore_index=True)
    if breakdown is not None and breakdown not in table:
        table[breakdown] = pd.Series(dtype=object)
    order = [column for column in ("lang", "method", "model") if column in table]
    if sort_by is not None:
        if sort_by not in table:
            raise ValueError(f"Unknown table column: {sort_by}")
        table = table.sort_values(sort_by, ascending=ascending, kind="stable")
    elif order:
        table = table.sort_values(order, kind="stable")
    leading = [
        column
        for column in ("lang", "method", "entry_type", "model")
        if column in table
    ]
    if columns is None:
        columns = leading + [column for column in table if column not in leading]
    if breakdown is not None and breakdown not in columns:
        columns = [breakdown, *columns]
    unknown = set(columns) - set(table.columns)
    if unknown:
        raise ValueError(f"Unknown table columns: {sorted(unknown)}")
    return table.loc[:, columns].reset_index(drop=True)
