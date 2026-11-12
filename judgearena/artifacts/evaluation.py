"""Portable results and artifacts shared by evaluation runners."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from judgearena.artifacts.metadata import to_jsonable


def file_checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metric_data_checksum(path):
    """Hash ordered rows independently of Parquet encoding and column order."""
    path = Path(path)
    if path.suffix != ".parquet":
        return file_checksum(path)
    frame = pd.read_parquet(path)
    return hashlib.sha256(
        frame.reindex(sorted(frame.columns), axis=1)
        .to_json(orient="split", index=False)
        .encode()
    ).hexdigest()


def finish_evaluation(output_dir, report, *, model, artifacts):
    """Save metrics and named artifact paths without imposing runner filenames."""
    output = Path(output_dir).resolve()
    paths = {"config": output / "config.yaml", **artifacts}
    if not {"report", "metric_data"} <= paths.keys():
        raise ValueError("Evaluation requires report and metric_data artifacts")
    paths = {
        role: str(Path(path).resolve().relative_to(output))
        for role, path in paths.items()
    }
    payload = {
        "schema_version": 1,
        "model": model,
        "metrics": report.metrics,
        "artifacts": paths,
        "metric_data_sha256": metric_data_checksum(output / paths["metric_data"]),
        "files": {
            str(path.relative_to(output)): file_checksum(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path.name not in {"evaluation.json", "release.json"}
            and path.suffix != ".log"
        },
    }
    target = output / "evaluation.json"
    target.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    return target
