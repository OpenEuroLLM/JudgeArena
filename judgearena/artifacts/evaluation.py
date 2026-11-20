"""Portable results and artifacts shared by evaluation runners."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from judgearena.artifacts.metadata import to_jsonable


def file_checksum(path):
    """Detect byte-level changes to a saved artifact."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metric_data_checksum(path):
    """Identify equivalent scoring rows despite Parquet encoding or column order.

    Row order remains significant; this identifies replayed data, not shuffled data.
    """
    path = Path(path)
    if path.suffix != ".parquet":
        return file_checksum(path)
    table = pq.read_table(path)
    table = table.select(sorted(table.column_names))
    # Native values retain float precision and nested data across Parquet encodings.
    payload = {"columns": table.column_names, "rows": table.to_pylist()}
    return hashlib.sha256(
        json.dumps(to_jsonable(payload), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def finish_evaluation(output_dir, report, *, model, artifacts):
    """Write evaluation.json so consumers need not know each runner's filenames.

    Call after saving the run's artifacts; checksums capture their final contents.
    """
    output = Path(output_dir).resolve()
    paths = {"config": output / "config.yaml", **artifacts}
    if not {"report", "metric_data"} <= paths.keys():
        raise ValueError("Evaluation requires report and metric_data artifacts")
    # Relative paths keep the manifest usable when the run directory is copied.
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
        # Include replay dependencies too, but exclude manifests and changing logs.
        "files": {
            str(path.relative_to(output)): file_checksum(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
            and path not in {output / "evaluation.json", output / "release.json"}
            and path.suffix != ".log"
        },
    }
    target = output / "evaluation.json"
    target.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")
    return target
