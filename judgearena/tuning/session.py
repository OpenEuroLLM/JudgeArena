"""Explicit tuning sessions and collection of shared NePS state."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from judgearena.artifacts import prepare_run_directory, safe_filename
from judgearena.config import RunConfig


def _config_hash(cfg: RunConfig) -> str:
    payload = cfg.model_dump(mode="json")
    payload["run"] = {"seed": cfg.run.seed}
    tuning = payload["tune_judge"]
    tuning.pop("run_dir")
    tuning.pop("search_only")
    tuning["neps"] = {
        key: value
        for key, value in tuning["neps"].items()
        if not key.startswith("worker_")
    }
    if cfg.judge.prompt is not None:
        payload["prompt_contents"] = [
            path.read_text()
            for path in (cfg.judge.prompt.system_file, cfg.judge.prompt.user_file)
        ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_session(cfg: RunConfig) -> Path:
    """Create a new run or verify an explicitly selected existing run."""
    tuning = cfg.tune_judge
    tune_dir = tuning.run_dir or (
        Path(cfg.run.result_folder)
        / f"{safe_filename(cfg.task)}-{datetime.now(UTC):%Y%m%d_%H%M%S_%f}"
    )
    tune_dir = tune_dir.resolve()
    metadata_path = tune_dir / "tuning-metadata.json"
    fingerprint = _config_hash(cfg)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["config_hash"] != fingerprint:
            raise ValueError(f"Incompatible configuration for tuning run {tune_dir}")
        if not tuning.search_only:
            prepare_run_directory(cfg, tune_dir)
    elif tuning.search_only:
        raise ValueError("Start the primary process before joining its run_dir")
    else:
        prepare_run_directory(cfg, tune_dir)
        versions = {"neural-pipeline-search": version("neural-pipeline-search")}
        try:
            versions["vllm"] = version("vllm")
        except PackageNotFoundError:
            pass
        metadata = {"config_hash": fingerprint, "versions": versions}
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2))
        temporary.replace(metadata_path)
    if tuning.search_only:
        if not (tune_dir / "neps" / "pipeline_space.pkl").exists():
            raise ValueError("The primary process has not initialized NePS yet")
    return tune_dir


def collect_trials(root: Path, *, wait: bool) -> pd.DataFrame:
    """Wait for active evaluations and read all workers' trial records."""
    from neps.state import NePSState

    state = NePSState.create_or_load(root, load_only=True)
    while True:
        trials = state.lock_and_read_trials()
        active = [
            t for t in trials.values() if t.metadata.state in {"pending", "evaluating"}
        ]
        if not wait or not active:
            break
        time.sleep(1)
    records = []
    for trial in trials.values():
        if trial.metadata.state in {"success", "failed", "crashed"}:
            records.append(
                {
                    "neps_trial_id": trial.id,
                    "status": "failed",
                    **trial.report.extra,
                }
            )
    return pd.DataFrame(records)
