"""Dataset adapter for YAML-defined m-ArenaHard task families."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import HuggingFaceDatasetSource, ResolvedTaskSpec


def split_m_arena_hard_dataset(dataset: str) -> tuple[str, str | None] | None:
    """Return the YAML definition ID and optional selected language view."""
    task = get_packaged_task(dataset)
    if task is None or task.spec.dataset.adapter != "m_arena_hard":
        return None
    selection = task.selection.name if task.selection is not None else None
    return task.definition_task, selection


def _source(task: ResolvedTaskSpec, name: str = "examples") -> HuggingFaceDatasetSource:
    source = task.spec.dataset.sources.get(name)
    if not isinstance(source, HuggingFaceDatasetSource):
        raise ValueError(
            f"Task {task.task!r} must define a Hugging Face dataset source "
            f"named {name!r}."
        )
    return source


def _source_local_dir(source: HuggingFaceDatasetSource, root: Path) -> Path:
    """Keep raw source snapshots separate from normalized JudgeArena tables."""
    return root / "_sources" / source.repo_id.replace("/", "--")


def download_task_sources(task: ResolvedTaskSpec, local_tables_path: Path) -> None:
    """Download the pinned examples and optional packaged model outputs."""
    if task.spec.dataset.adapter != "m_arena_hard":
        raise ValueError(f"Task {task.task!r} does not use the m-ArenaHard adapter.")

    for name in task.spec.dataset.sources:
        _download_source(task, name, local_tables_path)


def _download_source(
    task: ResolvedTaskSpec,
    name: str,
    local_tables_path: Path,
    *,
    allow_patterns: list[str] | None = None,
) -> None:
    declared_source = _source(task, name)
    local_dir = (
        _source_local_dir(declared_source, local_tables_path)
        if name == "examples"
        else local_tables_path
    )
    snapshot_download(
        repo_id=declared_source.repo_id,
        repo_type="dataset",
        revision=declared_source.revision,
        allow_patterns=(
            allow_patterns
            if allow_patterns is not None
            else list(declared_source.allow_patterns) or None
        ),
        local_dir=local_dir,
        force_download=False,
    )


def _selected_languages(task: ResolvedTaskSpec) -> tuple[str, ...]:
    variants = task.spec.variants
    if variants is None or variants.selector != "language":
        raise ValueError(f"Task {task.task!r} must define language suffix variants.")
    return task.selection.values if task.selection is not None else variants.values


def _load_source_frames(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    _download_source(task, "examples", local_tables_path)
    source = _source(task)
    source_root = _source_local_dir(source, local_tables_path)
    selected = set(_selected_languages(task))

    frames: list[pd.DataFrame] = []
    for parquet_path in sorted(source_root.rglob("*.parquet")):
        language = parquet_path.parent.name
        if language not in selected:
            continue
        frame = pd.read_parquet(parquet_path)
        frame["lang"] = language
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No m-ArenaHard parquet files for {sorted(selected)} under {source_root}."
        )

    df = pd.concat(frames, ignore_index=True)
    fields = task.spec.dataset.fields
    missing = sorted({fields.id, fields.instruction} - set(df.columns))
    if missing:
        raise ValueError(
            f"Task {task.task!r} is missing declared dataset fields: {missing}."
        )
    df[fields.id] = df[fields.id].astype(str) + "-" + df["lang"]
    if df[fields.id].duplicated().any():
        duplicates = df.loc[df[fields.id].duplicated(), fields.id].head().tolist()
        raise ValueError(
            f"Task {task.task!r} contains duplicate instruction IDs: {duplicates}."
        )
    return df


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load and normalize the languages selected by the task invocation."""
    df = _load_source_frames(task, local_tables_path)
    fields = task.spec.dataset.fields
    return (
        df.rename(
            columns={
                fields.id: "instruction_index",
                fields.instruction: "instruction",
            }
        )
        .sort_values("instruction_index")
        .reset_index(drop=True)
    )


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame | None:
    """Load pre-generated outputs when the selected task view ships them."""
    if task.selection is None:
        return None
    path = local_tables_path / "model_outputs" / f"{task.task}.csv.zip"
    _download_source(
        task,
        "outputs",
        local_tables_path,
        allow_patterns=[path.relative_to(local_tables_path).as_posix()],
    )
    return pd.read_csv(path) if path.exists() else None


def load_m_arenahard(
    local_path: Path,
    version: str,
    language: str | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper returning the former source-shaped DataFrame."""
    task_id = version if language is None else f"{version}-{language}"
    task = get_packaged_task(task_id)
    if task is None or task.spec.dataset.adapter != "m_arena_hard":
        raise ValueError(f"Unsupported m-ArenaHard task: {task_id!r}.")
    return _load_source_frames(task, local_path)


if __name__ == "__main__":
    from judgearena.paths import data_root

    load_m_arenahard(local_path=data_root, version="m-arena-hard-v0.1", language="EU")
