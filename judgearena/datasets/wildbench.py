"""Dataset adapter for YAML-defined WildBench V2 tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.tasks.schema import HuggingFaceDatasetSource, ResolvedTaskSpec

WILDBENCH_TASK_GROUPS = {
    "Information seeking": "Information/Advice seeking",
    "Creative Writing": "Creative Tasks",
    "Coding & Debugging": "Coding & Debugging",
    "Reasoning": "Planning & Reasoning",
    "Editing": "Creative Tasks",
    "Math": "Math & Data Analysis",
    "Planning": "Planning & Reasoning",
    "Brainstorming": "Creative Tasks",
    "Role playing": "Creative Tasks",
    "Advice seeking": "Information/Advice seeking",
    "Data Analysis": "Math & Data Analysis",
    "Others": "Creative Tasks",
}


def _require_wildbench(task: ResolvedTaskSpec) -> None:
    if task.spec.dataset.adapter != "wildbench":
        raise ValueError(
            f"Task {task.task!r} uses dataset adapter "
            f"{task.spec.dataset.adapter!r}, not 'wildbench'."
        )


def _source_root(local_dir: Path, source: HuggingFaceDatasetSource) -> Path:
    repo_slug = source.repo_id.replace("/", "--")
    return local_dir / "wildbench" / repo_slug / source.revision


def _download_source(
    name: str,
    source: object,
    *,
    local_dir: Path,
) -> Path:
    if not isinstance(source, HuggingFaceDatasetSource):
        raise ValueError(
            f"WildBench source {name!r} must be a Hugging Face dataset source."
        )
    target = _source_root(local_dir, source)
    snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        revision=source.revision,
        allow_patterns=list(source.allow_patterns) or None,
        local_dir=target,
        force_download=False,
    )
    return target


def download_task_sources(task: ResolvedTaskSpec, local_dir: Path) -> None:
    """Download every pinned source declared by a WildBench task."""
    _require_wildbench(task)
    for name, source in task.spec.dataset.sources.items():
        _download_source(name, source, local_dir=local_dir)


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        result = tolist()
        return result if isinstance(result, list) else [result]
    return []


def _normalize_messages(value: object, session_id: str) -> list[dict[str, str]]:
    messages = []
    for raw in _as_list(value):
        if not isinstance(raw, dict):
            raise ValueError(
                f"WildBench row {session_id!r} contains a malformed conversation turn."
            )
        role = str(raw.get("role", "")).lower()
        content = raw.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(
                f"WildBench row {session_id!r} has an invalid role/content pair."
            )
        messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"WildBench row {session_id!r} must end with a user query.")
    return messages


def _history_text(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"{message['role'].upper()}: {message['content']}\n\n"
        for message in messages[:-1]
    )


def _task_categories(primary_tag: object, secondary_tags: object) -> list[str]:
    raw_tags = [str(primary_tag), *[str(tag) for tag in _as_list(secondary_tags)]]
    unknown = sorted({tag for tag in raw_tags if tag not in WILDBENCH_TASK_GROUPS})
    if unknown:
        raise ValueError(f"Unknown WildBench task tag(s): {unknown}")
    return sorted({WILDBENCH_TASK_GROUPS[tag] for tag in raw_tags})


def normalize_wildbench(
    raw_df: pd.DataFrame,
    *,
    id_field: str = "session_id",
    conversation_field: str = "conversation_input",
    category_field: str = "primary_tag",
) -> pd.DataFrame:
    """Normalize the official V2 parquet into JudgeArena's canonical frame."""
    required = {
        id_field,
        conversation_field,
        "checklist",
        category_field,
        "secondary_tags",
    }
    missing = required.difference(raw_df.columns)
    if missing:
        raise ValueError(f"WildBench dataset is missing columns: {sorted(missing)}")

    rows = []
    for raw in raw_df.to_dict(orient="records"):
        session_id = str(raw[id_field])
        messages = _normalize_messages(raw[conversation_field], session_id)
        checklist = [str(item) for item in _as_list(raw["checklist"])]
        rows.append(
            {
                "instruction_index": session_id,
                "instruction": messages[-1]["content"],
                "conversation_input": messages,
                "history": _history_text(messages),
                "checklist": checklist,
                "primary_tag": str(raw[category_field]),
                "secondary_tags": [str(tag) for tag in _as_list(raw["secondary_tags"])],
                "task_categories": _task_categories(
                    raw[category_field], raw["secondary_tags"]
                ),
            }
        )

    normalized = pd.DataFrame(rows)
    if normalized["instruction_index"].duplicated().any():
        raise ValueError("WildBench session_id values must be unique.")
    return normalized


def _read_parquet_files(root: Path, source_name: str) -> pd.DataFrame:
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for WildBench source {source_name!r} under {root}."
        )
    return pd.concat(
        [pd.read_parquet(path) for path in parquet_files], ignore_index=True
    )


def load_task_instructions(task: ResolvedTaskSpec, local_dir: Path) -> pd.DataFrame:
    """Load and normalize the examples source selected by a task."""
    _require_wildbench(task)
    source = task.spec.dataset.sources.get("examples")
    root = _download_source("examples", source, local_dir=local_dir)
    raw_df = _read_parquet_files(root, "examples")
    fields = task.spec.dataset.fields
    return normalize_wildbench(
        raw_df,
        id_field=fields.id,
        conversation_field=fields.instruction,
        category_field=fields.category or "primary_tag",
    )


def _extract_output(value: object) -> str:
    if isinstance(value, str):
        return value
    values = _as_list(value)
    return str(values[0]) if values else ""


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_dir: Path
) -> pd.DataFrame | None:
    """Load the task's pinned official WB-Reward model outputs."""
    _require_wildbench(task)
    source = task.spec.dataset.sources.get("official_outputs")
    if source is None:
        return None
    root = _download_source("official_outputs", source, local_dir=local_dir)

    output_frames = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        parquet_files = sorted(model_dir.rglob("*.parquet"))
        if not parquet_files:
            continue
        raw_df = pd.concat(
            [pd.read_parquet(path) for path in parquet_files], ignore_index=True
        )
        if not {"session_id", "output"}.issubset(raw_df.columns):
            raise ValueError(
                f"Official WildBench outputs for {model_dir.name!r} have an "
                "invalid schema."
            )
        output_frames.append(
            pd.DataFrame(
                {
                    "instruction_index": raw_df["session_id"].astype(str),
                    "model": model_dir.name,
                    "output": raw_df["output"].apply(_extract_output),
                }
            )
        )

    if not output_frames:
        raise FileNotFoundError(
            f"No official WildBench model outputs found under {root}."
        )
    return pd.concat(output_frames, ignore_index=True).drop_duplicates(
        ["instruction_index", "model"], keep="last"
    )
