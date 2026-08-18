"""WildBench V2 dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.tasks.schema import HuggingFaceDatasetSource, ResolvedTaskSpec

TASK_GROUPS = {
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


def _require_adapter(task: ResolvedTaskSpec) -> None:
    if task.spec.dataset.adapter != "wildbench":
        raise ValueError(f"Task {task.task!r} does not use the WildBench adapter.")


def _source(task: ResolvedTaskSpec, name: str) -> HuggingFaceDatasetSource:
    source = task.spec.dataset.sources.get(name)
    if not isinstance(source, HuggingFaceDatasetSource):
        raise ValueError(
            f"Task {task.task!r} must define {name!r} as a Hugging Face source."
        )
    return source


def _download_source(
    task: ResolvedTaskSpec, name: str, local_tables_path: Path
) -> Path:
    source = _source(task, name)
    target = (
        local_tables_path
        / "wildbench"
        / source.repo_id.replace("/", "--")
        / source.revision
    )
    snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        revision=source.revision,
        allow_patterns=list(source.allow_patterns) or None,
        local_dir=target,
        force_download=False,
    )
    return target


def download_task_sources(task: ResolvedTaskSpec, local_tables_path: Path) -> None:
    """Download the pinned sources declared by a WildBench task."""
    _require_adapter(task)
    for name in task.spec.dataset.sources:
        _download_source(task, name, local_tables_path)


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        return converted if isinstance(converted, list) else [converted]
    return []


def _normalize_conversation(value: object, session_id: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw_message in _as_list(value):
        if not isinstance(raw_message, dict):
            raise ValueError(
                f"WildBench row {session_id!r} contains a malformed conversation."
            )
        role = str(raw_message.get("role", "")).lower()
        content = raw_message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(
                f"WildBench row {session_id!r} has an invalid role/content pair."
            )
        messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        raise ValueError(f"WildBench row {session_id!r} must end with a user turn.")
    return messages


def _history(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"{message['role'].upper()}: {message['content']}\n\n"
        for message in messages[:-1]
    )


def _task_categories(primary_tag: object, secondary_tags: object) -> list[str]:
    tags = [str(primary_tag), *[str(tag) for tag in _as_list(secondary_tags)]]
    unknown = sorted({tag for tag in tags if tag not in TASK_GROUPS})
    if unknown:
        raise ValueError(f"Unknown WildBench task tag(s): {unknown}")
    return sorted({TASK_GROUPS[tag] for tag in tags})


def normalize_wildbench(
    raw_df: pd.DataFrame,
    *,
    id_field: str = "session_id",
    conversation_field: str = "conversation_input",
    category_field: str = "primary_tag",
) -> pd.DataFrame:
    """Normalize the official V2 rows without losing conversation structure."""
    required = {
        id_field,
        conversation_field,
        "checklist",
        category_field,
        "secondary_tags",
    }
    missing = sorted(required.difference(raw_df.columns))
    if missing:
        raise ValueError(f"WildBench dataset is missing columns: {missing}")

    rows: list[dict[str, object]] = []
    for raw in raw_df.to_dict(orient="records"):
        session_id = str(raw[id_field])
        messages = _normalize_conversation(raw[conversation_field], session_id)
        primary_tag = str(raw[category_field])
        secondary_tags = [str(tag) for tag in _as_list(raw["secondary_tags"])]
        rows.append(
            {
                "instruction_index": session_id,
                "instruction": messages[-1]["content"],
                "conversation_input": messages,
                "history": _history(messages),
                "checklist": [str(item) for item in _as_list(raw["checklist"])],
                "category": primary_tag,
                "primary_tag": primary_tag,
                "secondary_tags": secondary_tags,
                "task_categories": _task_categories(primary_tag, secondary_tags),
            }
        )

    normalized = pd.DataFrame(rows)
    if normalized["instruction_index"].duplicated().any():
        raise ValueError("WildBench session_id values must be unique.")
    return normalized


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load and normalize the pinned WildBench examples."""
    _require_adapter(task)
    root = _download_source(task, "examples", local_tables_path)
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No WildBench parquet files found under {root}.")
    raw_df = pd.concat(
        [pd.read_parquet(path) for path in parquet_files], ignore_index=True
    )
    fields = task.spec.dataset.fields
    return normalize_wildbench(
        raw_df,
        id_field=fields.id,
        conversation_field=fields.instruction,
        category_field=fields.category or "primary_tag",
    )


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame | None:
    """Load the pinned model outputs declared by a WB-Reward task."""
    _require_adapter(task)
    if "official_outputs" not in task.spec.dataset.sources:
        return None
    root = _download_source(task, "official_outputs", local_tables_path)
    frames: list[pd.DataFrame] = []
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        parquet_files = sorted(model_dir.rglob("*.parquet"))
        if not parquet_files:
            continue
        raw_df = pd.concat(
            [pd.read_parquet(path) for path in parquet_files], ignore_index=True
        )
        frames.append(normalize_model_outputs(raw_df, model_dir.name))
    if not frames:
        raise FileNotFoundError(f"No official WildBench outputs found under {root}.")
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["instruction_index", "model"], keep="last"
    )


def normalize_model_outputs(raw_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Normalize one model directory from the official output dataset."""
    required = {"session_id", "output"}
    missing = sorted(required.difference(raw_df.columns))
    if missing:
        raise ValueError(
            f"WildBench outputs for {model_name!r} are missing columns: {missing}"
        )

    def first_output(value: object) -> str:
        if isinstance(value, str):
            return value
        values = _as_list(value)
        return str(values[0]) if values else ""

    return pd.DataFrame(
        {
            "instruction_index": raw_df["session_id"].astype(str),
            "model": model_name,
            "output": raw_df["output"].apply(first_output),
        }
    )
