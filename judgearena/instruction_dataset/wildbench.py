"""Pinned WildBench V2 dataset and official baseline-output adapters."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.dataset_revisions import hf_revision

WILDBENCH_REPO_ID = "allenai/WildBench"
WILDBENCH_VARIANT = "v2"
WILDBENCH_MODEL_OUTPUTS_REPO_ID = "allenai/WildBench-V2-Model-Outputs"

OFFICIAL_WILDBENCH_BASELINES = (
    "gpt-4-turbo-2024-04-09",
    "claude-3-haiku-20240307",
    "Llama-2-70b-chat-hf",
)

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

WILDBENCH_TASK_WEIGHTS = {
    "Creative Tasks": 0.5,
    "Planning & Reasoning": 1.25,
    "Math & Data Analysis": 1.0,
    "Information/Advice seeking": 0.75,
    "Coding & Debugging": 1.25,
}


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
        raise ValueError(
            f"WildBench row {session_id!r} must end with a user query."
        )
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


def normalize_wildbench(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the official V2 parquet into JudgeArena's instruction frame."""
    required = {
        "session_id",
        "conversation_input",
        "checklist",
        "primary_tag",
        "secondary_tags",
    }
    missing = required.difference(raw_df.columns)
    if missing:
        raise ValueError(f"WildBench dataset is missing columns: {sorted(missing)}")

    rows = []
    for raw in raw_df.to_dict(orient="records"):
        session_id = str(raw["session_id"])
        messages = _normalize_messages(raw["conversation_input"], session_id)
        checklist = [str(item) for item in _as_list(raw["checklist"])]
        rows.append(
            {
                "instruction_index": session_id,
                "instruction": messages[-1]["content"],
                "conversation_input": messages,
                "history": _history_text(messages),
                "checklist": checklist,
                "primary_tag": str(raw["primary_tag"]),
                "secondary_tags": [
                    str(tag) for tag in _as_list(raw["secondary_tags"])
                ],
                "task_categories": _task_categories(
                    raw["primary_tag"], raw["secondary_tags"]
                ),
            }
        )

    normalized = pd.DataFrame(rows)
    if normalized["instruction_index"].duplicated().any():
        raise ValueError("WildBench session_id values must be unique.")
    return normalized


def load_wildbench() -> pd.DataFrame:
    snapshot_root = snapshot_download(
        repo_id=WILDBENCH_REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{WILDBENCH_VARIANT}/*.parquet",
        force_download=False,
        revision=hf_revision(WILDBENCH_REPO_ID),
    )
    parquet_files = sorted(
        (Path(snapshot_root) / WILDBENCH_VARIANT).glob("*.parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(
            f"No WildBench V2 parquet files found under {snapshot_root}."
        )
    raw_df = pd.concat(
        [pd.read_parquet(path) for path in parquet_files], ignore_index=True
    )
    return normalize_wildbench(raw_df)


def _extract_output(value: object) -> str:
    if isinstance(value, str):
        return value
    values = _as_list(value)
    return str(values[0]) if values else ""


def load_official_wildbench_baseline(model_name: str) -> pd.DataFrame | None:
    """Load a released reference completion set, or return None for custom models."""
    if model_name not in OFFICIAL_WILDBENCH_BASELINES:
        return None
    snapshot_root = snapshot_download(
        repo_id=WILDBENCH_MODEL_OUTPUTS_REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{model_name}/*.parquet",
        force_download=False,
        revision=hf_revision(WILDBENCH_MODEL_OUTPUTS_REPO_ID),
    )
    parquet_files = sorted((Path(snapshot_root) / model_name).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No official WildBench outputs found for {model_name!r}."
        )
    raw_df = pd.concat(
        [pd.read_parquet(path) for path in parquet_files], ignore_index=True
    )
    if not {"session_id", "output"}.issubset(raw_df.columns):
        raise ValueError(
            f"Official WildBench outputs for {model_name!r} have an invalid schema."
        )
    outputs = pd.DataFrame(
        {
            "instruction_index": raw_df["session_id"].astype(str),
            "completion": raw_df["output"].apply(_extract_output),
        }
    )
    return outputs.drop_duplicates("instruction_index").reset_index(drop=True)
