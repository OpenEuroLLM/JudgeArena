from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download

ARENA_HARD_HF_REPO_ID = "lmarena-ai/arena-hard-auto"

# Mirrors upstream's `JUDGE_SETTINGS` baseline assignment in
# `arena-hard-auto/utils/judge_utils.py`: v0.1 has a single flat baseline,
# v2.0 routes per question category. `is_arena_hard_dataset` and the
# dispatcher in `generate_and_evaluate.py` key off this map.
ARENA_HARD_BASELINES: dict[str, str | Mapping[str, str]] = {
    "arena-hard-v0.1": "gpt-4-0314",
    "arena-hard-v2.0": {
        "hard_prompt": "o3-mini-2025-01-31",
        "coding": "o3-mini-2025-01-31",
        "math": "o3-mini-2025-01-31",
        "creative_writing": "gemini-2.0-flash-001",
    },
}

# Dataset name -> upstream HF `data/<variant>/` directory. Kept private so the
# public API of this module is just the baseline map and helpers below.
_ARENA_HARD_HF_VARIANTS: dict[str, str] = {
    "arena-hard-v0.1": "arena-hard-v0.1",
    "arena-hard-v2.0": "arena-hard-v2.0",
}


def is_arena_hard_dataset(dataset: str) -> bool:
    return dataset in ARENA_HARD_BASELINES


def arena_hard_native_baseline(
    dataset: str,
) -> str | Mapping[str, str] | None:
    """Dataset-native baseline assignment.

    Returns a plain string for flat datasets (v0.1), a `{category: model}`
    mapping for per-category datasets (v2.0), or `None` for datasets that
    don't ship a native baseline.
    """
    return ARENA_HARD_BASELINES.get(dataset)


def normalize_official_arena_hard(
    raw_df: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if dataset not in _ARENA_HARD_HF_VARIANTS:
        raise ValueError(f"Unsupported Arena-Hard dataset: {dataset}")
    df_instructions = _build_instructions(raw_df)
    df_model_outputs = _build_model_outputs(raw_df)
    return df_instructions, df_model_outputs


def download_arena_hard(dataset: str, local_tables_path: Path) -> None:
    """Populate `{dataset}.csv` and `{dataset}.csv.zip` on disk if missing.

    Pulls the raw jsonl files directly via `snapshot_download` and reads them
    with pandas: upstream's per-row `messages[].content` oscillates between
    string and dict across answer files, so `datasets.load_dataset` can't
    materialize them into a single Arrow schema.

    Re-downloads when the instructions table is stale - currently only v2.0
    detects this, because routing by category requires the `category` column
    that older caches were written without.
    """
    if dataset not in _ARENA_HARD_HF_VARIANTS:
        return
    instructions_path = local_tables_path / "instructions" / f"{dataset}.csv"
    model_outputs_path = local_tables_path / "model_outputs" / f"{dataset}.csv.zip"
    if (
        instructions_path.exists()
        and model_outputs_path.exists()
        and _instructions_cache_is_fresh(instructions_path, dataset)
    ):
        return

    variant = _ARENA_HARD_HF_VARIANTS[dataset]
    snapshot_root = snapshot_download(
        repo_id=ARENA_HARD_HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=[
            f"data/{variant}/question.jsonl",
            f"data/{variant}/model_answer/*.jsonl",
        ],
        force_download=False,
    )
    raw_df = _read_arena_hard_jsonl_frames(
        variant_dir=Path(snapshot_root) / "data" / variant
    )
    df_instructions, df_model_outputs = normalize_official_arena_hard(
        raw_df=raw_df, dataset=dataset
    )
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    model_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    df_instructions.to_csv(instructions_path, index=False)
    if df_model_outputs is not None:
        df_model_outputs.to_csv(model_outputs_path, index=False)


def _instructions_cache_is_fresh(instructions_path: Path, dataset: str) -> bool:
    """Category-aware datasets need a `category` column; older caches lack it."""
    native = arena_hard_native_baseline(dataset)
    if not isinstance(native, Mapping):
        return True
    cached_columns = pd.read_csv(instructions_path, nrows=0).columns
    return "category" in cached_columns


def _read_arena_hard_jsonl_frames(variant_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    question_path = variant_dir / "question.jsonl"
    if question_path.exists():
        frames.append(pd.read_json(question_path, lines=True))
    answer_dir = variant_dir / "model_answer"
    if answer_dir.exists():
        for jsonl_path in sorted(answer_dir.glob("*.jsonl")):
            frames.append(pd.read_json(jsonl_path, lines=True))
    if not frames:
        raise FileNotFoundError(f"No Arena-Hard jsonl files found under {variant_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_instructions(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Question rows are the ones with a prompt; model-answer rows don't have
    # one and must not leak into the instructions table.
    if "prompt" in raw_df.columns:
        question_rows = raw_df[raw_df["prompt"].notna()].reset_index(drop=True)
    else:
        question_rows = raw_df.reset_index(drop=True)

    if len(question_rows) == 0:
        return pd.DataFrame(columns=["instruction_index", "instruction"])

    columns: dict[str, pd.Series] = {
        "instruction_index": _pick_instruction_index(question_rows),
        "instruction": _pick_instruction(question_rows),
    }
    if "category" in question_rows.columns:
        columns["category"] = question_rows["category"]
    df = pd.DataFrame(columns)
    df = df.dropna(subset=["instruction_index", "instruction"])
    df["instruction"] = df["instruction"].astype(str)
    df = df.drop_duplicates(subset=["instruction_index"])
    df = df.sort_values("instruction_index").reset_index(drop=True)
    return df


def _build_model_outputs(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    if "model" not in raw_df.columns:
        return None
    extracted_output = raw_df.apply(_extract_assistant_output, axis=1)
    instruction_index = _pick_instruction_index(raw_df)
    df = pd.DataFrame(
        {
            "instruction_index": instruction_index,
            "model": raw_df["model"],
            "output": extracted_output,
        }
    )
    df = df[df["model"].notna() & df["output"].notna()]
    df = df.dropna(subset=["instruction_index"])
    if df.empty:
        return None
    df["instruction_index"] = df["instruction_index"].astype(str)
    df["model"] = df["model"].astype(str)
    df["output"] = df["output"].astype(str)
    return df.reset_index(drop=True)


def _extract_assistant_output(row: pd.Series) -> str | None:
    """Pull the assistant response out of either a flat `output` column or
    upstream's nested `messages[-1].content.answer` shape.
    """
    output_value = row.get("output")
    if isinstance(output_value, str) and output_value:
        return output_value
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        content = last.get("content") if isinstance(last, dict) else None
        if isinstance(content, dict):
            answer = content.get("answer")
            return answer if isinstance(answer, str) and answer else None
        if isinstance(content, str) and content:
            return content
    return None


def _pick_instruction_index(raw_df: pd.DataFrame) -> pd.Series:
    for col in ["instruction_index", "uid", "question_id", "id"]:
        if col in raw_df.columns:
            return raw_df[col].astype(str)
    return pd.Series(range(len(raw_df)), dtype=str, index=raw_df.index)


def _pick_instruction(raw_df: pd.DataFrame) -> pd.Series:
    for col in ["instruction", "prompt", "question", "turns"]:
        if col in raw_df.columns:
            if col == "turns":
                return raw_df[col].apply(_turns_to_text)
            return raw_df[col]
    raise ValueError(
        "Unable to infer instruction text column from Arena-Hard data. "
        f"Available columns: {raw_df.columns.tolist()}"
    )


def _turns_to_text(turns_value: Any) -> str:
    if isinstance(turns_value, list):
        if not turns_value:
            return ""
        first = turns_value[0]
        if isinstance(first, dict):
            for key in ["content", "text", "prompt"]:
                if key in first:
                    return str(first[key])
        return str(first)
    if isinstance(turns_value, dict):
        for key in ["content", "text", "prompt"]:
            if key in turns_value:
                return str(turns_value[key])
    return str(turns_value)
