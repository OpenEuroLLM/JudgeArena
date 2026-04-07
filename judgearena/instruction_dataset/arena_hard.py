from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset

ARENA_HARD_HF_REPO_ID = "lmarena-ai/arena-hard-auto"


@dataclass(frozen=True)
class ArenaHardSpec:
    hf_variant: str
    baseline_model: str


ARENA_HARD_DATASETS: dict[str, ArenaHardSpec] = {
    "arena-hard-v0.1": ArenaHardSpec(
        hf_variant="arena-hard-v0.1",
        baseline_model="gpt-4-0314",
    ),
    "arena-hard-v2.0": ArenaHardSpec(
        hf_variant="arena-hard-v2.0",
        baseline_model="o3-mini-2025-01-31",
    ),
}


def resolve_arena_hard_spec(dataset: str) -> ArenaHardSpec | None:
    return ARENA_HARD_DATASETS.get(dataset)


def is_arena_hard_dataset(dataset: str) -> bool:
    return resolve_arena_hard_spec(dataset) is not None


def arena_hard_baseline_model(dataset: str) -> str | None:
    spec = resolve_arena_hard_spec(dataset)
    if spec is None:
        return None
    return spec.baseline_model


def _load_official_arena_hard_dataset(spec: ArenaHardSpec) -> pd.DataFrame:
    data = load_dataset(
        path=ARENA_HARD_HF_REPO_ID,
        data_dir=f"data/{spec.hf_variant}",
    )
    return _dataset_like_to_dataframe(data)


def _dataset_like_to_dataframe(
    data: Dataset | DatasetDict | IterableDataset,
) -> pd.DataFrame:
    if isinstance(data, DatasetDict):
        if "train" in data:
            return data["train"].to_pandas()
        first_split = next(iter(data.keys()))
        return data[first_split].to_pandas()
    if isinstance(data, Dataset):
        return data.to_pandas()
    if isinstance(data, IterableDataset):
        return pd.DataFrame(list(data))
    raise TypeError(f"Unsupported dataset object type: {type(data)}")


def normalize_official_arena_hard(
    raw_df: pd.DataFrame, dataset: str
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    spec = resolve_arena_hard_spec(dataset)
    if spec is None:
        raise ValueError(f"Unsupported Arena-Hard dataset: {dataset}")

    instruction_index = _pick_instruction_index(raw_df)
    instruction = _pick_instruction(raw_df)
    df_instructions = pd.DataFrame(
        {
            "instruction_index": instruction_index,
            "instruction": instruction,
        }
    )
    df_instructions = df_instructions.dropna(
        subset=["instruction_index", "instruction"]
    )
    df_instructions = df_instructions.drop_duplicates(subset=["instruction_index"])
    df_instructions = df_instructions.sort_values("instruction_index")

    df_model_outputs = _build_model_outputs(raw_df)
    return df_instructions, df_model_outputs


def download_arena_hard(dataset: str, local_tables_path: Path) -> None:
    """Load Arena-Hard from the Hub if instruction and model-output files are missing."""
    spec = resolve_arena_hard_spec(dataset)
    if spec is None:
        return
    instructions_path = local_tables_path / "instructions" / f"{dataset}.csv"
    model_outputs_path = local_tables_path / "model_outputs" / f"{dataset}.csv.zip"
    if instructions_path.exists() and model_outputs_path.exists():
        return

    raw_df = _load_official_arena_hard_dataset(spec)
    df_instructions, df_model_outputs = normalize_official_arena_hard(
        raw_df=raw_df, dataset=dataset
    )
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    model_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    df_instructions.to_csv(instructions_path, index=False)
    if df_model_outputs is not None:
        df_model_outputs.to_csv(model_outputs_path, index=False)


def _pick_instruction_index(raw_df: pd.DataFrame) -> pd.Series:
    for col in ["instruction_index", "question_id", "id"]:
        if col in raw_df.columns:
            return raw_df[col].astype(str)
    return pd.Series(range(len(raw_df)), dtype=str)


def _pick_instruction(raw_df: pd.DataFrame) -> pd.Series:
    for col in ["instruction", "prompt", "question", "turns"]:
        if col in raw_df.columns:
            if col == "turns":
                return raw_df[col].apply(_turns_to_text)
            return raw_df[col].astype(str)
    raise ValueError(
        f"Unable to infer instruction text column from Arena-Hard data. Available columns: {raw_df.columns.tolist()}"
    )


def _turns_to_text(turns_value) -> str:
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


def _build_model_outputs(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    if not {"model", "output"}.issubset(raw_df.columns):
        return None
    instruction_index = _pick_instruction_index(raw_df)
    df_outputs = pd.DataFrame(
        {
            "instruction_index": instruction_index,
            "model": raw_df["model"].astype(str),
            "output": raw_df["output"].fillna("").astype(str),
        }
    )
    return df_outputs
