from pathlib import Path

import pandas as pd

import judgearena.generate_and_evaluate as generate_and_evaluate
import judgearena.instruction_dataset as instruction_dataset
from judgearena.instruction_dataset.arena_hard import (
    arena_hard_baseline_model,
    normalize_official_arena_hard,
)


def test_arena_hard_baseline_resolution():
    assert arena_hard_baseline_model("arena-hard-v0.1") == "gpt-4-0314"
    assert arena_hard_baseline_model("arena-hard-v2.0") == "o3-mini-2025-01-31"


def test_normalize_official_arena_hard_v01_shape():
    raw_df = pd.DataFrame(
        {
            "question_id": ["q1", "q2"],
            "prompt": ["First prompt", "Second prompt"],
            "model": ["baseline", "baseline"],
            "output": ["a1", "a2"],
        }
    )
    df_instructions, df_outputs = normalize_official_arena_hard(
        raw_df=raw_df, dataset="arena-hard-v0.1"
    )

    assert df_instructions.columns.tolist() == ["instruction_index", "instruction"]
    assert df_instructions["instruction_index"].tolist() == ["q1", "q2"]
    assert df_instructions["instruction"].tolist() == ["First prompt", "Second prompt"]
    assert df_outputs is not None
    assert set(df_outputs.columns) == {"instruction_index", "model", "output"}


def test_load_instructions_uses_explicit_version_filename(monkeypatch):
    captured = {}

    def _fake_ensure(dataset: str, local_tables_path: Path):
        captured["dataset"] = dataset
        captured["local_tables_path"] = local_tables_path

    def _fake_read_df(path: Path):
        captured["path"] = path
        return pd.DataFrame(
            {
                "instruction_index": ["0", "1"],
                "instruction": ["hello", "world"],
            }
        )

    monkeypatch.setattr(instruction_dataset, "download_arena_hard", _fake_ensure)
    monkeypatch.setattr(instruction_dataset, "read_df", _fake_read_df)
    df = instruction_dataset.load_instructions(dataset="arena-hard-v2.0")

    assert captured["dataset"] == "arena-hard-v2.0"
    assert captured["path"].name == "arena-hard-v2.0.csv"
    assert df.index.tolist() == ["0", "1"]


def test_try_load_dataset_completions_uses_dataset_output_file(monkeypatch, tmp_path):
    tables_dir = tmp_path / "tables" / "model_outputs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "arena-hard-v2.0.csv.zip"
    pd.DataFrame(
        {
            "instruction_index": [0, 0, 1, 1],
            "model": ["baseline", "candidate", "baseline", "candidate"],
            "output": ["b0", "c0", "b1", "c1"],
        }
    ).to_csv(output_path, index=False)

    monkeypatch.setattr(generate_and_evaluate, "data_root", tmp_path)
    monkeypatch.setattr(
        generate_and_evaluate,
        "download_arena_hard",
        lambda dataset, local_tables_path: None,
    )

    loaded = generate_and_evaluate.try_load_dataset_completions(
        dataset="arena-hard-v2.0", model="baseline", n_instructions=None
    )

    assert loaded is not None
    assert loaded["completion"].tolist() == ["b0", "b1"]
    assert loaded["instruction_index"].tolist() == [0, 1]
