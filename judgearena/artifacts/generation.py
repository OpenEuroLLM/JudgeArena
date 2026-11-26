"""Saved candidate answers for later evaluation, without judged metrics."""

import json
from pathlib import Path

import pandas as pd

from judgearena.artifacts.evaluation import file_checksum, verify_files
from judgearena.artifacts.run import read_metric_data
from judgearena.datasets.data import restore_nested_columns


def save_generation(folder, cfg, instructions, answers):
    folder = Path(folder)
    answers.rename_axis("instruction_index").reset_index().to_parquet(
        folder / "answers.parquet", index=False
    )
    instructions.to_parquet(folder / "instructions.parquet")
    payload = dict(
        schema_version=1,
        model=cfg.model.name,
        generation_kwargs=cfg.model.evaluated_generation_kwargs(),
        files={
            str(p.relative_to(folder)): file_checksum(p)
            for p in folder.rglob("*")
            if p.is_file() and p.suffix != ".log" and p != folder / "generation.json"
        },
    )
    target = folder / "generation.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return target


def read_answers(path, cfg, instructions):
    path = Path(path)
    manifest = path.parent / "generation.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if (
            saved.get("schema_version") != 1
            or not {"answers.parquet", "instructions.parquet", "config.yaml"}
            <= saved.get("files", {}).keys()
        ):
            raise ValueError("Incomplete generation manifest")
        verify_files(path.parent, saved["files"])
        if (
            path.name != "answers.parquet"
            or saved["model"] != cfg.model.name
            or saved["generation_kwargs"] != cfg.model.evaluated_generation_kwargs()
        ):
            raise ValueError("Saved answers have different model generation settings")
        original = pd.read_parquet(path.parent / "instructions.parquet")
        if original.to_json(orient="split") != instructions.to_json(orient="split"):
            raise ValueError("Saved answers belong to different instructions")
    answers = read_metric_data(path)
    if "messages" in answers:
        answers["messages"] = restore_nested_columns(answers[["messages"]])["messages"]
    if not {"instruction_index", "model", "completion"} <= set(answers):
        raise ValueError(
            "Saved answers require instruction_index, model and completion"
        )
    answers = answers.loc[answers.model == cfg.model.name]
    if (
        answers.instruction_index.duplicated().any()
        or set(answers.instruction_index) != set(instructions.index)
        or answers.completion.isna().any()
    ):
        raise ValueError("Saved answers must cover the selected instructions exactly")
    return answers
