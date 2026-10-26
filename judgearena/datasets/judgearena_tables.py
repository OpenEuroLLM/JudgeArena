"""Dataset adapter for JudgeArena's packaged instruction/output tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.datasets.data import CompletionData, PreferenceData
from judgearena.datasets.registry import DatasetAdapter, register_dataset
from judgearena.tasks.schema import HuggingFaceDatasetSource


@register_dataset("judgearena_tables")
class TableAdapter(DatasetAdapter):
    def _table_path(self, directory: str, suffix: str) -> Path:
        """Return an exact task-declared table path, falling back to its task ID."""
        paths = {
            Path(pattern)
            for source in self.task.spec.dataset.sources.values()
            if isinstance(source, HuggingFaceDatasetSource)
            for pattern in source.allow_patterns
            if pattern.startswith(f"{directory}/")
            and not any(character in pattern for character in "*?[")
        }
        if len(paths) > 1:
            raise ValueError(
                f"Task {self.task.task!r} declares multiple {directory!r} table paths."
            )
        return next(iter(paths), Path(directory) / f"{self.task.task}{suffix}")

    def _instruction_path(self) -> Path:
        return self.cache_dir / self._table_path("instructions", ".csv")

    def _model_output_path(self) -> Path:
        return self.cache_dir / self._table_path("model_outputs", ".csv.zip")

    def download(self) -> None:
        """Download every Hugging Face source declared by a table-backed task."""
        if not self.task.spec.dataset.sources:
            raise ValueError("Table adapter requires dataset.sources.")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        for name, source in self.task.spec.dataset.sources.items():
            if not isinstance(source, HuggingFaceDatasetSource):
                raise ValueError(
                    f"Dataset source {name!r} for task {self.task.task!r} is not supported "
                    "by the 'judgearena_tables' adapter."
                )
            snapshot_download(
                repo_id=source.repo_id,
                repo_type="dataset",
                revision=source.revision,
                allow_patterns=list(source.allow_patterns) or None,
                local_dir=self.cache_dir,
                force_download=False,
            )

    def _load_instruction_table(self) -> pd.DataFrame:
        fields = self.fields
        self.download()
        path = self._instruction_path()
        if not path.exists():
            raise FileNotFoundError(f"Instruction table not found at {path}")
        frame = pd.read_csv(path)
        required = {fields.id, fields.instruction}
        if fields.category is not None:
            required.add(fields.category)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Task {self.task.task!r} is missing declared dataset fields: {missing}."
            )
        if frame[fields.id].duplicated().any():
            raise ValueError(
                f"Task {self.task.task!r} contains duplicate instruction IDs."
            )
        return frame

    def load_data(self) -> CompletionData:
        if self.task.spec.dataset.format == "battles":
            self.download()
            return PreferenceData.from_battles(
                pd.read_csv(self.cache_dir / self._table_path("battles", ".csv")),
                columns=self.task.spec.dataset.battle_columns,
                outcome_preferences=self.task.spec.dataset.outcome_preferences,
            )
        instructions = self._load_instruction_table()
        return CompletionData.from_frames(
            self._normalize_instructions(instructions),
            self._read_model_outputs(instructions),
        )

    def _normalize_instructions(self, frame: pd.DataFrame) -> pd.DataFrame:
        fields = self.fields
        if fields.id_strategy == "position":
            frame = frame.assign(**{fields.id: range(len(frame))})
        field_mapping = {
            fields.id: "instruction_index",
            fields.instruction: "instruction",
        }
        if fields.category is not None:
            field_mapping[fields.category] = "category"
        return frame.rename(columns=field_mapping)

    def _read_model_outputs(self, instructions: pd.DataFrame) -> pd.DataFrame | None:
        path = self._model_output_path()
        if not path.exists():
            return None
        outputs = pd.read_csv(path).rename(
            columns=self.task.spec.dataset.output_columns
        )
        if self.fields.id_strategy != "position":
            return outputs

        source_id = self.fields.id
        id_map = pd.Series(
            range(len(instructions)),
            index=instructions[source_id].astype(str),
            dtype=int,
        )
        mapped = outputs["instruction_index"].astype(str).map(id_map)
        if mapped.isna().any():
            unknown = outputs.loc[mapped.isna(), "instruction_index"].unique()[:5]
            raise ValueError(
                f"Task {self.task.task!r} model outputs contain unknown instruction IDs: "
                f"{unknown.tolist()}."
            )
        outputs["instruction_index"] = mapped.astype(int)
        return outputs
