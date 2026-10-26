"""Dataset adapter for YAML-defined m-ArenaHard task families."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.datasets.data import CompletionData
from judgearena.datasets.registry import DatasetAdapter, register_dataset
from judgearena.tasks.schema import HuggingFaceDatasetSource


def _source_local_dir(source: HuggingFaceDatasetSource, root: Path) -> Path:
    """Keep raw source snapshots separate from normalized JudgeArena tables."""
    return root / "_sources" / source.repo_id.replace("/", "--")


@register_dataset("m_arena_hard")
class MArenaHardAdapter(DatasetAdapter):
    def _source(self, name: str = "examples") -> HuggingFaceDatasetSource:
        source = self.task.spec.dataset.sources.get(name)
        if not isinstance(source, HuggingFaceDatasetSource):
            raise ValueError(
                f"Task {self.task.task!r} must define a Hugging Face dataset source "
                f"named {name!r}."
            )
        return source

    def download(self) -> None:
        """Download the pinned examples and optional packaged model outputs."""
        self._source()
        for name in self.task.spec.dataset.sources:
            self._download_source(name)

    def _download_source(
        self, name: str, *, allow_patterns: list[str] | None = None
    ) -> None:
        declared_source = self._source(name)
        local_dir = (
            _source_local_dir(declared_source, self.cache_dir)
            if name == "examples"
            else self.cache_dir
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

    def _selected_languages(self) -> tuple[str, ...]:
        variants = self.task.spec.variants
        if variants is None or variants.selector != "lang":
            raise ValueError(
                f"Task {self.task.task!r} must define language suffix variants."
            )
        return (
            self.task.selection.values
            if self.task.selection is not None
            else variants.values
        )

    def _load_source_frames(self) -> pd.DataFrame:
        fields = self.fields
        self._download_source("examples")
        source = self._source()
        source_root = _source_local_dir(source, self.cache_dir)
        selected = set(self._selected_languages())

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
        missing = sorted({fields.id, fields.instruction} - set(df.columns))
        if missing:
            raise ValueError(
                f"Task {self.task.task!r} is missing declared dataset fields: {missing}."
            )
        df[fields.id] = df[fields.id].astype(str) + "-" + df["lang"]
        if df[fields.id].duplicated().any():
            duplicates = df.loc[df[fields.id].duplicated(), fields.id].head().tolist()
            raise ValueError(
                f"Task {self.task.task!r} contains duplicate instruction IDs: {duplicates}."
            )
        return df

    def load_data(self) -> CompletionData:
        return CompletionData.from_frames(
            self.load_instructions(),
            self.load_model_outputs(),
        )

    def load_instructions(self) -> pd.DataFrame:
        """Load and normalize the languages selected by the task invocation."""
        df = self._load_source_frames()
        fields = self.fields
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

    def load_model_outputs(self) -> pd.DataFrame | None:
        """Load pre-generated outputs when the selected task view ships them."""
        if self.task.selection is None:
            return None
        path = self.cache_dir / "model_outputs" / f"{self.task.task}.csv.zip"
        self._download_source(
            "outputs", allow_patterns=[path.relative_to(self.cache_dir).as_posix()]
        )
        return pd.read_csv(path) if path.exists() else None
