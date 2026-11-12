"""Normalize HelpSteer3 preference examples into labeled answer comparisons."""

from __future__ import annotations

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.datasets.data import PreferenceData, validate_messages
from judgearena.datasets.registry import DatasetAdapter
from judgearena.tasks.schema import HuggingFaceDatasetSource

_LANGUAGES = {
    "english": "en",
    "chinese": "zh",
    "korean": "ko",
    "french": "fr",
    "spanish": "es",
    "russian": "ru",
    "japanese": "ja",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
    "indonesian": "id",
    "dutch": "nl",
    "vietnamese": "vi",
}


class HelpSteer3Adapter(DatasetAdapter):
    def _source(self) -> HuggingFaceDatasetSource:
        source = self.task.spec.dataset.sources.get("preference")
        if not isinstance(source, HuggingFaceDatasetSource) or source.config not in (
            None,
            "preference",
        ):
            raise ValueError("HelpSteer3 requires a Hugging Face preference source.")
        return source

    def _split(self) -> str:
        split = self.task.spec.dataset.options.get("split", "validation")
        if split not in ("train", "validation"):
            raise ValueError("HelpSteer3 preference split must be train or validation.")
        return split

    def download(self) -> None:
        source = self._source()
        snapshot_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=[f"preference/{self._split()}.jsonl.gz"],
            local_dir=self.source_dir,
            force_download=False,
        )

    def load_data(self) -> PreferenceData:
        self.download()
        source = self._source()
        split = self._split()
        records = pd.read_json(
            self.source_dir / f"preference/{split}.jsonl.gz", lines=True
        ).to_dict("records")
        rows = []
        for position, record in enumerate(records):
            identity = f"{source.repo_id}:{source.revision}:{split}:{position}"
            context = record.get("context")
            try:
                validate_messages(context, ending="user")
            except ValueError as exc:
                raise ValueError(
                    f"HelpSteer3 {identity}: invalid context: {exc}"
                ) from exc
            score = record.get("overall_preference")
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or score not in range(-3, 4)
            ):
                raise ValueError(
                    f"HelpSteer3 {identity}: overall_preference must be an integer from -3 to 3."
                )
            for field in ("response1", "response2"):
                if not isinstance(record.get(field), str):
                    raise ValueError(f"HelpSteer3 {identity}: {field} must be text.")
            language = record.get("language")
            # In the code domain this field names the programming language;
            # the surrounding natural-language prompts are English.
            natural_language = "english" if record.get("domain") == "code" else language
            if natural_language not in _LANGUAGES:
                raise ValueError(
                    f"HelpSteer3 {identity}: unknown natural language {natural_language!r}."
                )
            rows.append(
                {
                    **record,
                    "battle_id": identity,
                    "instruction_index": identity,
                    "instruction": context[-1]["content"],
                    "messages": context,
                    "messages_a": context
                    + [{"role": "assistant", "content": record["response1"]}],
                    "messages_b": context
                    + [{"role": "assistant", "content": record["response2"]}],
                    # These identify response collections, not generating models.
                    "model_a": "helpsteer3/response1",
                    "model_b": "helpsteer3/response2",
                    "completion_a": record["response1"],
                    "completion_b": record["response2"],
                    "reference_pref": 0.0 if score < 0 else 1.0 if score > 0 else 0.5,
                    "lang": _LANGUAGES[natural_language],
                    "source": "human",
                    "split": split,
                }
            )
        if not rows:
            raise ValueError("HelpSteer3 preference source contains no examples.")
        return PreferenceData.from_battles(pd.DataFrame(rows))
