"""WildBench conversations and saved model answers for shared judging."""

import pandas as pd
from datasets import get_dataset_config_names, load_dataset

from judgearena.datasets.data import CompletionData, validate_messages
from judgearena.datasets.registry import DatasetAdapter, register_dataset
from judgearena.tasks.schema import HuggingFaceDatasetSource


@register_dataset("wildbench")
class WildBenchAdapter(DatasetAdapter):
    def _source(self, name: str) -> HuggingFaceDatasetSource:
        source = self.task.spec.dataset.sources.get(name)
        if not isinstance(source, HuggingFaceDatasetSource):
            raise ValueError(
                f"WildBench requires a Hugging Face dataset source named {name!r}."
            )
        return source

    def _load_examples(self):
        source = self._source("examples")
        return load_dataset(
            source.repo_id, source.config, revision=source.revision, split="test"
        )

    def download(self) -> None:
        """Cache example rows; saved answers are downloaded when a model is selected."""
        self._load_examples()

    def load_data(self) -> CompletionData:
        """Keep supplied history for generation and checklist text for the judge.

        ``examples`` selects the dataset/config; optional ``outputs`` holds saved
        answers in model-named configurations. Unknown models generate normally.
        """
        if self.task.spec.dataset.fields is not None:
            raise ValueError(
                "WildBench does not use dataset.fields; use a custom adapter for a different source schema."
            )
        records = self._load_examples()
        rows = []
        for record in records:
            messages = [
                {"role": message["role"], "content": message["content"]}
                for message in record["conversation_input"]
            ]
            validate_messages(messages, ending="user")
            checklist = record["checklist"]
            if not isinstance(checklist, list) or not all(
                isinstance(item, str) for item in checklist
            ):
                raise ValueError("WildBench checklist must be a list of strings.")
            rows.append(
                {
                    **record,
                    "instruction_index": record["session_id"],
                    "instruction": messages[-1]["content"],
                    "messages": messages,
                    "history": "".join(
                        f"{message['role'].upper()}: {message['content']}\n\n"
                        for message in messages[:-1]
                    ),
                    "checklist": "".join(f"- {item}\n" for item in checklist),
                    "category": record["primary_tag"],
                }
            )

        loader = None
        if "outputs" in self.task.spec.dataset.sources:
            outputs = self._source("outputs")
            models = get_dataset_config_names(
                outputs.repo_id, revision=outputs.revision
            )

            def loader(model):
                if model not in models:
                    return None
                answers = load_dataset(
                    outputs.repo_id, model, revision=outputs.revision, split="train"
                )
                frame = pd.DataFrame(answers).rename(
                    columns={"session_id": "instruction_index", "output": "completion"}
                )
                frame["completion"] = frame["completion"].map(
                    lambda value: value[0] if isinstance(value, list) else value
                )
                return frame.assign(model=model)

        return CompletionData.from_frames(
            pd.DataFrame(rows), model_output_loader=loader
        )
