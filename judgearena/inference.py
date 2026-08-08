"""Lazy model preparation and inference-cache context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd

from judgearena.cache_sqlite import (
    COMPLETION_DB_NAME,
    JUDGEMENT_DB_NAME,
    CacheKind,
    CompletionCache,
    JudgementCache,
    cache_folder,
    stable_json_dumps,
    write_descriptor,
)

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def canonicalize_chat_input(input_item: Any) -> str:
    """Serialize a logical model input for content-addressed cache lookup."""
    if isinstance(input_item, str):
        payload = {"type": "text", "text": input_item}
    elif hasattr(input_item, "to_messages"):
        payload = {
            "type": "messages",
            "messages": [
                {
                    "role": _ROLE_MAP.get(message.type, message.type),
                    "content": message.content,
                }
                for message in input_item.to_messages()
            ],
        }
    elif isinstance(input_item, list) and input_item:
        if isinstance(input_item[0], tuple):
            messages = [
                {
                    "role": "user" if role == "human" else role,
                    "content": content,
                }
                for role, content in input_item
            ]
        elif isinstance(input_item[0], dict):
            messages = [
                {"role": message["role"], "content": message["content"]}
                for message in input_item
            ]
        else:
            raise TypeError(f"Unsupported inference input: {type(input_item)!r}")
        payload = {"type": "messages", "messages": messages}
    else:
        raise TypeError(f"Unsupported inference input: {type(input_item)!r}")
    return stable_json_dumps(payload)


def build_model_descriptor(
    provider: str,
    model_name: str,
    resolved_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Describe output-affecting settings without constructing the backend."""
    if provider not in {"Dummy", "VLLM"}:
        return None

    backend_version = None
    if provider == "VLLM":
        try:
            backend_version = importlib_metadata.version("vllm")
        except importlib_metadata.PackageNotFoundError:
            return None

    descriptor = {
        "schema_version": "judgearena-inference-cache/v1",
        "provider": provider,
        "model": model_name,
        "backend_version": backend_version,
        "model_kwargs": resolved_kwargs,
    }
    if provider == "VLLM":
        descriptor["sampling"] = {"temperature": 0.6, "top_p": 0.95}
    try:
        stable_json_dumps(descriptor)
    except TypeError:
        return None
    return descriptor


@dataclass
class PreparedModel:
    """Carry cache identity while deferring backend construction until a miss."""

    model_spec: str
    descriptor: dict[str, Any] | None
    factory: Callable[[], Any]
    _model: Any = field(default=None, init=False, repr=False)

    def materialize(self) -> Any:
        if self._model is None:
            self._model = self.factory()
        return self._model


@dataclass(frozen=True)
class InferenceCache:
    """Select the role-specific store used by the inference cache boundary."""

    store_root: Path
    kind: CacheKind
    task: str
    pushed_by: str = "judgearena"

    def open_store(self, model: PreparedModel) -> CompletionCache | JudgementCache:
        assert model.descriptor is not None
        folder = cache_folder(
            self.store_root,
            self.kind,
            self.task,
            model.model_spec,
            model.descriptor,
        )
        write_descriptor(folder, model.descriptor)
        if self.kind == "completions":
            return CompletionCache(folder / COMPLETION_DB_NAME)
        return JudgementCache(folder / JUDGEMENT_DB_NAME)

    def save_outputs(
        self,
        store: CompletionCache | JudgementCache,
        model: PreparedModel,
        input_texts: list[str],
        outputs: list[str],
        metadata: list[dict[str, Any]],
        indices: list[int],
    ) -> None:
        rows = []
        for index, output in zip(indices, outputs, strict=True):
            row_metadata = metadata[index]
            common = {
                "benchmark": self.task,
                "instruction_id": row_metadata["instruction_id"],
            }
            if self.kind == "completions":
                rows.append(
                    {
                        **common,
                        "input_text": input_texts[index],
                        "completion": output,
                        "model": model.model_spec,
                    }
                )
            else:
                rows.append(
                    {
                        **common,
                        "judge_input": input_texts[index],
                        "judge_completion": output,
                        "model_a": row_metadata["model_a"],
                        "model_b": row_metadata["model_b"],
                        "judge": model.model_spec,
                        "orientation": row_metadata.get("orientation"),
                    }
                )
        store.save(pd.DataFrame(rows), pushed_by=self.pushed_by)
