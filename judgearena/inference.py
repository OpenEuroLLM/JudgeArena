"""Lazy model preparation and inference-cache context."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

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
    else:
        raise TypeError(f"Unsupported inference input: {type(input_item)!r}")
    return stable_json_dumps(payload)


def build_model_descriptor(
    provider: str,
    model_name: str,
    resolved_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Describe output-affecting settings without constructing the backend."""
    if provider != "Dummy":
        return None
    return {
        "schema_version": "judgearena-inference-cache/v1",
        "provider": provider,
        "model": model_name,
        "input_mode": "chat",
        "model_kwargs": resolved_kwargs,
    }


@dataclass(frozen=True)
class CachedInferenceResult:
    """Provider output fields required by downstream parsing."""

    text: str
    first_token_top_logprobs: dict[str, float] | None = None


@dataclass
class PreparedModel:
    """Carry cache identity while deferring backend construction until a miss."""

    model_spec: str
    descriptor: dict[str, Any] | None
    factory: Callable[[], Any]
    cache: InferenceCache | None = None
    _model: Any = field(default=None, init=False, repr=False)

    def materialize(self) -> Any:
        if self._model is None:
            self._model = self.factory()
        return self._model


@dataclass(frozen=True)
class InferenceCache(ABC):
    """Share cache lifecycle while subclasses define role-specific rows."""

    store_root: Path
    task: str
    pushed_by: str = "judgearena"

    kind: ClassVar[CacheKind]
    db_name: ClassVar[str]
    store_type: ClassVar[type[CompletionCache] | type[JudgementCache]]

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
        return self.store_type(folder / self.db_name)

    def save_outputs(
        self,
        store: CompletionCache | JudgementCache,
        model: PreparedModel,
        input_texts: list[str],
        outputs: list[Any],
        metadata: list[dict[str, Any]],
        indices: list[int],
    ) -> None:
        rows = [
            self.make_row(
                model=model,
                input_text=input_texts[index],
                output=output,
                metadata=metadata[index],
            )
            for index, output in zip(indices, outputs, strict=True)
        ]
        store.save(pd.DataFrame(rows), pushed_by=self.pushed_by)

    @abstractmethod
    def make_row(
        self,
        *,
        model: PreparedModel,
        input_text: str,
        output: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert one inference output to its role-specific storage row."""

    @abstractmethod
    def cached_result(self, row: pd.Series) -> CachedInferenceResult:
        """Restore output fields from a stored row."""


class CompletionInferenceCache(InferenceCache):
    """Cache generated model completions."""

    kind = "completions"
    db_name = COMPLETION_DB_NAME
    store_type = CompletionCache

    def make_row(
        self,
        *,
        model: PreparedModel,
        input_text: str,
        output: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "input_text": input_text,
            "completion": output.text,
            "benchmark": self.task,
            "instruction_id": metadata["instruction_id"],
            "model": model.model_spec,
        }

    def cached_result(self, row: pd.Series) -> CachedInferenceResult:
        return CachedInferenceResult(text=str(row["completion"]))


class JudgementInferenceCache(InferenceCache):
    """Cache raw judge completions."""

    kind = "judgements"
    db_name = JUDGEMENT_DB_NAME
    store_type = JudgementCache

    def make_row(
        self,
        *,
        model: PreparedModel,
        input_text: str,
        output: Any,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "judge_input": input_text,
            "judge_completion": output.text,
            "benchmark": self.task,
            "instruction_id": metadata["instruction_id"],
            "model_a": metadata["model_a"],
            "model_b": metadata["model_b"],
            "judge": model.model_spec,
            "top_logprobs": output.first_token_top_logprobs,
            "orientation": metadata.get("orientation"),
        }

    def cached_result(self, row: pd.Series) -> CachedInferenceResult:
        top_logprobs = row["top_logprobs"]
        return CachedInferenceResult(
            text=str(row["judge_completion"]),
            first_token_top_logprobs=(
                json.loads(top_logprobs) if pd.notna(top_logprobs) else None
            ),
        )
