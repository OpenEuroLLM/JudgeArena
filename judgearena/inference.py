"""Lazy model preparation and inference-cache context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, ClassVar, Literal

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
_CHAT_PROVIDERS = {"ChatOpenAI", "Dummy", "OpenRouter", "VLLM"}
_TEXT_PROVIDERS = {"LlamaCpp", "OpenAI", "Together"}
_CREDENTIAL_KEYS = {
    "api_key",
    "authorization",
    "openai_api_key",
    "token",
    "together_api_key",
}
VLLM_TEMPERATURE = 0.6
VLLM_TOP_P = 0.95
VLLM_EXECUTION_ONLY_KWARGS = {
    "enforce_eager",
    "gpu_memory_utilization",
    "tensor_parallel_size",
}

InputMode = Literal["auto", "chat", "text"]


def provider_input_mode(
    provider: str, resolved_kwargs: dict[str, Any] | None = None
) -> InputMode | None:
    if provider == "VLLM" and not (resolved_kwargs or {}).get("chat_template"):
        return "auto"
    if provider in _CHAT_PROVIDERS:
        return "chat"
    if provider in _TEXT_PROVIDERS:
        return "text"
    return None


def _canonical_messages(input_item: Any) -> list[dict[str, Any]]:
    if isinstance(input_item, str):
        return [{"role": "user", "content": input_item}]
    if hasattr(input_item, "to_messages"):
        return [
            {
                "role": _ROLE_MAP.get(message.type, message.type),
                "content": message.content,
            }
            for message in input_item.to_messages()
        ]
    raise TypeError(f"Unsupported inference input: {type(input_item)!r}")


def _canonical_text(input_item: Any) -> str:
    if isinstance(input_item, str):
        return input_item
    if hasattr(input_item, "to_string"):
        return input_item.to_string()
    raise TypeError(f"Unsupported inference input: {type(input_item)!r}")


def canonicalize_model_input(input_item: Any, input_mode: InputMode) -> str:
    """Serialize the exact chat or flattened text input seen by the backend."""
    if input_mode == "text":
        payload = {"type": "text", "text": _canonical_text(input_item)}
    elif input_mode == "chat":
        payload = {"type": "messages", "messages": _canonical_messages(input_item)}
    else:
        payload = {
            "type": "auto",
            "messages": _canonical_messages(input_item),
            "text": _canonical_text(input_item),
        }
    return stable_json_dumps(payload)


def _without_credentials(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_credentials(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in _CREDENTIAL_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_credentials(item) for item in value]
    return value


def build_model_descriptor(
    provider: str,
    model_name: str,
    resolved_kwargs: dict[str, Any],
    endpoint: str | None = None,
) -> dict[str, Any] | None:
    """Describe the configured request without constructing the backend.

    Omitted hosted settings retain provider defaults. LlamaCpp model identity
    remains path-based, so moving the same GGUF creates a different cache cell.
    Local-engine versions must be installed to derive their cache keys. VLLM
    tokenizer-template changes require a distinct model revision or template.
    """
    input_mode = provider_input_mode(provider, resolved_kwargs)
    if input_mode is None:
        return None

    descriptor_kwargs = _without_credentials(resolved_kwargs)
    if provider == "VLLM":
        descriptor_kwargs = {
            key: value
            for key, value in descriptor_kwargs.items()
            if key not in VLLM_EXECUTION_ONLY_KWARGS
        }

    descriptor = {
        "schema_version": "judgearena-inference-cache/v1",
        "provider": provider,
        "model": model_name,
        "input_mode": input_mode,
        "model_kwargs": descriptor_kwargs,
    }
    if provider == "VLLM":
        descriptor["backend_version"] = importlib_metadata.version("vllm")
        descriptor["sampling"] = {
            "temperature": VLLM_TEMPERATURE,
            "top_p": VLLM_TOP_P,
        }
    elif provider == "LlamaCpp":
        descriptor["backend_version"] = importlib_metadata.version("llama-cpp-python")
    if endpoint is not None:
        descriptor["endpoint"] = endpoint.rstrip("/")
    return descriptor


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
    output_column: ClassVar[str]
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
        outputs: list[str],
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
        output: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert one inference output to its role-specific storage row."""


class CompletionInferenceCache(InferenceCache):
    """Cache generated model completions."""

    kind = "completions"
    db_name = COMPLETION_DB_NAME
    output_column = "completion"
    store_type = CompletionCache

    def make_row(
        self,
        *,
        model: PreparedModel,
        input_text: str,
        output: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "input_text": input_text,
            "completion": output,
            "benchmark": self.task,
            "instruction_id": metadata["instruction_id"],
            "model": model.model_spec,
        }


class JudgementInferenceCache(InferenceCache):
    """Cache raw judge completions."""

    kind = "judgements"
    db_name = JUDGEMENT_DB_NAME
    output_column = "judge_completion"
    store_type = JudgementCache

    def make_row(
        self,
        *,
        model: PreparedModel,
        input_text: str,
        output: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "judge_input": input_text,
            "judge_completion": output,
            "benchmark": self.task,
            "instruction_id": metadata["instruction_id"],
            "model_a": metadata["model_a"],
            "model_b": metadata["model_b"],
            "judge": model.model_spec,
            "orientation": metadata.get("orientation"),
        }
