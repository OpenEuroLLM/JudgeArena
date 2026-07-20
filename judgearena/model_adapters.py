"""Cache descriptors, provider canonicalization, and PreparedModel adapters."""

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Callable
from typing import Any

from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain_together.llms import Together

from judgearena.store_sqlite import stable_json_dumps

DESCRIPTOR_SCHEMA_VERSION = "judgearena-inference-descriptor/v1"
HOSTED_ADAPTER_VERSION = "judgearena-hosted-adapter/v1"
LOCAL_LLAMACPP_ADAPTER_VERSION = "judgearena-local-llamacpp-adapter/v1"

_UNCACHED_MODEL_WARNINGS: set[type] = set()

_SECRET_EXACT_KEYS = frozenset(
    {
        "api_key",
        "openai_api_key",
        "together_api_key",
        "token",
        "password",
        "default_headers",
        "headers",
    }
)

_DESCRIPTOR_KEY_DENYLIST = frozenset(
    {
        "callback",
        "callbacks",
        "callback_manager",
        "client",
        "async_client",
        "http_client",
        "http_async_client",
        "root_client",
        "root_async_client",
        "cache",
        "verbose",
        "rate_limiter",
        "streaming",
        "disable_streaming",
        "tags",
        "name",
        "metadata",
    }
)

_TRANSIENT_MESSAGE_KEYS = frozenset({"id", "response_metadata", "usage_metadata"})


def top_k_from_settings(settings: dict[str, Any]) -> int | None:
    """Extract ``top_k`` without treating zero as missing."""
    if "top_k" in settings:
        value = settings["top_k"]
        return None if value is None else int(value)
    model_kwargs = settings.get("model_kwargs")
    if isinstance(model_kwargs, dict) and "top_k" in model_kwargs:
        value = model_kwargs["top_k"]
        return None if value is None else int(value)
    return None


def _is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in _SECRET_EXACT_KEYS:
        return True
    blocked_fragments = (
        "api_key",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "cookie",
        "password",
        "secret",
        "header",
    )
    if any(fragment in lowered for fragment in blocked_fragments):
        return True
    return lowered.endswith("_token") or lowered.endswith("_key")


def _provider_package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def normalize_descriptor_value(value: Any) -> Any | None:
    """Return a JSON-safe value, or None when normalization is unsafe."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        normalized_items = []
        for item in value:
            normalized = normalize_descriptor_value(item)
            if normalized is None and item is not None:
                return None
            normalized_items.append(normalized)
        return normalized_items
    if isinstance(value, (set, frozenset)):
        normalized_items = []
        for item in value:
            normalized = normalize_descriptor_value(item)
            if normalized is None and item is not None:
                return None
            normalized_items.append(normalized)
        return sorted(normalized_items, key=stable_json_dumps)
    if isinstance(value, dict):
        normalized_dict: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key_str = str(key)
            if _is_secret_key(key_str):
                continue
            normalized = normalize_descriptor_value(item)
            if normalized is None and item is not None:
                return None
            normalized_dict[key_str] = normalized
        return normalized_dict
    return None


def normalize_reasoning_config(value: Any) -> Any | None:
    if hasattr(value, "model_dump"):
        return normalize_descriptor_value(value.model_dump())
    if hasattr(value, "reasoning_start_str"):
        return normalize_descriptor_value(
            {
                "reasoning_start_str": getattr(value, "reasoning_start_str", None),
                "reasoning_end_str": getattr(value, "reasoning_end_str", None),
            }
        )
    return None


def normalize_constructor_settings(settings: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize every JSON-safe non-secret constructor setting conservatively."""
    normalized: dict[str, Any] = {}
    for key in sorted(settings):
        key_str = str(key)
        if key_str in _DESCRIPTOR_KEY_DENYLIST or _is_secret_key(key_str):
            continue
        value = settings[key]
        if value is None:
            continue
        if key_str == "reasoning_config":
            normalized_value = normalize_reasoning_config(value)
        else:
            normalized_value = normalize_descriptor_value(value)
        if normalized_value is None:
            return None
        normalized[key_str] = normalized_value
    return normalized


def _resolved_sampling_from_kwargs(resolved: Any) -> dict[str, Any] | None:
    sampling = normalize_descriptor_value(dict(resolved.sampling_params_kwargs))
    if sampling is None:
        return None
    sampling["max_tokens"] = resolved.max_tokens
    return sampling


def build_vllm_descriptor(model_spec: str, resolved: Any) -> dict[str, Any] | None:
    """Build a descriptor from fully resolved vLLM settings."""
    vllm_version = _provider_package_version("vllm")
    if vllm_version is None:
        return None

    engine_settings = normalize_constructor_settings(
        {
            **dict(resolved.vllm_kwargs),
            "chat_template": resolved.chat_template,
            "chat_template_kwargs": resolved.chat_template_kwargs,
        }
    )
    if engine_settings is None:
        return None

    sampling = _resolved_sampling_from_kwargs(resolved)
    if sampling is None:
        return None

    return {
        "descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "provider": "VLLM",
        "model_spec": model_spec,
        "model": resolved.model_path,
        "input_mode": resolved.input_mode,
        "chat_template": resolved.chat_template,
        "chat_template_kwargs": resolved.chat_template_kwargs,
        "sampling": sampling,
        "engine_settings": engine_settings,
        "vllm_version": vllm_version,
    }


def build_hosted_descriptor(
    *,
    provider: str,
    model_spec: str,
    model_name: str,
    max_tokens: int | None,
    input_mode: str,
    base_url: str | None,
    constructor_settings: dict[str, Any],
    resolved_sampling: dict[str, Any],
) -> dict[str, Any] | None:
    settings = normalize_constructor_settings(constructor_settings)
    if settings is None:
        return None
    sampling_settings = dict(resolved_sampling)
    if max_tokens is not None:
        sampling_settings["max_tokens"] = max_tokens
    sampling = normalize_descriptor_value(sampling_settings)
    if sampling is None:
        return None

    descriptor: dict[str, Any] = {
        "descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "provider": provider,
        "model_spec": model_spec,
        "model": model_name,
        "input_mode": input_mode,
        "hosted_adapter_version": HOSTED_ADAPTER_VERSION,
        "server_defaults": "unobserved",
        "sampling": sampling,
        "engine_settings": settings,
    }
    if base_url is not None:
        descriptor["base_url"] = base_url
    return descriptor


def build_llamacpp_descriptor(
    *,
    model_spec: str,
    model_name: str,
    max_tokens: int,
    constructor_settings: dict[str, Any],
    resolved_sampling: dict[str, Any],
) -> dict[str, Any] | None:
    llama_cpp_version = _provider_package_version("llama-cpp-python")
    if llama_cpp_version is None:
        return None
    settings = normalize_constructor_settings(constructor_settings)
    if settings is None:
        return None
    sampling = normalize_descriptor_value(
        {**resolved_sampling, "max_tokens": max_tokens}
    )
    if sampling is None:
        return None
    return {
        "descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "provider": "LlamaCpp",
        "model_spec": model_spec,
        "model": model_name,
        "input_mode": "raw",
        "local_adapter_version": LOCAL_LLAMACPP_ADAPTER_VERSION,
        "llama_cpp_python_version": llama_cpp_version,
        "sampling": sampling,
        "engine_settings": settings,
    }


def build_dummy_descriptor(
    *,
    model_spec: str,
    max_tokens: int,
    input_mode: str,
    constructor_settings: dict[str, Any],
    resolved_sampling: dict[str, Any],
) -> dict[str, Any] | None:
    settings = normalize_constructor_settings(constructor_settings)
    if settings is None:
        return None
    sampling = normalize_descriptor_value(
        {**resolved_sampling, "max_tokens": max_tokens}
    )
    if sampling is None:
        return None
    return {
        "descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "provider": "Dummy",
        "model_spec": model_spec,
        "model": model_spec.partition("/")[2],
        "input_mode": input_mode,
        "sampling": sampling,
        "engine_settings": settings,
    }


def build_producer_metadata(*, provider: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "provider": provider,
        "descriptor_schema_version": DESCRIPTOR_SCHEMA_VERSION,
    }
    if provider == "VLLM":
        version = _provider_package_version("vllm")
        if version is not None:
            metadata["vllm_version"] = version
    elif provider in {"OpenRouter", "OpenAI", "ChatOpenAI"}:
        metadata["hosted_adapter_version"] = HOSTED_ADAPTER_VERSION
        version = _provider_package_version("langchain-openai")
        if version is not None:
            metadata["langchain_openai_version"] = version
    elif provider == "Together":
        metadata["hosted_adapter_version"] = HOSTED_ADAPTER_VERSION
        version = _provider_package_version("langchain-together")
        if version is not None:
            metadata["langchain_together_version"] = version
    elif provider == "LlamaCpp":
        metadata["local_adapter_version"] = LOCAL_LLAMACPP_ADAPTER_VERSION
        llama_version = _provider_package_version("llama-cpp-python")
        if llama_version is not None:
            metadata["llama_cpp_python_version"] = llama_version
        lc_version = _provider_package_version("langchain-community")
        if lc_version is not None:
            metadata["langchain_community_version"] = lc_version
    elif provider == "Dummy":
        pass
    return metadata


def effective_sampling(
    *,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    seed: int | None,
) -> dict[str, Any]:
    sampling: dict[str, Any] = {}
    if temperature is not None:
        sampling["temperature"] = float(temperature)
    if top_p is not None:
        sampling["top_p"] = float(top_p)
    if top_k is not None:
        sampling["top_k"] = int(top_k)
    if seed is not None:
        sampling["seed"] = int(seed)
    return sampling


def _vllm_role(message: Any) -> str:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    if isinstance(message, dict):
        role = str(message.get("role", "user"))
    elif hasattr(message, "type"):
        role = str(message.type)
    elif isinstance(message, tuple) and message:
        role = str(message[0])
    else:
        role = "user"
    return role_map.get(role, role)


def _vllm_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content", "")
    if isinstance(message, tuple) and len(message) > 1:
        return message[1]
    return getattr(message, "content", "")


def vllm_input_to_messages(input_item: Any) -> list[dict[str, Any]]:
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    if hasattr(input_item, "to_messages"):
        lc_messages = input_item.to_messages()
        return [
            {"role": role_map.get(msg.type, msg.type), "content": msg.content}
            for msg in lc_messages
        ]
    if isinstance(input_item, list) and input_item and isinstance(input_item[0], tuple):
        return [
            {"role": role_map.get(role, role), "content": content}
            for role, content in input_item
        ]
    if isinstance(input_item, list) and input_item and isinstance(input_item[0], dict):
        return [
            {
                **message,
                "role": role_map.get(
                    message.get("role") or "user", message.get("role") or "user"
                ),
            }
            for message in input_item
        ]
    if isinstance(input_item, str):
        return [{"role": "user", "content": input_item}]
    raise ValueError(f"Unsupported input type: {type(input_item)}")


def vllm_input_to_raw_text(input_item: Any) -> str:
    if isinstance(input_item, str):
        return input_item
    if hasattr(input_item, "to_string"):
        return input_item.to_string()
    if isinstance(input_item, list) and input_item and isinstance(input_item[0], dict):
        return "\n".join(str(msg["content"]) for msg in input_item)
    raise ValueError(f"Cannot extract raw text from: {type(input_item)}")


def canonicalize_vllm_input(item: Any, *, input_mode: str) -> str:
    if input_mode == "raw":
        payload = {"kind": "raw", "text": vllm_input_to_raw_text(item)}
    else:
        if isinstance(item, str):
            messages = [{"role": "user", "content": item}]
        else:
            messages = vllm_input_to_messages(item)
        payload = {"kind": "chat", "messages": messages}
    return stable_json_dumps(payload)


def _canonicalize_message_field(value: Any) -> Any | None:
    return normalize_descriptor_value(value)


def _canonicalize_hosted_message(message: Any) -> dict[str, Any] | None:
    if isinstance(message, dict):
        raw = {
            key: value
            for key, value in message.items()
            if key not in _TRANSIENT_MESSAGE_KEYS
        }
        role = raw.pop("role", "user")
        if role == "human":
            role = "user"
        entry: dict[str, Any] = {"role": role}
        for key, value in raw.items():
            normalized = _canonicalize_message_field(value)
            if normalized is None and value is not None:
                return None
            if normalized is not None:
                entry[key] = normalized
        return entry

    entry: dict[str, Any] = {"role": _vllm_role(message)}
    for field in ("content", "name", "additional_kwargs", "tool_calls", "tool_call_id"):
        if hasattr(message, field):
            value = getattr(message, field)
            if value in (None, {}, []):
                continue
            normalized = _canonicalize_message_field(value)
            if normalized is None:
                return None
            entry[field] = normalized
    return entry


def canonicalize_hosted_chat_input(item: Any) -> str:
    if isinstance(item, str):
        messages = [{"role": "user", "content": item}]
    elif hasattr(item, "to_messages"):
        messages = []
        for message in item.to_messages():
            canonical = _canonicalize_hosted_message(message)
            if canonical is None:
                raise ValueError(f"Unsupported hosted chat message: {type(message)!r}")
            messages.append(canonical)
    elif isinstance(item, list) and item:
        messages = []
        for message in item:
            canonical = _canonicalize_hosted_message(message)
            if canonical is None:
                raise ValueError(f"Unsupported hosted chat message: {type(message)!r}")
            messages.append(canonical)
    else:
        raise ValueError(f"Unsupported hosted chat input type: {type(item)!r}")
    return stable_json_dumps({"kind": "chat", "messages": messages})


def canonicalize_raw_input(item: Any) -> str:
    if isinstance(item, str):
        text = item
    elif hasattr(item, "to_string"):
        text = item.to_string()
    else:
        text = vllm_input_to_raw_text(item)
    return stable_json_dumps({"kind": "raw", "text": text})


def canonicalize_dummy_input(item: Any, *, input_mode: str) -> str:
    if input_mode == "raw":
        return canonicalize_raw_input(item)
    return canonicalize_hosted_chat_input(item)


class PreparedModel:
    """Lazy model wrapper with cache descriptors and deferred backend init."""

    def __init__(
        self,
        *,
        provider: str,
        model_spec: str,
        model_name: str,
        max_tokens: int | None,
        engine_kwargs: dict[str, Any],
        sampling: dict[str, Any],
        input_mode: str,
        descriptor: dict[str, Any] | None,
        materialize: Callable[[PreparedModel], Any],
        producer_metadata: dict[str, Any],
        base_url: str | None = None,
        vllm_resolved: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model_spec = model_spec
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.engine_kwargs = dict(engine_kwargs)
        self.sampling = dict(sampling)
        self.input_mode = input_mode
        self._descriptor = descriptor
        self._materialize_fn = materialize
        self._producer_metadata = dict(producer_metadata)
        self.base_url = base_url
        self._vllm_resolved = vllm_resolved
        self._backend: Any | None = None

    def cache_descriptor(self) -> dict[str, Any] | None:
        return None if self._descriptor is None else dict(self._descriptor)

    def producer_metadata(self) -> dict[str, Any]:
        return dict(self._producer_metadata)

    def canonicalize_input(self, item: Any) -> str:
        if self.provider == "VLLM":
            return canonicalize_vllm_input(item, input_mode=self.input_mode)
        if self.provider == "Dummy":
            return canonicalize_dummy_input(item, input_mode=self.input_mode)
        if self.input_mode == "raw":
            return canonicalize_raw_input(item)
        return canonicalize_hosted_chat_input(item)

    def _sync_descriptor_sampling_field(self, field: str, value: Any) -> None:
        if self._descriptor is None:
            return
        self._descriptor.setdefault("sampling", {})[field] = value
        engine_settings = self._descriptor.get("engine_settings")
        if self.provider != "VLLM" and isinstance(engine_settings, dict):
            engine_settings[field] = value

    def set_temperature(self, temperature: float) -> None:
        value = float(temperature)
        self.sampling["temperature"] = value
        self.engine_kwargs["temperature"] = value
        self._sync_descriptor_sampling_field("temperature", value)
        if self._vllm_resolved is not None:
            self._vllm_resolved.sampling_params_kwargs["temperature"] = value
        if self._backend is not None:
            if hasattr(self._backend, "set_temperature"):
                self._backend.set_temperature(value)
            elif hasattr(self._backend, "temperature"):
                self._backend.temperature = value

    def materialize(self) -> Any:
        if self._backend is None:
            self._backend = self._materialize_fn(self)
        return self._backend

    def batch(self, inputs: list, **invoke_kwargs) -> list[str]:
        return self.materialize().batch(inputs, **invoke_kwargs)

    def invoke(self, input_item, **invoke_kwargs) -> str:
        return self.materialize().invoke(input_item, **invoke_kwargs)

    async def ainvoke(self, input_item, **invoke_kwargs):
        return await self.materialize().ainvoke(input_item, **invoke_kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.materialize(), name)


class CachedModelAdapter(PreparedModel):
    """Cache adapter wrapping an already-constructed backend."""

    def __init__(
        self,
        *,
        backend: Any,
        provider: str,
        model_spec: str,
        model_name: str,
        max_tokens: int | None,
        engine_kwargs: dict[str, Any],
        sampling: dict[str, Any],
        input_mode: str,
        descriptor: dict[str, Any] | None,
        producer_metadata: dict[str, Any],
        base_url: str | None = None,
        vllm_resolved: Any | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            model_spec=model_spec,
            model_name=model_name,
            max_tokens=max_tokens,
            engine_kwargs=engine_kwargs,
            sampling=sampling,
            input_mode=input_mode,
            descriptor=descriptor,
            producer_metadata=producer_metadata,
            base_url=base_url,
            vllm_resolved=vllm_resolved,
            materialize=lambda prepared: backend,
        )
        self._backend = backend


def _warn_uncached_model_type(chat_model: Any) -> None:
    model_type = type(chat_model)
    if model_type in _UNCACHED_MODEL_WARNINGS:
        return
    warnings.warn(
        f"Inference cache skipped for unsupported model type {model_type.__name__}; "
        "running uncached.",
        stacklevel=3,
    )
    _UNCACHED_MODEL_WARNINGS.add(model_type)


def resolve_hosted_base_url(settings: dict[str, Any]) -> str | None:
    """Return the hosted endpoint URL from constructor settings."""
    for key in ("base_url", "openai_api_base"):
        value = settings.get(key)
        if value:
            return str(value)
    return None


def hosted_provider_for_endpoint(
    base_url: str | None,
    default_provider: str,
) -> str:
    """Return OpenRouter when the endpoint targets OpenRouter."""
    if base_url and "openrouter.ai" in base_url:
        return "OpenRouter"
    return default_provider


def _extract_langchain_constructor_settings(model: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    model_fields = getattr(type(model), "model_fields", {})
    fields_set = getattr(model, "model_fields_set", None)
    fields_to_capture = fields_set if isinstance(fields_set, set) else model_fields
    canonical_names = {
        "model_name": "model",
        "openai_api_base": "base_url",
    }
    for field in fields_to_capture:
        if field in _DESCRIPTOR_KEY_DENYLIST or _is_secret_key(field):
            continue
        value = getattr(model, field, None)
        if value is None:
            continue
        if field == "model_kwargs" and not value:
            continue
        settings[canonical_names.get(field, field)] = value
    model_kwargs = getattr(model, "model_kwargs", None)
    if isinstance(model_kwargs, dict) and model_kwargs:
        settings["model_kwargs"] = dict(model_kwargs)
    return settings


def _optional_max_tokens(model: Any) -> int | None:
    value = getattr(model, "max_tokens", None)
    return None if value is None else int(value)


def adapt_dummy_backend(
    model: Any, *, model_spec: str | None = None
) -> PreparedModel | None:
    """Build a cache adapter for a constructed DummyModel backend."""
    name = getattr(model, "name", None)
    init_kwargs = getattr(model, "init_kwargs", None)
    if name is None or init_kwargs is None:
        return None
    provider = "Dummy"
    spec = model_spec or name
    _provider, _, model_name = spec.partition("/")
    settings = dict(init_kwargs)
    sampling = effective_sampling(
        temperature=settings.get("temperature"),
        top_p=settings.get("top_p"),
        top_k=top_k_from_settings(settings),
        seed=settings.get("seed"),
    )
    descriptor = build_dummy_descriptor(
        model_spec=spec,
        max_tokens=int(settings.get("max_tokens", 8192)),
        input_mode="chat",
        constructor_settings=settings,
        resolved_sampling=sampling,
    )
    return CachedModelAdapter(
        backend=model,
        provider=provider,
        model_spec=spec,
        model_name=model_name or spec,
        max_tokens=int(settings.get("max_tokens", 8192)),
        engine_kwargs=settings,
        sampling=sampling,
        input_mode="chat",
        descriptor=descriptor,
        producer_metadata=build_producer_metadata(provider=provider),
    )


def adapt_vllm_backend(
    model: Any, *, model_spec: str | None = None
) -> PreparedModel | None:
    """Build a cache adapter for a constructed ChatVLLM backend."""
    model_path = getattr(model, "model_path", None)
    resolved = getattr(model, "_resolved", None)
    max_tokens = getattr(model, "max_tokens", None)
    if model_path is None or resolved is None or max_tokens is None:
        return None
    spec = model_spec or f"VLLM/{model_path}"
    descriptor = build_vllm_descriptor(spec, resolved)
    if descriptor is None:
        return None
    return CachedModelAdapter(
        backend=model,
        provider="VLLM",
        model_spec=spec,
        model_name=model_path,
        max_tokens=max_tokens,
        engine_kwargs=dict(resolved.vllm_kwargs),
        sampling=dict(resolved.sampling_params_kwargs),
        input_mode=resolved.input_mode,
        descriptor=descriptor,
        producer_metadata=build_producer_metadata(provider="VLLM"),
        vllm_resolved=resolved,
    )


def wrap_known_model(
    model: Any, *, model_spec: str | None = None
) -> PreparedModel | None:
    """Return a cache adapter for a known constructed backend, or None."""
    if isinstance(model, PreparedModel):
        return model

    to_adapter = getattr(model, "to_prepared_cache_adapter", None)
    if callable(to_adapter):
        return to_adapter(model_spec=model_spec)

    if isinstance(model, ChatOpenAI):
        settings = _extract_langchain_constructor_settings(model)
        base_url = resolve_hosted_base_url(settings)
        provider = hosted_provider_for_endpoint(base_url, "ChatOpenAI")
        spec = model_spec or f"{provider}/{model.model_name}"
        max_tokens = _optional_max_tokens(model)
        sampling = effective_sampling(
            temperature=settings.get("temperature"),
            top_p=settings.get("top_p"),
            top_k=top_k_from_settings(settings),
            seed=settings.get("seed"),
        )
        descriptor = build_hosted_descriptor(
            provider=provider,
            model_spec=spec,
            model_name=model.model_name,
            max_tokens=max_tokens,
            input_mode="chat",
            base_url=base_url,
            constructor_settings=settings,
            resolved_sampling=sampling,
        )
        return CachedModelAdapter(
            backend=model,
            provider=provider,
            model_spec=spec,
            model_name=model.model_name,
            max_tokens=max_tokens,
            engine_kwargs=settings,
            sampling=sampling,
            input_mode="chat",
            descriptor=descriptor,
            producer_metadata=build_producer_metadata(provider=provider),
            base_url=base_url,
        )

    if isinstance(model, OpenAI):
        spec = model_spec or f"OpenAI/{model.model_name}"
        settings = _extract_langchain_constructor_settings(model)
        base_url = resolve_hosted_base_url(settings)
        max_tokens = _optional_max_tokens(model)
        sampling = effective_sampling(
            temperature=settings.get("temperature"),
            top_p=settings.get("top_p"),
            top_k=None,
            seed=settings.get("seed"),
        )
        descriptor = build_hosted_descriptor(
            provider="OpenAI",
            model_spec=spec,
            model_name=model.model_name,
            max_tokens=max_tokens,
            input_mode="raw",
            base_url=base_url,
            constructor_settings=settings,
            resolved_sampling=sampling,
        )
        return CachedModelAdapter(
            backend=model,
            provider="OpenAI",
            model_spec=spec,
            model_name=model.model_name,
            max_tokens=max_tokens,
            engine_kwargs=settings,
            sampling=sampling,
            input_mode="raw",
            descriptor=descriptor,
            producer_metadata=build_producer_metadata(provider="OpenAI"),
            base_url=base_url,
        )

    if isinstance(model, Together):
        spec = model_spec or f"Together/{model.model}"
        settings = _extract_langchain_constructor_settings(model)
        base_url = resolve_hosted_base_url(settings)
        max_tokens = _optional_max_tokens(model)
        sampling = effective_sampling(
            temperature=settings.get("temperature"),
            top_p=settings.get("top_p"),
            top_k=top_k_from_settings(settings),
            seed=None,
        )
        descriptor = build_hosted_descriptor(
            provider="Together",
            model_spec=spec,
            model_name=model.model,
            max_tokens=max_tokens,
            input_mode="raw",
            base_url=base_url,
            constructor_settings=settings,
            resolved_sampling=sampling,
        )
        return CachedModelAdapter(
            backend=model,
            provider="Together",
            model_spec=spec,
            model_name=model.model,
            max_tokens=max_tokens,
            engine_kwargs=settings,
            sampling=sampling,
            input_mode="raw",
            descriptor=descriptor,
            producer_metadata=build_producer_metadata(provider="Together"),
            base_url=base_url,
        )

    if isinstance(model, LlamaCpp):
        spec = model_spec or f"LlamaCpp/{getattr(model, 'model_path', 'unknown')}"
        settings = _extract_langchain_constructor_settings(model)
        sampling = effective_sampling(
            temperature=settings.get("temperature"),
            top_p=settings.get("top_p"),
            top_k=settings.get("top_k"),
            seed=settings.get("seed"),
        )
        descriptor = build_llamacpp_descriptor(
            model_spec=spec,
            model_name=str(getattr(model, "model_path", "unknown")),
            max_tokens=int(settings.get("max_tokens", 8192) or 8192),
            constructor_settings=settings,
            resolved_sampling=sampling,
        )
        return CachedModelAdapter(
            backend=model,
            provider="LlamaCpp",
            model_spec=spec,
            model_name=str(getattr(model, "model_path", "unknown")),
            max_tokens=int(settings.get("max_tokens", 8192) or 8192),
            engine_kwargs=settings,
            sampling=sampling,
            input_mode="raw",
            descriptor=descriptor,
            producer_metadata=build_producer_metadata(provider="LlamaCpp"),
        )

    return None


def resolve_cacheable_model(
    chat_model: Any,
    *,
    model_spec: str | None = None,
) -> PreparedModel | None:
    wrapped = wrap_known_model(chat_model, model_spec=model_spec)
    if wrapped is not None:
        return wrapped
    _warn_uncached_model_type(chat_model)
    return None
