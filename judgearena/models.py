"""Model/inference layer: provider wrappers, the vLLM engine, and batched inference."""

from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from langchain_openai.llms import OpenAI
from langchain_together.llms import Together
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from judgearena.constants import VLLM_REASONING_END_STR, VLLM_REASONING_START_STR
from judgearena.log import get_logger
from judgearena.model_adapters import (
    DESCRIPTOR_SCHEMA_VERSION,
    HOSTED_ADAPTER_VERSION,
    PreparedModel,
    adapt_dummy_backend,
    adapt_vllm_backend,
    build_dummy_descriptor,
    build_hosted_descriptor,
    build_llamacpp_descriptor,
    build_producer_metadata,
    build_vllm_descriptor,
    effective_sampling,
    hosted_provider_for_endpoint,
    resolve_cacheable_model,
    resolve_hosted_base_url,
    top_k_from_settings,
    vllm_input_to_messages,
    vllm_input_to_raw_text,
)
from judgearena.utils.io import safe_parse_int

logger = get_logger(__name__)

DEFAULT_VLLM_JUDGE_THINKING_TOKEN_BUDGET = 512
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_THINKING_MODEL_PARSER_BY_SUBSTRING = (
    ("qwen3", "qwen3"),
    ("smollm3", "qwen3"),
    ("olmo-3-7b-think", "olmo3"),
)

_VLLM_INIT_RETRY_SIGNATURES = (
    "cudaErrorDevicesUnavailable",
    "CUDA-capable device(s) is/are busy or unavailable",
    "CUDA error: initialization error",
)
_VLLM_INIT_MAX_ATTEMPTS = safe_parse_int("JUDGEARENA_VLLM_INIT_MAX_ATTEMPTS") or 4
_VLLM_INIT_BACKOFF_SECONDS = (
    safe_parse_int("JUDGEARENA_VLLM_INIT_BACKOFF_SECONDS") or 20
)


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    provider, sep, model_name = model_spec.partition("/")
    if not sep:
        return model_spec, ""
    return provider, model_name


def is_thinking_model(model_name: str) -> bool:
    """Return True for reasoning models that emit thinking traces."""
    return _default_reasoning_parser_for_model(model_name) is not None


def _default_reasoning_parser_for_model(model_name: str) -> str | None:
    lowered = model_name.lower()
    for token, reasoning_parser in _THINKING_MODEL_PARSER_BY_SUBSTRING:
        if token in lowered:
            return reasoning_parser
    return None


def build_default_judge_model_kwargs(
    judge_model: str,
    engine_kwargs: dict[str, object],
    *,
    judge_engine_kwargs_override: dict[str, object] | None = None,
) -> dict[str, object]:
    """Copy judge engine kwargs and add supported built-in defaults."""
    provider, model_name = _split_model_spec(judge_model)
    judge_model_kwargs = dict(engine_kwargs) if provider == "VLLM" else {}
    if judge_engine_kwargs_override:
        judge_model_kwargs.update(judge_engine_kwargs_override)
    if provider == "VLLM":
        if "thinking_token_budget" not in judge_model_kwargs and is_thinking_model(
            model_name
        ):
            judge_model_kwargs["thinking_token_budget"] = (
                DEFAULT_VLLM_JUDGE_THINKING_TOKEN_BUDGET
            )
        if "kv_cache_dtype" not in judge_model_kwargs and "fp8" in model_name.lower():
            judge_model_kwargs["kv_cache_dtype"] = "fp8"
    return judge_model_kwargs


def _resolve_chat_template_kwargs(
    *,
    explicit_chat_template_kwargs: dict[str, object] | None,
    disable_thinking: bool,
) -> dict[str, object] | None:
    chat_template_kwargs = dict(explicit_chat_template_kwargs or {})
    if disable_thinking and "enable_thinking" not in chat_template_kwargs:
        chat_template_kwargs["enable_thinking"] = False
    return chat_template_kwargs or None


def _hf_from_pretrained_kwargs(engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(engine_kwargs.get("trust_remote_code", True)),
    }
    if engine_kwargs.get("revision") is not None:
        kwargs["revision"] = engine_kwargs["revision"]
    tokenizer_revision = engine_kwargs.get("tokenizer_revision")
    if tokenizer_revision is not None:
        kwargs["tokenizer_revision"] = tokenizer_revision
    return kwargs


def _is_retryable_error(e: Exception) -> bool:
    _RETRYABLE_CODES = {408, 429, 502, 503, 504}
    if isinstance(e, ValueError) and e.args:
        arg = e.args[0]
        if isinstance(arg, dict) and arg.get("code") in _RETRYABLE_CODES:
            return True
    if isinstance(e, json.JSONDecodeError):
        return True
    error_str = str(e)
    return (
        any(str(code) in error_str for code in _RETRYABLE_CODES)
        or "rate" in error_str.lower()
        or "Expecting value" in error_str
        or "JSONDecodeError" in error_str
    )


def _init_llm_with_retry(llm_cls, **kwargs):
    """Instantiate ``vllm.LLM`` with retries on transient GPU-init races."""
    last_exc: Exception | None = None
    for attempt in range(1, _VLLM_INIT_MAX_ATTEMPTS + 1):
        try:
            return llm_cls(**kwargs)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if not any(sig in message for sig in _VLLM_INIT_RETRY_SIGNATURES):
                raise
            last_exc = exc
            if attempt == _VLLM_INIT_MAX_ATTEMPTS:
                break
            delay = _VLLM_INIT_BACKOFF_SECONDS * (2 ** (attempt - 1))
            warnings.warn(
                f"vLLM init attempt {attempt}/{_VLLM_INIT_MAX_ATTEMPTS} failed "
                f"with transient GPU-init signature ({message.splitlines()[0]}); "
                f"sleeping {delay}s before retry.",
                RuntimeWarning,
                stacklevel=2,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


class DummyModel:
    def __init__(self, name: str, **init_kwargs):
        self.name = name
        self.init_kwargs = dict(init_kwargs)
        self.message = "/".join(name.split("/")[1:])

    def batch(self, inputs, **invoke_kwargs) -> list[str]:
        return [self.message] * len(inputs)

    def invoke(self, input, **invoke_kwargs) -> str:
        return self.message

    async def ainvoke(self, input, **invoke_kwargs):
        return self.message

    def set_temperature(self, temperature: float) -> None:
        self.init_kwargs["temperature"] = float(temperature)

    def to_prepared_cache_adapter(
        self,
        *,
        model_spec: str | None = None,
    ) -> PreparedModel | None:
        """Return a cache adapter for this constructed Dummy backend."""
        return adapt_dummy_backend(self, model_spec=model_spec)


@dataclass
class VLLMResolvedSettings:
    """Shared, lazy-resolved vLLM configuration for descriptors and ChatVLLM."""

    model_path: str
    max_tokens: int
    chat_template: str | None
    chat_template_kwargs: dict[str, object] | None
    use_generate: bool
    input_mode: str
    sampling_params_kwargs: dict[str, Any]
    vllm_kwargs: dict[str, Any]
    explicit_reasoning_settings: bool = False


def resolve_vllm_settings(
    model: str,
    max_tokens: int = 8192,
    chat_template: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    **vllm_kwargs: Any,
) -> VLLMResolvedSettings:
    """Resolve vLLM chat/raw mode and sampling without loading ``vllm.LLM``."""
    from vllm.config.reasoning import ReasoningConfig

    engine_kwargs = dict(vllm_kwargs)
    disable_thinking = bool(engine_kwargs.pop("disable_thinking", False))
    thinking_token_budget = engine_kwargs.pop("thinking_token_budget", None)
    explicit_chat_template_kwargs = engine_kwargs.pop("chat_template_kwargs", None)
    explicit_reasoning_settings = (
        "reasoning_parser" in engine_kwargs or "reasoning_config" in engine_kwargs
    )
    chat_template_kwargs = _resolve_chat_template_kwargs(
        explicit_chat_template_kwargs=explicit_chat_template_kwargs,
        disable_thinking=disable_thinking,
    )
    hf_kwargs = _hf_from_pretrained_kwargs(engine_kwargs)
    config_hf_kwargs = {
        key: hf_kwargs[key]
        for key in ("trust_remote_code", "revision")
        if key in hf_kwargs
    }
    tokenizer_hf_kwargs = dict(config_hf_kwargs)
    if "tokenizer_revision" in hf_kwargs:
        tokenizer_hf_kwargs["revision"] = hf_kwargs["tokenizer_revision"]

    max_model_len = engine_kwargs.get("max_model_len")
    if max_model_len is not None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model, **config_hf_kwargs)
            model_max_pos = getattr(config, "max_position_embeddings", None)
            if model_max_pos is not None and max_model_len > model_max_pos:
                warnings.warn(
                    f"Capping max_model_len from {max_model_len} to "
                    f"{model_max_pos} (max_position_embeddings) for '{model}'.",
                    stacklevel=2,
                )
                engine_kwargs["max_model_len"] = model_max_pos
        except Exception as exc:
            warnings.warn(
                "Could not validate max_model_len against "
                f"max_position_embeddings for '{model}': {exc}. "
                "Proceeding without clamping; vLLM may raise if the value is too large.",
                RuntimeWarning,
                stacklevel=2,
            )

    if seed is not None:
        engine_kwargs.setdefault("seed", int(seed))

    sampling_params_kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": 0.6 if temperature is None else float(temperature),
        "top_p": 0.95 if top_p is None else float(top_p),
    }
    if top_k is not None:
        sampling_params_kwargs["top_k"] = int(top_k)
    if seed is not None:
        sampling_params_kwargs["seed"] = int(seed)

    if thinking_token_budget is not None:
        if max_tokens is not None:
            thinking_token_budget = min(int(thinking_token_budget), int(max_tokens))
        if explicit_reasoning_settings:
            sampling_params_kwargs["thinking_token_budget"] = int(thinking_token_budget)
        elif is_thinking_model(model):
            reasoning_parser = _default_reasoning_parser_for_model(model)
            assert reasoning_parser is not None
            engine_kwargs.setdefault(
                "reasoning_config",
                ReasoningConfig(
                    reasoning_start_str=VLLM_REASONING_START_STR,
                    reasoning_end_str=VLLM_REASONING_END_STR,
                ),
            )
            engine_kwargs.setdefault("reasoning_parser", reasoning_parser)
            sampling_params_kwargs["thinking_token_budget"] = int(thinking_token_budget)
        else:
            warnings.warn(
                f"Model '{model}' is not in JudgeArena's built-in thinking-model "
                "defaults (Qwen3/SmolLM3/Olmo-3-7B-Think). Ignoring "
                "thinking_token_budget unless reasoning_parser or "
                "reasoning_config is provided explicitly.",
                stacklevel=2,
            )

    if chat_template:
        use_generate = False
        input_mode = "chat"
        effective_chat_template: str | None = chat_template
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer_id = engine_kwargs.get("tokenizer", model)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                **tokenizer_hf_kwargs,
            )
            has_template = bool(getattr(tokenizer, "chat_template", None))
        except Exception as exc:
            raise RuntimeError(
                f"Could not resolve chat template metadata for '{model}': {exc}"
            ) from exc
        if not has_template:
            warnings.warn(
                f"Model '{model}' tokenizer does not define a chat template. "
                "Falling back to llm.generate() (no chat formatting). "
                "Override with --chat_template if this model needs one.",
                stacklevel=2,
            )
            use_generate = True
            input_mode = "raw"
            effective_chat_template = None
            if disable_thinking:
                warnings.warn(
                    f"Model '{model}' has no chat template, so disable_thinking "
                    "cannot be applied when falling back to llm.generate().",
                    stacklevel=2,
                )
        else:
            use_generate = False
            input_mode = "chat"
            effective_chat_template = None

    return VLLMResolvedSettings(
        model_path=model,
        max_tokens=max_tokens,
        chat_template=effective_chat_template,
        chat_template_kwargs=chat_template_kwargs,
        use_generate=use_generate,
        input_mode=input_mode,
        sampling_params_kwargs=sampling_params_kwargs,
        vllm_kwargs=engine_kwargs,
        explicit_reasoning_settings=explicit_reasoning_settings,
    )


class ChatVLLM:
    """VLLM wrapper that auto-detects whether to use chat() or generate()."""

    def __init__(
        self,
        model: str,
        max_tokens: int = 8192,
        chat_template: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        *,
        _resolved: VLLMResolvedSettings | None = None,
        **vllm_kwargs,
    ):
        from vllm import LLM, SamplingParams

        if _resolved is None:
            _resolved = resolve_vllm_settings(
                model,
                max_tokens=max_tokens,
                chat_template=chat_template,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                **vllm_kwargs,
            )

        self._resolved = _resolved
        self.model_path = _resolved.model_path
        self.max_tokens = _resolved.max_tokens
        self._chat_template_kwargs = _resolved.chat_template_kwargs
        self._sampling_params_kwargs = dict(_resolved.sampling_params_kwargs)
        self.sampling_params = SamplingParams(**self._sampling_params_kwargs)
        self.chat_template = _resolved.chat_template
        self._use_generate = _resolved.use_generate

        llm_init_kwargs = dict(_resolved.vllm_kwargs)
        trust_remote_code = bool(llm_init_kwargs.pop("trust_remote_code", True))
        self.llm = _init_llm_with_retry(
            LLM,
            model=_resolved.model_path,
            trust_remote_code=trust_remote_code,
            **llm_init_kwargs,
        )
        self.tokenizer = self.llm.get_tokenizer()

        if self.chat_template:
            logger.info(
                "ChatVLLM: using explicit chat template for '%s'",
                _resolved.model_path,
            )
        elif self._use_generate:
            logger.info(
                "ChatVLLM: falling back to llm.generate() for '%s'",
                _resolved.model_path,
            )
        else:
            logger.info(
                "ChatVLLM: using tokenizer's chat template for '%s'",
                _resolved.model_path,
            )

    def set_temperature(self, temperature: float) -> None:
        from vllm import SamplingParams

        self._sampling_params_kwargs["temperature"] = float(temperature)
        self.sampling_params = SamplingParams(**self._sampling_params_kwargs)
        self._resolved.sampling_params_kwargs["temperature"] = float(temperature)

    def _to_messages(self, input_item) -> list[dict]:
        return vllm_input_to_messages(input_item)

    def _to_raw_text(self, input_item) -> str:
        return vllm_input_to_raw_text(input_item)

    def _run_raw_batch(self, inputs: list):
        if self._use_generate:
            prompts = [self._to_raw_text(inp) for inp in inputs]
            outputs = self.llm.generate(prompts, self.sampling_params)
        else:
            messages_batch = [self._to_messages(inp) for inp in inputs]
            outputs = self.llm.chat(
                messages_batch,
                self.sampling_params,
                add_generation_prompt=True,
                chat_template=self.chat_template,
                chat_template_kwargs=self._chat_template_kwargs,
            )
        return outputs

    def batch(self, inputs: list, **invoke_kwargs) -> list[str]:
        outputs = self._run_raw_batch(inputs)
        return [out.outputs[0].text for out in outputs]

    def invoke(self, input_item, **invoke_kwargs) -> str:
        return self.batch([input_item], **invoke_kwargs)[0]

    async def ainvoke(self, input_item, **invoke_kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.invoke(input_item, **invoke_kwargs)
        )

    def to_prepared_cache_adapter(
        self,
        *,
        model_spec: str | None = None,
    ) -> PreparedModel | None:
        """Return a cache adapter for this constructed ChatVLLM backend."""
        return adapt_vllm_backend(self, model_spec=model_spec)


def _normalize_inference_outputs(results: list[Any]) -> list[str]:
    return [x.content if hasattr(x, "content") else x for x in results]


def _do_inference_uncached(
    chat_model: Any,
    inputs: Sequence[Any],
    *,
    use_tqdm: bool = False,
) -> list[str]:
    invoke_kwargs: dict[str, Any] = {}
    if use_tqdm:
        cap = safe_parse_int("JUDGEARENA_JUDGE_MAX_CONCURRENCY")
        cap = cap if cap and cap > 0 else None

        async def process_with_real_progress(model, batch_inputs, pbar):
            sem = asyncio.Semaphore(cap) if cap else None

            async def process_single(input_item, max_retries=5, base_delay=1.0):
                for attempt in range(max_retries):
                    try:
                        result = await model.ainvoke(input_item, **invoke_kwargs)
                        pbar.update(1)
                        return result
                    except Exception as exc:
                        if attempt == max_retries - 1 or not _is_retryable_error(exc):
                            raise
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "Retry because of a server error, %d/%d: %s. Waiting %ss...",
                            attempt + 1,
                            max_retries,
                            exc,
                            delay,
                        )
                        await asyncio.sleep(delay)

            async def gated(inp):
                if sem is None:
                    return await process_single(inp)
                async with sem:
                    return await process_single(inp)

            return await asyncio.gather(*[gated(inp) for inp in batch_inputs])

        with logging_redirect_tqdm(), tqdm(total=len(inputs)) as pbar:
            res = asyncio.run(
                process_with_real_progress(
                    chat_model=chat_model, batch_inputs=inputs, pbar=pbar
                )
            )
    else:

        def batch_with_retry(batch_inputs, max_retries=5, base_delay=1.0):
            for attempt in range(max_retries):
                num_chunks = 4**attempt
                chunk_size = max(1, len(batch_inputs) // num_chunks)
                chunks = [
                    batch_inputs[i : i + chunk_size]
                    for i in range(0, len(batch_inputs), chunk_size)
                ]
                try:
                    results = []
                    for chunk in chunks:
                        results.extend(chat_model.batch(inputs=chunk, **invoke_kwargs))
                    return results
                except Exception as exc:
                    if attempt == max_retries - 1 or not _is_retryable_error(exc):
                        raise
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Retry because of a server error, %d/%d: %s. Waiting %ss...",
                        attempt + 1,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            raise RuntimeError("batch_with_retry exhausted retries without returning")

        res = batch_with_retry(inputs)

    return _normalize_inference_outputs(res)


def do_inference(
    chat_model,
    inputs,
    use_tqdm: bool = False,
    *,
    cache: Any | None = None,
    cache_meta: dict[str, Any] | None = None,
):
    """Run inference over *inputs*, optionally via the unified inference cache."""
    if not inputs:
        return []

    if cache is not None and getattr(cache, "mode", None) == "off":
        return _do_inference_uncached(chat_model, inputs, use_tqdm=use_tqdm)

    if cache is None:
        return _do_inference_uncached(chat_model, inputs, use_tqdm=use_tqdm)

    cacheable = resolve_cacheable_model(chat_model)
    if cacheable is None:
        return _do_inference_uncached(chat_model, inputs, use_tqdm=use_tqdm)

    descriptor = cacheable.cache_descriptor()
    if descriptor is None:
        warnings.warn(
            f"Inference cache skipped for {cacheable.model_spec}; descriptor is unsafe.",
            stacklevel=2,
        )
        return _do_inference_uncached(chat_model, inputs, use_tqdm=use_tqdm)

    cache_meta = cache_meta or {}
    row_metadata = cache_meta.get("metadata")
    if row_metadata is not None and len(row_metadata) != len(inputs):
        raise ValueError("cache_meta['metadata'] length must match inputs")

    canonical_inputs = [cacheable.canonicalize_input(item) for item in inputs]

    def miss_runner(miss_inputs: list[Any]) -> list[str]:
        return _do_inference_uncached(cacheable, miss_inputs, use_tqdm=use_tqdm)

    return cache.get_or_run(
        model_spec=cacheable.model_spec,
        descriptor=descriptor,
        canonical_inputs=canonical_inputs,
        original_inputs=list(inputs),
        miss_runner=miss_runner,
        row_metadata=row_metadata,
        producer_metadata=cacheable.producer_metadata(),
    )


def _route_sampling_params(
    engine_kwargs: dict,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
    supported_fields: set[str] | None = None,
    top_k_via_model_kwargs: bool = False,
    provider: str = "",
) -> dict:
    for key, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("seed", seed),
        ("top_k", top_k),
    ):
        if value is None:
            continue
        if key == "top_k" and top_k_via_model_kwargs:
            engine_kwargs.setdefault("model_kwargs", {})["top_k"] = value
            continue
        if supported_fields is not None and key not in supported_fields:
            logger.warning(
                "%s backend does not support sampling param %r; dropping it.",
                provider or "This",
                key,
            )
            continue
        engine_kwargs[key] = value
    return engine_kwargs


def _provider_model_class(model_provider: str):
    model_classes = [LlamaCpp, ChatOpenAI, Together, OpenAI]
    model_cls_dict = {model_cls.__name__: model_cls for model_cls in model_classes}
    assert model_provider in model_cls_dict, (
        f"{model_provider} not available, choose among {list(model_cls_dict.keys())}"
    )
    return model_cls_dict[model_provider]


def _vllm_constructor_kwargs(raw_engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw_engine_kwargs.items()
        if key
        not in {
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "chat_template",
        }
    }


def make_model(
    model: str, max_tokens: int | None = 8192, **engine_kwargs
) -> PreparedModel:
    """Prepare a lazy model wrapper from a provider/model-name string."""
    engine_kwargs = engine_kwargs.copy()
    resolved_max_tokens = max_tokens or 8192
    engine_kwargs["max_tokens"] = resolved_max_tokens

    temperature = engine_kwargs.pop("temperature", None)
    top_p = engine_kwargs.pop("top_p", None)
    top_k = engine_kwargs.pop("top_k", None)
    seed = engine_kwargs.pop("seed", None)

    model_provider, model_name = _split_model_spec(model)
    raw_engine_kwargs = dict(engine_kwargs)

    if model_provider == "Dummy":
        dummy_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        _route_sampling_params(
            dummy_kwargs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        sampling = effective_sampling(
            temperature=dummy_kwargs.get("temperature"),
            top_p=dummy_kwargs.get("top_p"),
            top_k=dummy_kwargs.get("top_k"),
            seed=dummy_kwargs.get("seed"),
        )
        descriptor = build_dummy_descriptor(
            model_spec=model,
            max_tokens=resolved_max_tokens,
            input_mode="chat",
            constructor_settings=dummy_kwargs,
            resolved_sampling=sampling,
        )
        return PreparedModel(
            provider=model_provider,
            model_spec=model,
            model_name=model_name,
            max_tokens=resolved_max_tokens,
            engine_kwargs=dummy_kwargs,
            sampling=sampling,
            input_mode="chat",
            descriptor=descriptor,
            producer_metadata=build_producer_metadata(provider=model_provider),
            materialize=lambda prepared: DummyModel(
                prepared.model_spec,
                **prepared.engine_kwargs,
            ),
        )

    logger.info("Preparing %s(model=%s)", model_provider, model_name)

    if model_provider == "VLLM":
        vllm_kwargs = {k: v for k, v in raw_engine_kwargs.items() if v is not None}
        vllm_kwargs["chat_template"] = vllm_kwargs.get("chat_template", None)
        _route_sampling_params(
            vllm_kwargs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
        )
        resolved = resolve_vllm_settings(
            model_name,
            max_tokens=resolved_max_tokens,
            chat_template=vllm_kwargs.get("chat_template"),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            **_vllm_constructor_kwargs(vllm_kwargs),
        )
        descriptor = build_vllm_descriptor(model, resolved)
        sampling = dict(resolved.sampling_params_kwargs)

        def materialize_vllm(prepared: PreparedModel) -> ChatVLLM:
            assert prepared._vllm_resolved is not None
            kwargs = dict(prepared.engine_kwargs)
            return ChatVLLM(
                model=model_name,
                max_tokens=prepared.max_tokens,
                chat_template=kwargs.get("chat_template"),
                _resolved=prepared._vllm_resolved,
                **_vllm_constructor_kwargs(kwargs),
            )

        return PreparedModel(
            provider=model_provider,
            model_spec=model,
            model_name=model_name,
            max_tokens=resolved_max_tokens,
            engine_kwargs=vllm_kwargs,
            sampling=sampling,
            input_mode=resolved.input_mode,
            descriptor=descriptor,
            producer_metadata=build_producer_metadata(provider=model_provider),
            materialize=materialize_vllm,
            vllm_resolved=resolved,
        )

    hosted_kwargs = dict(engine_kwargs)
    if model_provider != "VLLM":
        for key in (
            "max_model_len",
            "chat_template",
            "language_model_only",
            "gpu_memory_utilization",
            "enforce_eager",
            "tensor_parallel_size",
            "quantization",
            "kv_cache_dtype",
            "reasoning_parser",
            "reasoning_config",
            "trust_remote_code",
            "disable_thinking",
            "thinking_token_budget",
            "chat_template_kwargs",
            "revision",
            "tokenizer_revision",
        ):
            hosted_kwargs.pop(key, None)

    if model_provider == "OpenRouter":
        _route_sampling_params(
            hosted_kwargs,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            top_k_via_model_kwargs=True,
        )
        hosted_kwargs.setdefault("base_url", OPENROUTER_BASE_URL)
        hosted_kwargs.setdefault("api_key", os.getenv("OPENROUTER_API_KEY"))
        hosted_kwargs.setdefault("model", model_name)
        base_url = str(hosted_kwargs["base_url"])
        sampling = effective_sampling(
            temperature=hosted_kwargs.get("temperature"),
            top_p=hosted_kwargs.get("top_p"),
            top_k=top_k_from_settings(hosted_kwargs),
            seed=hosted_kwargs.get("seed"),
        )
        descriptor = build_hosted_descriptor(
            provider=model_provider,
            model_spec=model,
            model_name=model_name,
            max_tokens=resolved_max_tokens,
            input_mode="chat",
            base_url=base_url,
            constructor_settings=hosted_kwargs,
            resolved_sampling=sampling,
        )
        return PreparedModel(
            provider=model_provider,
            model_spec=model,
            model_name=model_name,
            max_tokens=resolved_max_tokens,
            engine_kwargs=hosted_kwargs,
            sampling=sampling,
            input_mode="chat",
            descriptor=descriptor,
            base_url=base_url,
            producer_metadata=build_producer_metadata(provider=model_provider),
            materialize=lambda prepared: ChatOpenAI(**prepared.engine_kwargs),
        )

    model_cls = _provider_model_class(model_provider)
    provider_kwargs = dict(hosted_kwargs)
    if model_provider == "LlamaCpp":
        provider_kwargs["model_path"] = model_name
        input_mode = "raw"
    else:
        provider_kwargs["model"] = model_name
        input_mode = "raw" if model_provider in {"OpenAI", "Together"} else "chat"

    supported_fields = set(getattr(model_cls, "model_fields", {}))
    _route_sampling_params(
        provider_kwargs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        supported_fields=supported_fields,
        top_k_via_model_kwargs=(
            "top_k" not in supported_fields and "model_kwargs" in supported_fields
        ),
        provider=model_provider,
    )
    sampling = effective_sampling(
        temperature=provider_kwargs.get("temperature"),
        top_p=provider_kwargs.get("top_p"),
        top_k=top_k_from_settings(provider_kwargs),
        seed=provider_kwargs.get("seed"),
    )
    base_url = resolve_hosted_base_url(provider_kwargs)
    if model_cls is ChatOpenAI:
        model_provider = hosted_provider_for_endpoint(base_url, model_provider)
        if model_provider == "OpenRouter":
            model = f"OpenRouter/{model_name}"

    if model_provider == "LlamaCpp":
        descriptor = build_llamacpp_descriptor(
            model_spec=model,
            model_name=model_name,
            max_tokens=resolved_max_tokens,
            constructor_settings=provider_kwargs,
            resolved_sampling=sampling,
        )
    else:
        descriptor = build_hosted_descriptor(
            provider=model_provider,
            model_spec=model,
            model_name=model_name,
            max_tokens=resolved_max_tokens,
            input_mode=input_mode,
            base_url=base_url,
            constructor_settings=provider_kwargs,
            resolved_sampling=sampling,
        )

    return PreparedModel(
        provider=model_provider,
        model_spec=model,
        model_name=model_name,
        max_tokens=resolved_max_tokens,
        engine_kwargs=provider_kwargs,
        sampling=sampling,
        input_mode=input_mode,
        descriptor=descriptor,
        base_url=base_url,
        producer_metadata=build_producer_metadata(provider=model_provider),
        materialize=lambda prepared: model_cls(**prepared.engine_kwargs),
    )


__all__ = [
    "ChatVLLM",
    "DESCRIPTOR_SCHEMA_VERSION",
    "DummyModel",
    "HOSTED_ADAPTER_VERSION",
    "PreparedModel",
    "VLLMResolvedSettings",
    "build_default_judge_model_kwargs",
    "do_inference",
    "is_thinking_model",
    "make_model",
    "resolve_vllm_settings",
]
