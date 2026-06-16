"""Model/inference layer: provider wrappers, the vLLM engine, and batched inference."""

from __future__ import annotations

import asyncio
import os
import time
import warnings

from langchain_community.llms import LlamaCpp
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from judgearena.constants import VLLM_REASONING_END_STR, VLLM_REASONING_START_STR
from judgearena.log import get_logger
from judgearena.utils.io import safe_parse_int

logger = get_logger(__name__)


DEFAULT_VLLM_JUDGE_THINKING_TOKEN_BUDGET = 512
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
    """Return True for reasoning models that emit `<think>...</think>` traces.

    Covers the Qwen3 family (e.g. `Qwen/Qwen3.5-9B`) and SmolLM3 (e.g.
    `HuggingFaceTB/SmolLM3-3B`) plus `allenai/Olmo-3-7B-Think`; all emit
    `<think>`/`</think>` traces so vLLM's budget enforcement and our
    tag-stripping apply uniformly. Matching is case-insensitive to tolerate
    mixed-case HF repo ids like `HuggingFaceTB/SmolLM3-3B`.
    """
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
        # FP8 weights leave little KV headroom on consumer-class GPUs; default
        # to FP8 KV cache so judges like Skywork-70B-FP8 fit comfortably on
        # 2x L40S at 32k context. Explicit caller overrides still win.
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


def _is_retryable_error(e: Exception) -> bool:
    """Return True if the exception is a transient server error that should be retried.

    Handles two formats:
    - String representation contains the HTTP code (most providers)
    - ValueError raised by langchain-openai with a dict arg: {'message': ..., 'code': 429}
    """
    # langchain-openai raises ValueError(response_dict.get("error")) where the
    # error value is a dict like {'message': '...', 'code': 408}
    _RETRYABLE_CODES = {408, 429, 502, 503, 504}
    if isinstance(e, ValueError) and e.args:
        arg = e.args[0]
        if isinstance(arg, dict) and arg.get("code") in _RETRYABLE_CODES:
            return True

    error_str = str(e)
    return (
        any(str(code) in error_str for code in _RETRYABLE_CODES)
        or "rate" in error_str.lower()
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
    def __init__(self, name: str):
        self.name = name
        self.message = "/".join(name.split("/")[1:])

    def batch(self, inputs, **invoke_kwargs) -> list[str]:
        return [self.message] * len(inputs)

    def invoke(self, input, **invoke_kwargs) -> str:
        return self.message

    async def ainvoke(self, input, **invoke_kwargs):
        return self.message


class ChatVLLM:
    """VLLM wrapper that auto-detects whether to use chat() or generate().

    Chat template handling:
        - If ``chat_template`` is explicitly provided, always uses ``llm.chat()``
          with that template (useful for models whose tokenizer lacks a template
          but you know the correct one).
        - If the tokenizer defines a chat template, uses ``llm.chat()`` and lets
          vLLM apply the tokenizer's template automatically.
        - If no chat template is found (typical for base/pretrained models),
          falls back to ``llm.generate()`` and emits a warning.  This avoids the
          ``ValueError`` raised by ``transformers >= v4.44`` which removed the
          default chat template.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 8192,
        chat_template: str | None = None,
        **vllm_kwargs,
    ):
        from vllm import LLM, SamplingParams
        from vllm.config.reasoning import ReasoningConfig

        self.model_path = model
        self.max_tokens = max_tokens
        disable_thinking = bool(vllm_kwargs.pop("disable_thinking", False))
        thinking_token_budget = vllm_kwargs.pop("thinking_token_budget", None)
        explicit_chat_template_kwargs = vllm_kwargs.pop("chat_template_kwargs", None)
        explicit_reasoning_settings = (
            "reasoning_parser" in vllm_kwargs or "reasoning_config" in vllm_kwargs
        )
        self._chat_template_kwargs = _resolve_chat_template_kwargs(
            explicit_chat_template_kwargs=explicit_chat_template_kwargs,
            disable_thinking=disable_thinking,
        )

        # Cap max_model_len to the model's max_position_embeddings so that
        # vLLM doesn't reject an overly large context window.
        max_model_len = vllm_kwargs.get("max_model_len")
        if max_model_len is not None:
            try:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(model, trust_remote_code=True)
                model_max_pos = getattr(config, "max_position_embeddings", None)
                if model_max_pos is not None and max_model_len > model_max_pos:
                    warnings.warn(
                        f"Capping max_model_len from {max_model_len} to "
                        f"{model_max_pos} (max_position_embeddings) for '{model}'.",
                        stacklevel=2,
                    )
                    vllm_kwargs["max_model_len"] = model_max_pos
            except Exception as e:
                warnings.warn(
                    "Could not validate max_model_len against "
                    f"max_position_embeddings for '{model}': {e}. "
                    "Proceeding without clamping; vLLM may raise if the value is too large.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._sampling_params_kwargs = {
            "max_tokens": max_tokens,
            "temperature": float(vllm_kwargs.pop("temperature", 0.6)),
            "top_p": float(vllm_kwargs.pop("top_p", 0.95)),
        }
        if thinking_token_budget is not None:
            if max_tokens is not None:
                thinking_token_budget = min(int(thinking_token_budget), int(max_tokens))
            if explicit_reasoning_settings:
                self._sampling_params_kwargs["thinking_token_budget"] = int(
                    thinking_token_budget
                )
            elif is_thinking_model(model):
                reasoning_parser = _default_reasoning_parser_for_model(model)
                assert reasoning_parser is not None  # guarded by is_thinking_model()
                vllm_kwargs.setdefault(
                    "reasoning_config",
                    ReasoningConfig(
                        reasoning_start_str=VLLM_REASONING_START_STR,
                        # Shared forced end marker so vLLM can enforce the
                        # thinking-token budget for offline `LLM.chat()`. The
                        # parser itself still varies by model family (e.g. OLMo
                        # uses `olmo3`) on vLLM's OpenAI server.
                        reasoning_end_str=VLLM_REASONING_END_STR,
                    ),
                )
                vllm_kwargs.setdefault("reasoning_parser", reasoning_parser)
                self._sampling_params_kwargs["thinking_token_budget"] = int(
                    thinking_token_budget
                )
            else:
                warnings.warn(
                    f"Model '{model}' is not in JudgeArena's built-in thinking-model "
                    "defaults (Qwen3/SmolLM3/Olmo-3-7B-Think). Ignoring "
                    "thinking_token_budget unless reasoning_parser or "
                    "reasoning_config is provided explicitly.",
                    stacklevel=2,
                )
        self.sampling_params = SamplingParams(**self._sampling_params_kwargs)

        self.llm = _init_llm_with_retry(
            LLM, model=model, trust_remote_code=True, **vllm_kwargs
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Resolve chat template:
        # 1. Explicit override always wins → use chat() with that template
        # 2. If tokenizer has one, use it → use chat() (pass None to vLLM)
        # 3. No template found → fall back to generate() for base models
        if chat_template:
            self.chat_template = chat_template
            self._use_generate = False
            logger.info("ChatVLLM: using explicit chat template for '%s'", model)
        else:
            if not getattr(self.tokenizer, "chat_template", None):
                warnings.warn(
                    f"Model '{model}' tokenizer does not define a chat template. "
                    f"Falling back to llm.generate() (no chat formatting). "
                    f"Override with --chat_template if this model needs one.",
                    stacklevel=2,
                )
                self.chat_template = None
                self._use_generate = True
                if disable_thinking:
                    warnings.warn(
                        f"Model '{model}' has no chat template, so disable_thinking "
                        "cannot be applied when falling back to llm.generate().",
                        stacklevel=2,
                    )
            else:
                self.chat_template = None  # let vLLM use the tokenizer's own
                self._use_generate = False
                logger.info("ChatVLLM: using tokenizer's chat template for '%s'", model)

    def set_temperature(self, temperature: float) -> None:
        from vllm import SamplingParams

        self._sampling_params_kwargs["temperature"] = float(temperature)
        self.sampling_params = SamplingParams(**self._sampling_params_kwargs)

    def _to_messages(self, input_item) -> list[dict]:
        """Convert LangChain prompt input to OpenAI-style messages."""
        # Map LangChain message types to OpenAI roles
        role_map = {"human": "user", "ai": "assistant", "system": "system"}

        # Handle ChatPromptValue from LangChain
        if hasattr(input_item, "to_messages"):
            lc_messages = input_item.to_messages()
            return [
                {"role": role_map.get(msg.type, msg.type), "content": msg.content}
                for msg in lc_messages
            ]
        # Handle list of tuples like [("system", "..."), ("user", "...")]
        elif (
            isinstance(input_item, list)
            and input_item
            and isinstance(input_item[0], tuple)
        ):
            return [
                {"role": role if role != "human" else "user", "content": content}
                for role, content in input_item
            ]
        # Handle already formatted messages
        elif (
            isinstance(input_item, list)
            and input_item
            and isinstance(input_item[0], dict)
        ):
            return input_item
        # Handle plain string (wrap as user message)
        elif isinstance(input_item, str):
            return [{"role": "user", "content": input_item}]
        else:
            raise ValueError(f"Unsupported input type: {type(input_item)}")

    def _to_raw_text(self, input_item) -> str:
        """Extract raw text from an input item for use with llm.generate()."""
        if isinstance(input_item, str):
            return input_item
        # ChatPromptValue from LangChain
        if hasattr(input_item, "to_string"):
            return input_item.to_string()
        # List of dicts (messages) - concatenate contents
        if (
            isinstance(input_item, list)
            and input_item
            and isinstance(input_item[0], dict)
        ):
            return "\n".join(msg["content"] for msg in input_item)
        raise ValueError(f"Cannot extract raw text from: {type(input_item)}")

    def _run_raw_batch(self, inputs: list):
        """Process a batch of inputs using vllm.LLM.chat() or llm.generate().

        Uses ``llm.chat()`` when a chat template is available (instruct models),
        and ``llm.generate()`` when no template is found (base models).
        """
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
        """Return the text completion for each input in *inputs*."""
        outputs = self._run_raw_batch(inputs)
        return [out.outputs[0].text for out in outputs]

    def invoke(self, input_item, **invoke_kwargs) -> str:
        """Process a single input."""
        results = self.batch([input_item], **invoke_kwargs)
        return results[0]

    async def ainvoke(self, input_item, **invoke_kwargs):
        """Async version - runs sync version in executor for compatibility."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.invoke(input_item, **invoke_kwargs)
        )


def do_inference(chat_model, inputs, use_tqdm: bool = False):
    """Run inference over *inputs*, returning a list of text completions.

    Retries on rate-limit/server errors with exponential backoff. The async
    path (``use_tqdm=True``) retries individual calls; the batch path splits
    into ``4**attempt`` chunks on failure.
    """
    invoke_kwargs = {
        # "stop": ["```"],
        # "max_tokens": 100,
    }
    if use_tqdm:
        # perform inference asynchronously to be able to update tqdm, chat_model.batch does not work as it blocks until
        # all requests are received
        # JUDGEARENA_JUDGE_MAX_CONCURRENCY caps simultaneous in-flight ainvokes
        # (e.g. against OpenRouter). Unset = unbounded, preserving prior behaviour.
        cap = safe_parse_int("JUDGEARENA_JUDGE_MAX_CONCURRENCY")
        cap = cap if cap and cap > 0 else None

        async def process_with_real_progress(chat_model, inputs, pbar):
            sem = asyncio.Semaphore(cap) if cap else None

            async def process_single(input_item, max_retries=5, base_delay=1.0):
                for attempt in range(max_retries):
                    try:
                        result = await chat_model.ainvoke(input_item, **invoke_kwargs)
                        pbar.update(1)
                        return result
                    except Exception as e:
                        if attempt == max_retries - 1 or not _is_retryable_error(e):
                            raise
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "Retry because of a server error, %d/%d: %s. Waiting %ss...",
                            attempt + 1,
                            max_retries,
                            e,
                            delay,
                        )
                        await asyncio.sleep(delay)

            async def gated(inp):
                if sem is None:
                    return await process_single(inp)
                async with sem:
                    return await process_single(inp)

            # asyncio.gather preserves order (unlike as_completed)
            results = await asyncio.gather(*[gated(inp) for inp in inputs])
            return results

        with logging_redirect_tqdm(), tqdm(total=len(inputs)) as pbar:
            res = asyncio.run(
                process_with_real_progress(
                    chat_model=chat_model, inputs=inputs, pbar=pbar
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
                except Exception as e:
                    if attempt == max_retries - 1 or not _is_retryable_error(e):
                        raise
                    delay = base_delay * (2**attempt)
                    next_chunks = 4 ** (attempt + 1)
                    logger.warning(
                        "Retry because of a server error, %d/%d: %s. Waiting %ss, then splitting into %d chunks...",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                        next_chunks,
                    )
                    time.sleep(delay)

        res = batch_with_retry(inputs)

    # Not sure why the API of Langchain returns sometime a string and sometimes an AIMessage object
    # is it because of using Chat and barebones models?
    # when using OpenAI, the output is AIMessage not a string...
    res = [x.content if hasattr(x, "content") else x for x in res]
    return res


def make_model(model: str, max_tokens: int | None = 8192, **engine_kwargs):
    """Instantiate a model wrapper from a provider/model-name string.

    Args:
        model: Format ``{Provider}/{model_path}``, e.g.
            ``VLLM/meta-llama/Llama-3.3-70B-Instruct``.
        max_tokens: Maximum tokens the model may generate.
        **engine_kwargs: Engine-specific options forwarded to the model wrapper.
    """
    # Avoid mutating the original engine_kwargs dictionary
    # NOTE: this is a shallow copy since we are not modifying any
    # mutable objects in the dictionary.
    engine_kwargs = engine_kwargs.copy()

    # Dedicated arguments like max_tokens always win over engine_kwargs.
    engine_kwargs["max_tokens"] = max_tokens or 8192

    model_provider, model_name = _split_model_spec(model)

    # vLLM-engine-only kwargs must not leak to remote-API providers
    # (OpenRouter, OpenAI, Together): langchain-openai forwards unknown
    # kwargs via model_kwargs into chat.completions.create, which rejects them.
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
        ):
            engine_kwargs.pop(key, None)

    if model_provider == "Dummy":
        return DummyModel(model)

    logger.info("Loading %s(model=%s)", model_provider, model_name)

    # Use our custom ChatVLLM wrapper which properly applies chat templates
    if model_provider == "VLLM":
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        engine_kwargs["chat_template"] = engine_kwargs.get("chat_template", None)

        return ChatVLLM(
            model=model_name,
            **engine_kwargs,
        )

    if model_provider == "OpenRouter":
        # Special case we need to override API url and key
        return ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            model=model_name,
            **engine_kwargs,
        )
    else:
        model_classes = [
            LlamaCpp,
            ChatOpenAI,
        ]
        if model_provider == "LlamaCpp":
            engine_kwargs["model_path"] = model_name
        else:
            engine_kwargs["model"] = model_name

        try:
            from langchain_together.llms import Together

            model_classes.append(Together)
        except ImportError as e:
            logger.debug("Optional provider not available: %s", e)
        try:
            from langchain_openai.llms import OpenAI

            model_classes.append(OpenAI)
        except ImportError as e:
            logger.debug("Optional provider not available: %s", e)
        model_cls_dict = {model_cls.__name__: model_cls for model_cls in model_classes}
        assert model_provider in model_cls_dict, (
            f"{model_provider} not available, choose among {list(model_cls_dict.keys())}"
        )
        return model_cls_dict[model_provider](**engine_kwargs)
