import asyncio
import os
import re
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download
from langchain_community.cache import SQLiteCache
from langchain_community.llms import LlamaCpp
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm

from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.openrouter_reference_pricing import OpenRouterReferencePricingTracker


def _data_root_path() -> Path:
    raw = os.environ.get("JUDGEARENA_DATA") or os.environ.get("OPENJURY_DATA")
    if raw:
        return Path(raw).expanduser()
    return Path("~/judgearena-data/").expanduser()


data_root = _data_root_path()

DEFAULT_VLLM_JUDGE_THINKING_TOKEN_BUDGET = 512
VLLM_QWEN_REASONING_START_STR = "<think>"
VLLM_QWEN_REASONING_END_STR = (
    "I have to give the solution based on the thinking directly now.</think>"
)


@dataclass(frozen=True)
class ReasoningModelDefaults:
    reasoning_parser: str
    reasoning_config_kwargs: dict[str, str] | None = None
    enabled_chat_template_kwargs: dict[str, object] | None = None
    disabled_chat_template_kwargs: dict[str, object] | None = None


_REASONING_MODEL_DEFAULTS: tuple[
    tuple[tuple[str, ...], ReasoningModelDefaults], ...
] = (
    (
        ("qwen3",),
        ReasoningModelDefaults(
            reasoning_parser="qwen3",
            reasoning_config_kwargs={
                "reasoning_start_str": VLLM_QWEN_REASONING_START_STR,
                "reasoning_end_str": VLLM_QWEN_REASONING_END_STR,
            },
            disabled_chat_template_kwargs={"enable_thinking": False},
        ),
    ),
    (("qwq-32b",), ReasoningModelDefaults(reasoning_parser="deepseek_r1")),
    (
        ("deepseek-r1", "r1-distill"),
        ReasoningModelDefaults(reasoning_parser="deepseek_r1"),
    ),
    (
        ("deepseek-v3.1",),
        ReasoningModelDefaults(
            reasoning_parser="deepseek_v3",
            enabled_chat_template_kwargs={"thinking": True},
            disabled_chat_template_kwargs={"thinking": False},
        ),
    ),
    (
        ("ernie-4.5", "ernie4.5"),
        ReasoningModelDefaults(reasoning_parser="ernie45"),
    ),
    (("glm-4.5", "glm4.5"), ReasoningModelDefaults(reasoning_parser="glm45")),
    (
        ("holo2",),
        ReasoningModelDefaults(
            reasoning_parser="holo2",
            disabled_chat_template_kwargs={"thinking": False},
        ),
    ),
    (
        ("hunyuan-a13b",),
        ReasoningModelDefaults(reasoning_parser="hunyuan_a13b"),
    ),
    (
        ("granite-3.2",),
        ReasoningModelDefaults(
            reasoning_parser="granite",
            enabled_chat_template_kwargs={"thinking": True},
            disabled_chat_template_kwargs={"thinking": False},
        ),
    ),
    (
        ("minimax-m2",),
        ReasoningModelDefaults(reasoning_parser="minimax_m2_append_think"),
    ),
)


def get_reasoning_model_defaults(model_name: str) -> ReasoningModelDefaults | None:
    """Return JudgeArena's explicit reasoning defaults for known model families."""
    normalized = model_name.lower()
    for markers, defaults in _REASONING_MODEL_DEFAULTS:
        if any(marker in normalized for marker in markers):
            return defaults
    return None


def should_default_thinking_token_budget(
    model_name: str, vllm_kwargs: dict[str, object]
) -> bool:
    """Return True when JudgeArena should auto-apply a thinking-token budget."""
    return (
        get_reasoning_model_defaults(model_name) is not None
        or "reasoning_parser" in vllm_kwargs
        or "reasoning_config" in vllm_kwargs
    )


def _resolve_chat_template_kwargs(
    *,
    explicit_chat_template_kwargs: dict[str, object] | None,
    reasoning_defaults: ReasoningModelDefaults | None,
    enable_reasoning: bool,
    disable_thinking: bool,
) -> dict[str, object] | None:
    chat_template_kwargs = dict(explicit_chat_template_kwargs or {})
    explicit_keys = set(chat_template_kwargs)

    if enable_reasoning and not disable_thinking and reasoning_defaults is not None:
        for key, value in (
            reasoning_defaults.enabled_chat_template_kwargs or {}
        ).items():
            chat_template_kwargs.setdefault(key, value)

    if disable_thinking:
        disabled_defaults = (
            reasoning_defaults.disabled_chat_template_kwargs
            if reasoning_defaults is not None
            else {"enable_thinking": False}
        )
        for key, value in disabled_defaults.items():
            if key not in explicit_keys:
                chat_template_kwargs[key] = value

    return chat_template_kwargs or None


def _attach_provider_metadata(model_instance: object, provider_name: str) -> object:
    try:
        model_instance._judgearena_provider = provider_name
    except Exception:
        pass
    return model_instance


def set_langchain_cache():
    set_llm_cache(SQLiteCache(database_path=str(data_root / ".langchain.db")))


def download_hf(name: str, local_path: Path):
    local_path.mkdir(exist_ok=True, parents=True)
    # downloads the model from huggingface into `local_path` folder
    snapshot_download(
        repo_id="judge-arena/judge-arena-dataset",
        repo_type="dataset",
        allow_patterns=f"*{name}*",
        local_dir=local_path,
        force_download=False,
    )


def read_df(filename: Path, **pandas_kwargs) -> pd.DataFrame:
    assert filename.exists(), f"Dataframe file not found at {filename}"
    if filename.name.endswith(".csv.zip") or filename.name.endswith(".csv"):
        return pd.read_csv(filename, **pandas_kwargs)
    else:
        assert filename.name.endswith(".parquet"), f"Unsupported extension {filename}"
        return pd.read_parquet(filename, **pandas_kwargs)


def compute_pref_summary(prefs: pd.Series) -> dict[str, float | int]:
    """Compute win/loss/tie stats for preference series (0=A, 0.5=tie, 1=B)."""
    prefs = pd.Series(prefs, dtype="float64")
    valid = prefs.dropna()
    num_wins = int((valid < 0.5).sum())
    num_losses = int((valid > 0.5).sum())
    num_ties = int((valid == 0.5).sum())
    num_battles = int(len(prefs))
    denom = num_wins + num_losses + num_ties
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom > 0 else float("nan")
    return {
        "num_battles": num_battles,
        "winrate": winrate,
        "num_wins": num_wins,
        "num_losses": num_losses,
        "num_ties": num_ties,
        "num_missing": int(num_battles - denom),
    }


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


def truncate(s: str, max_len: int | None = None) -> str:
    """Truncate a string to *max_len* characters.

    Non-string inputs (e.g. ``None`` or ``float('nan')``) are coerced to the
    empty string so that callers don't have to guard against missing data.
    """
    if not isinstance(s, str):
        return ""
    if max_len is not None:
        return s[:max_len]
    return s


def safe_text(value: object, truncate_chars: int | None) -> str:
    """Coerce *value* to a string and optionally truncate.

    Returns the empty string for ``None`` and NaN-like values so callers
    don't have to guard against missing data.
    """
    if value is None:
        return ""
    is_missing = pd.isna(value)
    if isinstance(is_missing, bool) and is_missing:
        return ""
    return truncate(str(value), max_len=truncate_chars)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking_tags(text: str | None) -> str:
    """Remove full `<think>...</think>` blocks from raw model output."""
    if not isinstance(text, str):
        return ""
    return _THINK_BLOCK_RE.sub("", text)


def do_inference(
    chat_model,
    inputs,
    use_tqdm: bool = False,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    usage_model_spec: str | None = None,
):
    # Retries on rate-limit/server errors with exponential backoff.
    # Async path retries individual calls; batch path splits into 4^attempt chunks on failure.
    invoke_kwargs = {
        # "stop": ["```"],
        # "max_tokens": 100,
    }
    if use_tqdm:
        # perform inference asynchronously to be able to update tqdm, chat_model.batch does not work as it blocks until
        # all requests are received
        async def process_with_real_progress(chat_model, inputs, pbar):
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
                        print(
                            f"Retry because of a server error, {attempt + 1}/{max_retries}: {e}. Waiting {delay}s..."
                        )
                        await asyncio.sleep(delay)

            # asyncio.gather preserves order (unlike as_completed)
            results = await asyncio.gather(*[process_single(inp) for inp in inputs])
            return results

        with tqdm(total=len(inputs)) as pbar:
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
                    print(
                        f"Retry because of a server error, {attempt + 1}/{max_retries}: {e}. Waiting {delay}s, then splitting into {next_chunks} chunks..."
                    )
                    time.sleep(delay)

        res = batch_with_retry(inputs)

    # Not sure why the API of Langchain returns sometime a string and sometimes an AIMessage object
    # is it because of using Chat and barebones models?
    # when using OpenAI, the output is AIMessage not a string...
    res = [x.content if hasattr(x, "content") else x for x in res]
    if (
        usage_tracker is not None
        and usage_phase is not None
        and usage_model_spec is not None
    ):
        try:
            usage_tracker.record_batch_from_model(
                phase=usage_phase,
                model_spec=usage_model_spec,
                chat_model=chat_model,
                inputs=list(inputs),
                outputs=res,
            )
        except Exception as e:
            print(
                f"Warning: failed to record token usage for phase "
                f"'{usage_phase}' ({usage_model_spec}): {e}"
            )
    return res


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
        reasoning_defaults = get_reasoning_model_defaults(model)
        explicit_reasoning_settings = (
            "reasoning_parser" in vllm_kwargs or "reasoning_config" in vllm_kwargs
        )
        enable_reasoning = (
            explicit_reasoning_settings or thinking_token_budget is not None
        )
        self._chat_template_kwargs = _resolve_chat_template_kwargs(
            explicit_chat_template_kwargs=explicit_chat_template_kwargs,
            reasoning_defaults=reasoning_defaults,
            enable_reasoning=enable_reasoning,
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
            if reasoning_defaults is None and not explicit_reasoning_settings:
                warnings.warn(
                    f"Model '{model}' is not in JudgeArena's supported reasoning-family map. "
                    "Ignoring thinking_token_budget unless reasoning_parser or "
                    "reasoning_config is provided explicitly.",
                    stacklevel=2,
                )
            else:
                if reasoning_defaults is not None:
                    vllm_kwargs.setdefault(
                        "reasoning_parser", reasoning_defaults.reasoning_parser
                    )
                    if reasoning_defaults.reasoning_config_kwargs is not None:
                        vllm_kwargs.setdefault(
                            "reasoning_config",
                            ReasoningConfig(
                                **reasoning_defaults.reasoning_config_kwargs
                            ),
                        )
                self._sampling_params_kwargs["thinking_token_budget"] = int(
                    thinking_token_budget
                )
        self.sampling_params = SamplingParams(**self._sampling_params_kwargs)

        self.llm = LLM(model=model, trust_remote_code=True, **vllm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        # Resolve chat template:
        # 1. Explicit override always wins → use chat() with that template
        # 2. If tokenizer has one, use it → use chat() (pass None to vLLM)
        # 3. No template found → fall back to generate() for base models
        if chat_template:
            self.chat_template = chat_template
            self._use_generate = False
            print(f"ChatVLLM: using explicit chat template for '{model}'")
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
                print(f"ChatVLLM: using tokenizer's chat template for '{model}'")

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

    def batch(self, inputs: list, **invoke_kwargs) -> list[str]:
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
        return [out.outputs[0].text for out in outputs]

    def _count_chat_prompt_tokens(self, messages: list[dict]) -> int:
        tokenizer_kwargs: dict[str, object] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if self.chat_template is not None:
            tokenizer_kwargs["chat_template"] = self.chat_template
        if self._chat_template_kwargs is not None:
            tokenizer_kwargs["chat_template_kwargs"] = self._chat_template_kwargs
        try:
            token_ids = self.tokenizer.apply_chat_template(messages, **tokenizer_kwargs)
        except TypeError:
            tokenizer_kwargs.pop("chat_template_kwargs", None)
            token_ids = self.tokenizer.apply_chat_template(messages, **tokenizer_kwargs)
        return len(token_ids)

    def count_prompt_tokens_batch(self, inputs: list) -> list[int]:
        counts: list[int] = []
        for input_item in inputs:
            if self._use_generate:
                counts.append(len(self.tokenizer.encode(self._to_raw_text(input_item))))
            else:
                counts.append(
                    self._count_chat_prompt_tokens(self._to_messages(input_item))
                )
        return counts

    def count_completion_tokens_batch(self, outputs: list[str]) -> list[int]:
        return [
            len(self.tokenizer.encode(output, add_special_tokens=False))
            for output in outputs
        ]

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

    model_provider = model.split("/")[0]

    if model_provider == "Dummy":
        return _attach_provider_metadata(DummyModel(model), model_provider)

    model_name = "/".join(model.split("/")[1:])
    print(f"Loading {model_provider}(model={model_name})")

    # Use our custom ChatVLLM wrapper which properly applies chat templates
    if model_provider == "VLLM":
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        engine_kwargs["chat_template"] = engine_kwargs.get("chat_template", None)

        return _attach_provider_metadata(
            ChatVLLM(
                model=model_name,
                **engine_kwargs,
            ),
            model_provider,
        )

    if model_provider == "OpenRouter":
        # Special case we need to override API url and key
        return _attach_provider_metadata(
            ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=model_name,
                **engine_kwargs,
            ),
            model_provider,
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
            print(str(e))
        try:
            from langchain_openai.llms import OpenAI

            model_classes.append(OpenAI)
        except ImportError as e:
            print(str(e))
        model_cls_dict = {model_cls.__name__: model_cls for model_cls in model_classes}
        assert model_provider in model_cls_dict, (
            f"{model_provider} not available, choose among {list(model_cls_dict.keys())}"
        )
        return _attach_provider_metadata(
            model_cls_dict[model_provider](**engine_kwargs), model_provider
        )


def infer_model_spec_from_instance(model: object) -> str | None:
    if isinstance(model, DummyModel):
        return model.name
    provider_name = getattr(model, "_judgearena_provider", None)
    model_path = getattr(model, "model_path", None)
    if isinstance(model_path, str):
        if isinstance(provider_name, str):
            return f"{provider_name}/{model_path}"
        return model_path
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    if isinstance(model_name, str):
        if isinstance(provider_name, str):
            return f"{provider_name}/{model_name}"
        return f"{model.__class__.__name__}/{model_name}"
    return None


def download_all():
    print(f"Downloading all dataset in {data_root}")
    local_path_tables = data_root / "tables"
    for dataset in [
        "alpaca-eval",
        "arena-hard-v0.1",
        "arena-hard-v2.0",
        "m-arena-hard",
    ]:
        if is_arena_hard_dataset(dataset):
            download_arena_hard(dataset=dataset, local_tables_path=local_path_tables)
        else:
            download_hf(name=dataset, local_path=local_path_tables)

    snapshot_download(
        repo_id="geoalgo/multilingual-contexts-to-be-completed",
        repo_type="dataset",
        allow_patterns="*",
        local_dir=data_root / "contexts",
        force_download=False,
    )

    from judgearena.instruction_dataset.mt_bench import download_mt_bench

    download_mt_bench()


class Timeblock:
    """Timer context manager"""

    def __init__(self, name: str | None = None, verbose: bool = True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        """Stop the context manager timer"""
        self.end = time.time()
        self.duration = self.end - self.start
        if self.verbose:
            print(self)

    def __str__(self):
        name = self.name if self.name else "block"
        msg = f"{name} took {self.duration} seconds"
        return msg


def cache_function_dataframe(
    fun: Callable[[], pd.DataFrame],
    cache_name: str,
    ignore_cache: bool = False,
    cache_path: Path | None = None,
    parquet: bool = False,
) -> pd.DataFrame:
    """
    :param fun: a function whose dataframe result obtained `fun()` will be cached
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.csv.zip`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :param parquet: whether to store the data in parquet, if not specified use csv.zip
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = data_root / "cache"

    if parquet:
        cache_file = cache_path / (cache_name + ".parquet")
    else:
        cache_file = cache_path / (cache_name + ".csv.zip")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        if parquet:
            return pd.read_parquet(cache_file)
        else:
            return pd.read_csv(cache_file)
    else:
        print(
            f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file"
        )
        with Timeblock("Evaluate function."):
            df = fun()
            assert isinstance(df, pd.DataFrame)
            if parquet:
                # object cols cannot be saved easily in parquet; numpy arrays must be
                # deep-converted to plain Python so str() produces ast.literal_eval-safe
                # repr (no "array([...])" syntax, which breaks literal_eval)
                import numpy as np

                def _to_python(x):
                    """Recursively convert numpy arrays/scalars to Python lists/dicts."""
                    if isinstance(x, np.ndarray):
                        return [_to_python(i) for i in x]
                    if isinstance(x, dict):
                        return {k: _to_python(v) for k, v in x.items()}
                    if isinstance(x, list):
                        return [_to_python(i) for i in x]
                    return x

                for col in df.select_dtypes(include="object").columns:
                    df[col] = df[col].apply(_to_python).astype(str)
                df.to_parquet(cache_file, index=False)
                return pd.read_parquet(cache_file)
            else:
                df.to_csv(cache_file, index=False)
                return pd.read_csv(cache_file)


if __name__ == "__main__":
    download_all()
