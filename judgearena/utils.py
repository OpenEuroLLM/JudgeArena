import asyncio
import os
import re
import time
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import snapshot_download
from langchain_community.cache import SQLiteCache
from langchain_community.llms import LlamaCpp
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from judgearena.chat_models import (
    OpenRouterGeminiSafetyTolerantChatOpenAI,
    is_openrouter_gemini_model,
)
from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.log import get_logger
from judgearena.openrouter_reference_pricing import OpenRouterReferencePricingTracker

logger = get_logger(__name__)


def _data_root_path() -> Path:
    raw = os.environ.get("JUDGEARENA_DATA") or os.environ.get("OPENJURY_DATA")
    if raw:
        return Path(raw).expanduser()
    return Path("~/judgearena-data/").expanduser()


data_root = _data_root_path()

DEFAULT_VLLM_JUDGE_THINKING_TOKEN_BUDGET = 512
VLLM_REASONING_START_STR = "<think>"
VLLM_REASONING_END_STR = (
    "I have to give the solution based on the thinking directly now.</think>"
)
_THINKING_MODEL_SUBSTRINGS = ("qwen3", "smollm3")


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    provider, sep, model_name = model_spec.partition("/")
    if not sep:
        return model_spec, ""
    return provider, model_name


def is_thinking_model(model_name: str) -> bool:
    """Return True for reasoning models that emit `<think>...</think>` traces.

    Covers the Qwen3 family (e.g. `Qwen/Qwen3.5-9B`) and SmolLM3 (e.g.
    `HuggingFaceTB/SmolLM3-3B`); both share the same `<think>`/`</think>` tag
    convention so vLLM's budget enforcement and our tag-stripping apply
    uniformly. Matching is case-insensitive to tolerate mixed-case HF repo
    ids like `HuggingFaceTB/SmolLM3-3B`.
    """
    lowered = model_name.lower()
    return any(token in lowered for token in _THINKING_MODEL_SUBSTRINGS)


def build_default_judge_model_kwargs(
    judge_model: str,
    engine_kwargs: dict[str, object],
    *,
    judge_engine_kwargs_override: dict[str, object] | None = None,
) -> dict[str, object]:
    """Copy judge engine kwargs and add supported built-in defaults.

    ``judge_engine_kwargs_override`` is layered on top of ``engine_kwargs``
    so callers can pin judge-only tweaks (e.g. a higher tensor-parallel size
    for a 70B judge) without poisoning the battle-model engine config, which
    must often stay on TP=1 to dodge compile-time deadlocks on hybrid models
    such as Qwen3.5.
    """
    judge_model_kwargs = dict(engine_kwargs)
    if judge_engine_kwargs_override:
        judge_model_kwargs.update(judge_engine_kwargs_override)
    provider, model_name = _split_model_spec(judge_model)
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


@dataclass(frozen=True)
class LimitEvent:
    kind: str
    stage: str
    field: str | None = None
    case_id: str | None = None
    model_spec: str | None = None
    original_length: int | None = None
    final_length: int | None = None
    note: str | None = None


class LimitEventTracker:
    def __init__(self) -> None:
        self.events: list[LimitEvent] = []

    def record(
        self,
        kind: str,
        *,
        stage: str,
        field: str | None = None,
        case_id: object | None = None,
        model_spec: str | None = None,
        original_length: int | None = None,
        final_length: int | None = None,
        note: str | None = None,
    ) -> None:
        self.events.append(
            LimitEvent(
                kind=kind,
                stage=stage,
                field=field,
                case_id=None if case_id is None else str(case_id),
                model_spec=model_spec,
                original_length=original_length,
                final_length=final_length,
                note=note,
            )
        )

    def build_summary(self) -> dict[str, Any]:
        counts_by_kind: Counter[str] = Counter()
        counts_by_stage: Counter[str] = Counter()
        counts_by_kind_and_field: dict[str, Counter[str]] = {}
        affected_cases_total: set[str] = set()
        affected_cases_by_kind: dict[str, set[str]] = {}

        for event in self.events:
            counts_by_kind[event.kind] += 1
            counts_by_stage[event.stage] += 1
            field_key = event.field or "_all"
            counts_by_kind_and_field.setdefault(event.kind, Counter())[field_key] += 1
            if event.case_id is None:
                continue
            case_key = f"{event.stage}:{event.case_id}"
            affected_cases_total.add(case_key)
            affected_cases_by_kind.setdefault(event.kind, set()).add(case_key)

        return {
            "total_events": len(self.events),
            "counts_by_kind": dict(sorted(counts_by_kind.items())),
            "counts_by_stage": dict(sorted(counts_by_stage.items())),
            "counts_by_kind_and_field": {
                kind: dict(sorted(counter.items()))
                for kind, counter in sorted(counts_by_kind_and_field.items())
            },
            "affected_cases_total": len(affected_cases_total),
            "affected_cases_by_kind": {
                kind: len(case_ids)
                for kind, case_ids in sorted(affected_cases_by_kind.items())
            },
        }


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


def truncate_with_metadata(
    s: str | None,
    max_len: int | None = None,
    *,
    tracker: LimitEventTracker | None = None,
    kind: str | None = None,
    stage: str | None = None,
    field: str | None = None,
    case_id: object | None = None,
    model_spec: str | None = None,
) -> tuple[str, bool]:
    original = s if isinstance(s, str) else ""
    truncated = truncate(original, max_len=max_len)
    was_truncated = truncated != original
    if was_truncated and tracker is not None and kind is not None and stage is not None:
        tracker.record(
            kind,
            stage=stage,
            field=field,
            case_id=case_id,
            model_spec=model_spec,
            original_length=len(original),
            final_length=len(truncated),
        )
    return truncated, was_truncated


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


def safe_text_with_metadata(
    value: object,
    truncate_chars: int | None,
    *,
    tracker: LimitEventTracker | None = None,
    kind: str | None = None,
    stage: str | None = None,
    field: str | None = None,
    case_id: object | None = None,
    model_spec: str | None = None,
) -> tuple[str, bool]:
    if value is None:
        return "", False
    is_missing = pd.isna(value)
    if isinstance(is_missing, bool) and is_missing:
        return "", False
    return truncate_with_metadata(
        str(value),
        max_len=truncate_chars,
        tracker=tracker,
        kind=kind,
        stage=stage,
        field=field,
        case_id=case_id,
        model_spec=model_spec,
    )


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking_tags(text: str | None) -> str:
    """Remove full `<think>...</think>` blocks from raw model output."""
    return strip_thinking_tags_with_metadata(text)[0]


def strip_thinking_tags_with_metadata(text: str | None) -> tuple[str, bool]:
    """Remove visible reasoning spans from raw model output."""
    if not isinstance(text, str):
        return "", False

    cleaned = _THINK_BLOCK_RE.sub("", text)
    if cleaned != text:
        return cleaned.lstrip(), True

    lowered = text.lower()
    closing_tag = "</think>"
    closing_idx = lowered.find(closing_tag)
    if closing_idx != -1 and "<think>" not in lowered[:closing_idx]:
        return text[closing_idx + len(closing_tag) :].lstrip(), True

    forced_end_idx = text.find(VLLM_REASONING_END_STR)
    if forced_end_idx != -1:
        return (
            text[forced_end_idx + len(VLLM_REASONING_END_STR) :].lstrip(),
            True,
        )

    return text, False


def _extract_ai_message_metadata(result: object) -> dict[str, Any]:
    """Extract finish_reason/stop_reason from a LangChain AIMessage result.

    LangChain chat models (ChatOpenAI for OpenRouter, Anthropic, etc.) return
    AIMessage objects with a ``response_metadata`` dict. We propagate the
    subset that downstream code consumes (finish_reason is critical: it gates
    truncation detection in _record_generation_output_limit_events).
    """
    response_metadata = getattr(result, "response_metadata", None) or {}
    finish_reason = response_metadata.get("finish_reason")
    stop_reason = response_metadata.get("stop_reason")
    if finish_reason is None and isinstance(result, dict):
        finish_reason = result.get("finish_reason")
        stop_reason = result.get("stop_reason", stop_reason)
    return {"finish_reason": finish_reason, "stop_reason": stop_reason}


def do_inference(
    chat_model,
    inputs,
    use_tqdm: bool = False,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    usage_model_spec: str | None = None,
    return_metadata: bool = False,
):
    # Retries on rate-limit/server errors with exponential backoff.
    # Async path retries individual calls; batch path splits into 4^attempt chunks on failure.
    invoke_kwargs = {
        # "stop": ["```"],
        # "max_tokens": 100,
    }
    metadata: list[dict[str, Any]] | None = None
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
                        logger.warning(
                            "Retry because of a server error, %d/%d: %s. Waiting %ss...",
                            attempt + 1,
                            max_retries,
                            e,
                            delay,
                        )
                        await asyncio.sleep(delay)

            # asyncio.gather preserves order (unlike as_completed)
            results = await asyncio.gather(*[process_single(inp) for inp in inputs])
            return results

        with logging_redirect_tqdm(), tqdm(total=len(inputs)) as pbar:
            res = asyncio.run(
                process_with_real_progress(
                    chat_model=chat_model, inputs=inputs, pbar=pbar
                )
            )
        if return_metadata:
            metadata = [_extract_ai_message_metadata(r) for r in res]
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
                    results_metadata = []
                    for chunk in chunks:
                        if return_metadata and hasattr(
                            chat_model, "batch_with_metadata"
                        ):
                            chunk_results, chunk_metadata = (
                                chat_model.batch_with_metadata(
                                    inputs=chunk, **invoke_kwargs
                                )
                            )
                        else:
                            chunk_results = chat_model.batch(
                                inputs=chunk, **invoke_kwargs
                            )
                            chunk_metadata = [
                                _extract_ai_message_metadata(r) for r in chunk_results
                            ]
                        results.extend(chunk_results)
                        results_metadata.extend(chunk_metadata)
                    return results, results_metadata
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

        res, metadata = batch_with_retry(inputs)

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
    if return_metadata:
        return res, (metadata or [{} for _ in res])
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
        limit_event_tracker: LimitEventTracker | None = vllm_kwargs.pop(
            "limit_event_tracker", None
        )
        limit_event_stage = str(vllm_kwargs.pop("limit_event_stage", "model_init"))
        limit_event_model_spec = str(
            vllm_kwargs.pop("limit_event_model_spec", f"VLLM/{model}")
        )
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
                    if limit_event_tracker is not None:
                        limit_event_tracker.record(
                            "max_model_len_clamped",
                            stage=limit_event_stage,
                            field="max_model_len",
                            model_spec=limit_event_model_spec,
                            original_length=int(max_model_len),
                            final_length=int(model_max_pos),
                        )
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
        self._thinking_budget_marker: str | None = None
        self._thinking_budget_value: int | None = None
        if thinking_token_budget is not None:
            if max_tokens is not None:
                thinking_token_budget = min(int(thinking_token_budget), int(max_tokens))
            if explicit_reasoning_settings:
                self._sampling_params_kwargs["thinking_token_budget"] = int(
                    thinking_token_budget
                )
                self._thinking_budget_marker = VLLM_REASONING_END_STR
                self._thinking_budget_value = int(thinking_token_budget)
            elif is_thinking_model(model):
                vllm_kwargs.setdefault(
                    "reasoning_config",
                    ReasoningConfig(
                        reasoning_start_str=VLLM_REASONING_START_STR,
                        reasoning_end_str=VLLM_REASONING_END_STR,
                    ),
                )
                # The `qwen3` reasoning_parser only runs inside vLLM's
                # OpenAI-compatible server for `reasoning_content` extraction.
                # For offline batch inference via LLM.chat() it is inert, so
                # it is safe to reuse for any `<think>`/`</think>` model
                # (Qwen3 + SmolLM3).
                vllm_kwargs.setdefault("reasoning_parser", "qwen3")
                self._sampling_params_kwargs["thinking_token_budget"] = int(
                    thinking_token_budget
                )
                self._thinking_budget_marker = VLLM_REASONING_END_STR
                self._thinking_budget_value = int(thinking_token_budget)
            else:
                warnings.warn(
                    f"Model '{model}' is not in JudgeArena's built-in thinking-model "
                    "defaults (Qwen3/SmolLM3). Ignoring thinking_token_budget unless "
                    "reasoning_parser or reasoning_config is provided explicitly.",
                    stacklevel=2,
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

    def batch_with_metadata(
        self, inputs: list, **invoke_kwargs
    ) -> tuple[list[str], list[dict[str, Any]]]:
        outputs = self._run_raw_batch(inputs)
        texts: list[str] = []
        metadata: list[dict[str, Any]] = []
        marker = self._thinking_budget_marker
        for out in outputs:
            first_output = out.outputs[0]
            text = first_output.text
            texts.append(text)
            row: dict[str, Any] = {
                "finish_reason": getattr(first_output, "finish_reason", None),
                "stop_reason": getattr(first_output, "stop_reason", None),
            }
            if marker is not None:
                # vLLM emits the forced reasoning-end marker verbatim when the
                # per-request thinking-token budget is exhausted; the marker is
                # absent otherwise. Detecting it here gives
                # `_record_generation_output_limit_events` a deterministic
                # signal to log a `generation_thinking_token_budget` event.
                row["thinking_budget_exhausted"] = marker in text
                row["thinking_token_budget"] = self._thinking_budget_value
            metadata.append(row)
        return texts, metadata

    def batch(self, inputs: list, **invoke_kwargs) -> list[str]:
        texts, _metadata = self.batch_with_metadata(inputs, **invoke_kwargs)
        return texts

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
    limit_event_tracker = engine_kwargs.pop("limit_event_tracker", None)
    limit_event_stage = engine_kwargs.pop("limit_event_stage", None)
    limit_event_model_spec = engine_kwargs.pop("limit_event_model_spec", None)

    # Dedicated arguments like max_tokens always win over engine_kwargs.
    engine_kwargs["max_tokens"] = max_tokens or 8192

    model_provider, model_name = _split_model_spec(model)

    # vLLM-engine-only kwargs must not leak to remote-API providers
    # (OpenRouter, OpenAI, Together): langchain-openai forwards unknown
    # kwargs via model_kwargs into chat.completions.create, which rejects them.
    if model_provider != "VLLM":
        engine_kwargs.pop("max_model_len", None)
        engine_kwargs.pop("chat_template", None)

    if model_provider == "Dummy":
        return DummyModel(model)

    logger.info("Loading %s(model=%s)", model_provider, model_name)

    # Use our custom ChatVLLM wrapper which properly applies chat templates
    if model_provider == "VLLM":
        engine_kwargs = {k: v for k, v in engine_kwargs.items() if v is not None}
        engine_kwargs["chat_template"] = engine_kwargs.get("chat_template", None)
        if limit_event_tracker is not None:
            engine_kwargs["limit_event_tracker"] = limit_event_tracker
        if limit_event_stage is not None:
            engine_kwargs["limit_event_stage"] = limit_event_stage
        if limit_event_model_spec is not None:
            engine_kwargs["limit_event_model_spec"] = limit_event_model_spec

        return ChatVLLM(
            model=model_name,
            **engine_kwargs,
        )

    if model_provider == "OpenRouter":
        # Gemini's core policy filter rejects a small fraction of prompts with
        # a hard PROHIBITED_CONTENT error that safety_settings cannot override;
        # the subclass converts those into stub refusals so batch generation
        # (e.g. benchmark baselines) completes instead of crashing.
        chat_model_cls = (
            OpenRouterGeminiSafetyTolerantChatOpenAI
            if is_openrouter_gemini_model(model)
            else ChatOpenAI
        )
        return chat_model_cls(
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


def infer_model_spec_from_instance(model: object) -> str | None:
    if isinstance(model, DummyModel):
        return model.name
    if isinstance(model, ChatVLLM):
        return f"VLLM/{model.model_path}"
    if isinstance(model, LlamaCpp):
        model_path = getattr(model, "model_path", None)
        if isinstance(model_path, str):
            return f"LlamaCpp/{model_path}"
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
    if isinstance(model_name, str):
        return f"{model.__class__.__name__}/{model_name}"
    return None


def download_all():
    from judgearena.instruction_dataset.m_arenahard import M_ARENA_HARD_BASELINES

    logger.info("Downloading all datasets in %s", data_root)
    local_path_tables = data_root / "tables"
    datasets = [
        "alpaca-eval",
        "arena-hard-v0.1",
        "arena-hard-v2.0",
        *M_ARENA_HARD_BASELINES.keys(),
    ]
    for dataset in datasets:
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
            logger.info("%s", self)

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
        logger.info("Loading cache %s", cache_file)
        if parquet:
            return pd.read_parquet(cache_file)
        else:
            return pd.read_csv(cache_file)
    else:
        logger.info(
            "Cache %s not found or ignore_cache set to True, regenerating the file",
            cache_file,
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
