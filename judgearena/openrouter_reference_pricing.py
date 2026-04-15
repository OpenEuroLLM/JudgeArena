from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_PRICE_CACHE_ENV = "JUDGEARENA_OPENROUTER_PRICE_CACHE"
DEFAULT_CACHE_RELATIVE_PATH = Path("reference_pricing") / "openrouter_models.json"
_KNOWN_PROVIDER_PREFIXES = frozenset(
    {
        "ChatOpenAI",
        "Dummy",
        "LlamaCpp",
        "OpenAI",
        "OpenRouter",
        "Together",
        "VLLM",
    }
)
_UNAPPLIED_PRICE_COMPONENTS = (
    "image",
    "input_cache_read",
    "input_cache_write",
    "internal_reasoning",
    "web_search",
)
_LOCAL_VARIANT_SUFFIX_RE = re.compile(
    r"(?i)(?:[-_](?:fp8|fp16|bf16|int8|int4|int3|awq|gptq(?:[-_][a-z0-9]+)*))+$"
)


def _data_root_path() -> Path:
    raw = os.environ.get("JUDGEARENA_DATA") or os.environ.get("OPENJURY_DATA")
    if raw:
        return Path(raw).expanduser()
    return Path("~/judgearena-data/").expanduser()


def get_openrouter_price_cache_path() -> Path:
    raw = os.environ.get(OPENROUTER_PRICE_CACHE_ENV)
    if raw:
        return Path(raw).expanduser()
    return _data_root_path() / DEFAULT_CACHE_RELATIVE_PATH


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_price_float(raw_value: object) -> float:
    if raw_value in (None, ""):
        return 0.0
    return float(raw_value)


@dataclass(frozen=True)
class OpenRouterModelPricing:
    prompt: float
    completion: float
    request: float = 0.0
    image: float = 0.0
    web_search: float = 0.0
    internal_reasoning: float = 0.0
    input_cache_read: float = 0.0
    input_cache_write: float = 0.0


@dataclass(frozen=True)
class OpenRouterModelEntry:
    model_id: str
    canonical_slug: str | None
    hugging_face_id: str | None
    name: str
    pricing: OpenRouterModelPricing

    def exact_match_candidates(self) -> tuple[str, ...]:
        candidates = [self.model_id]
        if self.canonical_slug:
            candidates.append(self.canonical_slug)
        if self.hugging_face_id:
            candidates.append(self.hugging_face_id)
        return tuple(candidates)


@dataclass(frozen=True)
class OpenRouterPriceCatalog:
    source_url: str
    fetched_at_utc: str | None
    cache_path: str
    models: tuple[OpenRouterModelEntry, ...]


@dataclass(frozen=True)
class TokenUsageRecord:
    phase: str
    model_spec: str
    prompt_tokens: int
    completion_tokens: int
    requests: int = 1


class OpenRouterReferencePricingTracker:
    def __init__(self) -> None:
        self._records: list[TokenUsageRecord] = []

    @property
    def records(self) -> list[TokenUsageRecord]:
        return list(self._records)

    def has_records(self) -> bool:
        return bool(self._records)

    def record_batch_from_model(
        self,
        *,
        phase: str,
        model_spec: str,
        chat_model: object,
        inputs: list,
        outputs: list[str],
    ) -> bool:
        if not hasattr(chat_model, "count_prompt_tokens_batch") or not hasattr(
            chat_model, "count_completion_tokens_batch"
        ):
            return False

        prompt_tokens = chat_model.count_prompt_tokens_batch(inputs)
        completion_tokens = chat_model.count_completion_tokens_batch(outputs)
        if len(prompt_tokens) != len(completion_tokens) or len(prompt_tokens) != len(
            outputs
        ):
            raise ValueError("Prompt/completion token counts must align with outputs.")

        for prompt_count, completion_count in zip(
            prompt_tokens, completion_tokens, strict=True
        ):
            self._records.append(
                TokenUsageRecord(
                    phase=phase,
                    model_spec=model_spec,
                    prompt_tokens=int(prompt_count),
                    completion_tokens=int(completion_count),
                )
            )
        return True


def _parse_catalog_model(raw_model: dict[str, Any]) -> OpenRouterModelEntry:
    raw_pricing = raw_model.get("pricing") or {}
    pricing = OpenRouterModelPricing(
        prompt=_as_price_float(raw_pricing.get("prompt")),
        completion=_as_price_float(raw_pricing.get("completion")),
        request=_as_price_float(raw_pricing.get("request")),
        image=_as_price_float(raw_pricing.get("image")),
        web_search=_as_price_float(raw_pricing.get("web_search")),
        internal_reasoning=_as_price_float(raw_pricing.get("internal_reasoning")),
        input_cache_read=_as_price_float(raw_pricing.get("input_cache_read")),
        input_cache_write=_as_price_float(raw_pricing.get("input_cache_write")),
    )
    return OpenRouterModelEntry(
        model_id=str(raw_model["id"]),
        canonical_slug=(
            str(raw_model["canonical_slug"])
            if raw_model.get("canonical_slug") is not None
            else None
        ),
        hugging_face_id=(
            str(raw_model["hugging_face_id"])
            if raw_model.get("hugging_face_id") is not None
            else None
        ),
        name=str(raw_model.get("name") or raw_model["id"]),
        pricing=pricing,
    )


def parse_openrouter_catalog_payload(
    payload: dict[str, Any],
    *,
    fetched_at_utc: str | None = None,
    cache_path: str | Path | None = None,
) -> OpenRouterPriceCatalog:
    raw_models = payload.get("models")
    if raw_models is None:
        raw_models = payload.get("data")
    if not isinstance(raw_models, list):
        raise ValueError("OpenRouter models payload is missing a `data` list.")
    return OpenRouterPriceCatalog(
        source_url=str(payload.get("source_url") or OPENROUTER_MODELS_URL),
        fetched_at_utc=(
            str(payload["fetched_at_utc"])
            if payload.get("fetched_at_utc") is not None
            else fetched_at_utc
        ),
        cache_path=str(cache_path or payload.get("cache_path") or ""),
        models=tuple(_parse_catalog_model(model) for model in raw_models),
    )


def _cache_payload_from_raw_response(raw_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_url": OPENROUTER_MODELS_URL,
        "fetched_at_utc": _utc_now_iso(),
        "models": raw_payload.get("data", []),
    }


def _fetch_openrouter_catalog_payload(timeout_seconds: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(OPENROUTER_MODELS_URL)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def load_openrouter_price_catalog(
    *,
    refresh: bool = False,
    cache_path: str | Path | None = None,
) -> OpenRouterPriceCatalog:
    resolved_cache_path = (
        Path(cache_path)
        if cache_path is not None
        else get_openrouter_price_cache_path()
    )
    resolved_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not resolved_cache_path.is_file():
        fetched_payload = _fetch_openrouter_catalog_payload()
        cache_payload = _cache_payload_from_raw_response(fetched_payload)
        with open(resolved_cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache_payload, handle, indent=2, sort_keys=True)
    with open(resolved_cache_path, encoding="utf-8") as handle:
        cached_payload = json.load(handle)
    return parse_openrouter_catalog_payload(
        cached_payload,
        cache_path=resolved_cache_path,
    )


def load_openrouter_price_catalog_with_fallback(
    *,
    refresh: bool = False,
    cache_path: str | Path | None = None,
) -> tuple[OpenRouterPriceCatalog | None, str | None]:
    resolved_cache_path = (
        Path(cache_path)
        if cache_path is not None
        else get_openrouter_price_cache_path()
    )
    try:
        catalog = load_openrouter_price_catalog(
            refresh=refresh,
            cache_path=resolved_cache_path,
        )
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        if resolved_cache_path.is_file():
            try:
                catalog = load_openrouter_price_catalog(
                    refresh=False,
                    cache_path=resolved_cache_path,
                )
                return (
                    catalog,
                    f"Using cached OpenRouter price catalog after refresh failed: {exc}",
                )
            except (OSError, ValueError, json.JSONDecodeError) as cached_exc:
                return None, (
                    "OpenRouter price catalog refresh and cache load failed: "
                    f"{cached_exc}"
                )
        return None, f"OpenRouter price catalog unavailable: {exc}"
    return catalog, None


def _strip_provider_prefix(model_spec: str) -> str | None:
    if not model_spec:
        return None
    if "/" not in model_spec:
        return model_spec
    provider, remainder = model_spec.split("/", 1)
    if provider in _KNOWN_PROVIDER_PREFIXES:
        return remainder
    return model_spec


def _candidate_match_variants(candidate: str) -> tuple[str, ...]:
    variants = [candidate]
    owner_prefix = ""
    model_name = candidate
    if "/" in candidate:
        owner, model_name = candidate.rsplit("/", 1)
        owner_prefix = f"{owner}/"
    normalized_model_name = _LOCAL_VARIANT_SUFFIX_RE.sub("", model_name)
    if normalized_model_name and normalized_model_name != model_name:
        variants.append(f"{owner_prefix}{normalized_model_name}")
    return tuple(dict.fromkeys(variants))


def find_openrouter_match(
    catalog: OpenRouterPriceCatalog,
    model_spec: str,
) -> tuple[OpenRouterModelEntry | None, str | None]:
    candidate = _strip_provider_prefix(model_spec)
    if not candidate:
        return None, None
    candidate_variants = _candidate_match_variants(candidate)
    lowered_exact = candidate_variants[0].casefold()
    lowered_normalized = {variant.casefold() for variant in candidate_variants[1:]}
    for model in catalog.models:
        lowered_candidates = {
            match_candidate.casefold()
            for match_candidate in model.exact_match_candidates()
        }
        if lowered_exact in lowered_candidates:
            return model, "exact_case_insensitive"
        if lowered_normalized.intersection(lowered_candidates):
            return model, "local_variant_suffix_stripped"
    return None, None


def find_exact_openrouter_match(
    catalog: OpenRouterPriceCatalog,
    model_spec: str,
) -> OpenRouterModelEntry | None:
    matched_model, match_strategy = find_openrouter_match(catalog, model_spec)
    if match_strategy == "exact_case_insensitive":
        return matched_model
    return None


def _sum_phase_records(records: list[TokenUsageRecord]) -> dict[str, int]:
    prompt_tokens = sum(record.prompt_tokens for record in records)
    completion_tokens = sum(record.completion_tokens for record in records)
    requests = sum(record.requests for record in records)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "request_count": requests,
    }


def _ignored_pricing_components(pricing: OpenRouterModelPricing) -> list[str]:
    ignored: list[str] = []
    for field_name in _UNAPPLIED_PRICE_COMPONENTS:
        if getattr(pricing, field_name) != 0.0:
            ignored.append(field_name)
    return ignored


def _phase_summary_for_unmatched(
    *,
    model_spec: str,
    usage_totals: dict[str, int],
    pricing_status: str,
) -> dict[str, Any]:
    return {
        "model_spec": model_spec,
        "pricing_status": pricing_status,
        **usage_totals,
        "openrouter_model_id": None,
        "openrouter_canonical_slug": None,
        "openrouter_hugging_face_id": None,
        "openrouter_reference_cost_usd": None,
        "applied_pricing_usd": None,
        "ignored_pricing_components": [],
    }


def build_openrouter_reference_pricing_summary(
    *,
    tracker: OpenRouterReferencePricingTracker,
    phase_model_specs: dict[str, str],
    refresh_catalog: bool = False,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    phase_records: dict[str, list[TokenUsageRecord]] = {
        phase: [record for record in tracker.records if record.phase == phase]
        for phase in phase_model_specs
    }
    should_load_catalog = any(phase_records.values())
    catalog: OpenRouterPriceCatalog | None = None
    catalog_warning: str | None = None
    if should_load_catalog:
        catalog, catalog_warning = load_openrouter_price_catalog_with_fallback(
            refresh=refresh_catalog,
            cache_path=cache_path,
        )

    phase_summaries: dict[str, dict[str, Any]] = {}
    priced_costs: list[float] = []
    for phase, model_spec in phase_model_specs.items():
        records = phase_records[phase]
        usage_totals = _sum_phase_records(records)
        if not records:
            phase_summaries[phase] = _phase_summary_for_unmatched(
                model_spec=model_spec,
                usage_totals=usage_totals,
                pricing_status="no_runtime_token_data",
            )
            continue
        if catalog is None:
            phase_summaries[phase] = _phase_summary_for_unmatched(
                model_spec=model_spec,
                usage_totals=usage_totals,
                pricing_status="price_catalog_unavailable",
            )
            continue

        matched_model, match_strategy = find_openrouter_match(catalog, model_spec)
        if matched_model is None:
            phase_summaries[phase] = _phase_summary_for_unmatched(
                model_spec=model_spec,
                usage_totals=usage_totals,
                pricing_status="no_exact_openrouter_match",
            )
            continue

        ignored_components = _ignored_pricing_components(matched_model.pricing)
        phase_cost = (
            usage_totals["prompt_tokens"] * matched_model.pricing.prompt
            + usage_totals["completion_tokens"] * matched_model.pricing.completion
            + usage_totals["request_count"] * matched_model.pricing.request
        )
        priced_costs.append(phase_cost)
        if match_strategy == "local_variant_suffix_stripped":
            base_status = "matched_openrouter_model_after_variant_normalization"
        else:
            base_status = "matched_exact_openrouter_model"
        phase_summaries[phase] = {
            "model_spec": model_spec,
            "pricing_status": (
                base_status if not ignored_components else f"{base_status}_partial"
            ),
            **usage_totals,
            "openrouter_model_id": matched_model.model_id,
            "openrouter_canonical_slug": matched_model.canonical_slug,
            "openrouter_hugging_face_id": matched_model.hugging_face_id,
            "openrouter_reference_cost_usd": phase_cost,
            "applied_pricing_usd": {
                "prompt": matched_model.pricing.prompt,
                "completion": matched_model.pricing.completion,
                "request": matched_model.pricing.request,
            },
            "ignored_pricing_components": ignored_components,
        }

    total_prompt_tokens = sum(
        phase_summary["prompt_tokens"] for phase_summary in phase_summaries.values()
    )
    total_completion_tokens = sum(
        phase_summary["completion_tokens"] for phase_summary in phase_summaries.values()
    )
    total_request_count = sum(
        phase_summary["request_count"] for phase_summary in phase_summaries.values()
    )
    total_reference_cost = sum(priced_costs) if priced_costs else None

    return {
        "pricing_model": "openrouter_reference",
        "pricing_currency": "USD",
        "catalog_source_url": OPENROUTER_MODELS_URL,
        "catalog_cache_path": str(
            cache_path if cache_path is not None else get_openrouter_price_cache_path()
        ),
        "catalog_fetched_at_utc": catalog.fetched_at_utc if catalog else None,
        "catalog_warning": catalog_warning,
        "exact_match_policy": {
            "strategy": "exact_case_insensitive",
            "match_fields": ["id", "canonical_slug", "hugging_face_id"],
            "fallback_normalizations": ["strip_common_local_quantization_suffixes"],
        },
        "phases": phase_summaries,
        "total": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "request_count": total_request_count,
            "openrouter_reference_cost_usd": total_reference_cost,
        },
    }


def format_openrouter_reference_pricing_summary(summary: dict[str, Any]) -> str:
    lines = ["OpenRouter reference pricing:"]
    for phase, phase_summary in summary["phases"].items():
        phase_cost = phase_summary.get("openrouter_reference_cost_usd")
        cost_str = f" | usd={phase_cost:.6f}" if phase_cost is not None else ""
        lines.append(
            "  "
            + f"{phase}: status={phase_summary['pricing_status']}"
            + f" | prompt={phase_summary['prompt_tokens']}"
            + f" | completion={phase_summary['completion_tokens']}"
            + f" | total={phase_summary['total_tokens']}"
            + cost_str
        )
    total = summary["total"]
    total_cost = total.get("openrouter_reference_cost_usd")
    total_cost_str = f" | usd={total_cost:.6f}" if total_cost is not None else ""
    lines.append(
        "  total:"
        + f" prompt={total['prompt_tokens']}"
        + f" | completion={total['completion_tokens']}"
        + f" | total={total['total_tokens']}"
        + total_cost_str
    )
    warning = summary.get("catalog_warning")
    if warning:
        lines.append(f"  warning: {warning}")
    return "\n".join(lines)


def refresh_openrouter_price_catalog(
    cache_path: str | Path | None = None,
) -> OpenRouterPriceCatalog:
    return load_openrouter_price_catalog(refresh=True, cache_path=cache_path)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m judgearena.openrouter_reference_pricing",
        description="Refresh or inspect the cached OpenRouter model pricing catalog.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force-refresh the cached OpenRouter models catalog.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional local model spec to resolve against the cached catalog.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    catalog = load_openrouter_price_catalog(refresh=args.refresh)
    print(
        json.dumps(
            {
                "catalog_source_url": catalog.source_url,
                "catalog_fetched_at_utc": catalog.fetched_at_utc,
                "catalog_cache_path": catalog.cache_path,
                "model_count": len(catalog.models),
                "matched_model": (
                    asdict(find_exact_openrouter_match(catalog, args.model))
                    if args.model
                    and find_exact_openrouter_match(catalog, args.model) is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
