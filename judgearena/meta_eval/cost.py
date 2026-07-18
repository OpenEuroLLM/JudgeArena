"""Cost estimation helpers for meta-evaluation annotations."""

from __future__ import annotations

import json

from judgearena.utils import data_root

_PRICING_CACHE_FILE = data_root / "cache" / "openrouter_pricing.json"
_openrouter_pricing_cache: dict[str, tuple[float, float]] = {}


def _openrouter_model_key(model_name: str) -> str:
    if model_name.count("/") >= 2:
        return "/".join(model_name.split("/")[-2:])
    return model_name


def load_openrouter_pricing() -> dict[str, tuple[float, float]]:
    if _openrouter_pricing_cache:
        return _openrouter_pricing_cache
    if not _PRICING_CACHE_FILE.exists():
        return {}
    with _PRICING_CACHE_FILE.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    for model_id, prices in raw.items():
        prompt, completion = prices
        _openrouter_pricing_cache[model_id] = (float(prompt), float(completion))
    return _openrouter_pricing_cache


def lookup_openrouter_pricing(model_name: str) -> tuple[float, float] | None:
    pricing = load_openrouter_pricing()
    key = _openrouter_model_key(model_name)
    return pricing.get(key)


def estimate_token_count(text: str) -> int:
    return len(text) // 4 if isinstance(text, str) else 0


def estimate_annotation_cost_usd(
    *,
    judge_input: str,
    judge_completion: str,
    judge_model: str,
) -> tuple[float | None, str]:
    """Estimate cost from text length and cached OpenRouter reference pricing."""
    pricing = lookup_openrouter_pricing(judge_model)
    if pricing is None:
        return None, "unavailable"

    input_price, output_price = pricing
    prompt_tokens = estimate_token_count(judge_input)
    completion_tokens = estimate_token_count(judge_completion)
    cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1e6
    return float(cost), "estimated"
