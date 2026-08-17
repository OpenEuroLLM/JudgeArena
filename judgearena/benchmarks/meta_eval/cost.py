"""Token counts and OpenRouter reference-pricing estimates for meta-eval."""

from __future__ import annotations

import json
import math

import pandas as pd
from pydantic import BaseModel

from judgearena.log import get_logger
from judgearena.paths import data_root

logger = get_logger(__name__)

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
    return load_openrouter_pricing().get(_openrouter_model_key(model_name))


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


class AnnotationTelemetry(BaseModel):
    """Token and cost rollup over the judge passes of one meta-eval run."""

    judge_passes_per_battle: int
    """1 for swap_mode=fixed, 2 for swap_mode=both."""
    estimated_input_tokens: int
    """Sum of chars/4 token estimates over judge inputs."""
    estimated_output_tokens: int
    """Sum of chars/4 token estimates over judge completions."""
    token_count_source: str
    """How tokens were counted (chars/4; never a provider usage API)."""
    total_cost_usd: float
    """OpenRouter reference price applied to the token estimates; NaN (written as
    null) when no local pricing covers the judge."""
    cost_per_1k_judgements_usd: float
    """Mean estimated USD per 1,000 judge passes, NaN if pricing is missing."""
    cost_source_counts: dict[str, int]
    """Per-row cost_source value counts (estimated vs unavailable)."""


def _column_sum(df_ann: pd.DataFrame, column: str) -> int:
    return int(pd.to_numeric(df_ann[column], errors="coerce").fillna(0).sum())


def annotation_telemetry(
    df_ann: pd.DataFrame, *, swap_mode: str
) -> AnnotationTelemetry:
    """Roll up per-pass token and cost columns. ``swap_mode=both`` is two rows per battle."""
    costs = pd.to_numeric(df_ann["cost_usd"], errors="coerce").dropna()
    sources = df_ann["cost_source"].dropna()
    if costs.empty:
        logger.warning(
            "OpenRouter reference pricing is unavailable; cost fields will be null."
        )
    return AnnotationTelemetry(
        judge_passes_per_battle=2 if swap_mode == "both" else 1,
        estimated_input_tokens=_column_sum(df_ann, "estimated_input_tokens"),
        estimated_output_tokens=_column_sum(df_ann, "estimated_output_tokens"),
        token_count_source="estimated_chars_div_4",
        # NaN rather than None so the keys survive the report's exclude_none dump
        # and still serialize as JSON null, like the NaN kappa of a degenerate run.
        total_cost_usd=float(costs.sum()) if not costs.empty else math.nan,
        cost_per_1k_judgements_usd=(
            float(costs.mean() * 1000) if not costs.empty else math.nan
        ),
        cost_source_counts={
            str(source): int(count) for source, count in sources.value_counts().items()
        },
    )
