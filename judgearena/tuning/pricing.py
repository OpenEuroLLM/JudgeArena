"""Explicit judge-token prices used by the tuning cost objective."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    input: float
    output: float
    source: str


def resolve_prices(models, overrides, *, require_cost=True) -> dict[str, TokenPrice]:
    """Resolve explicit USD-per-million-token rates before searching."""
    prices = {}
    for model in sorted(models):
        if model not in overrides:
            continue
        rate = overrides[model]
        if isinstance(rate, (int, float)):
            rate = {"input": rate, "output": rate}
        prices[model] = TokenPrice(rate["input"], rate["output"], "user_override")
    missing = sorted(set(models) - prices.keys())
    if missing and require_cost:
        raise ValueError(
            f"No token price for {missing}; set tune_judge.price_per_million_tokens"
        )
    return prices
