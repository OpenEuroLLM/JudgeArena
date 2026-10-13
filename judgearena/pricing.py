"""Shared reference token pricing and request cost calculation."""

from dataclasses import dataclass

from judgearena.usage import RequestUsage


@dataclass(frozen=True)
class TokenPrice:
    input: float
    output: float
    source: str


def reference_cost(usage: RequestUsage, price: TokenPrice) -> float:
    """Calculate USD from native input/output token counts and per-million rates."""
    missing = [
        name
        for name in ("input_tokens", "output_tokens")
        if getattr(usage, name) is None
    ]
    if missing:
        fields = " and ".join(missing)
        raise ValueError(
            f"Cannot calculate reference cost: missing native {fields}; "
            "provide provider-reported input_tokens and output_tokens usage"
        )
    return (
        usage.input_tokens * price.input + usage.output_tokens * price.output
    ) / 1_000_000


def resolve_prices(models, overrides, *, require_cost=True) -> dict[str, TokenPrice]:
    """Resolve explicit USD-per-million-token rates before searching."""
    if not require_cost:
        return {}
    prices = {}
    for model in sorted(models):
        if model not in overrides:
            continue
        rate = overrides[model]
        if isinstance(rate, (int, float)):
            rate = {"input": rate, "output": rate}
        prices[model] = TokenPrice(rate["input"], rate["output"], "user_override")
    missing = sorted(set(models) - prices.keys())
    if missing:
        raise ValueError(
            f"No token price for {missing}; set tune_judge.price_per_million_tokens"
        )
    return prices
