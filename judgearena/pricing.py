"""Shared reference token pricing and request cost calculation."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from judgearena.usage import RequestUsage

_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"


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


def _fetch_openrouter(catalog_cache: Path | None) -> list[dict]:
    try:
        with urlopen(_OPENROUTER_URL, timeout=10) as response:
            records = json.load(response)["data"]
        if catalog_cache is not None:
            catalog_cache.parent.mkdir(parents=True, exist_ok=True)
            catalog_cache.write_text(json.dumps(records), encoding="utf-8")
        return records
    except (URLError, TimeoutError, OSError) as error:
        if catalog_cache is not None and catalog_cache.exists():
            cached = json.loads(catalog_cache.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                return cached
            raise ValueError(
                "Cached OpenRouter pricing catalog uses an unsupported format; "
                "refresh the catalog or set price_per_million_tokens overrides"
            ) from error
        raise ValueError(
            "Could not fetch OpenRouter pricing and no cached catalog is available; "
            "fetch the catalog on a login node or set "
            "price_per_million_tokens overrides"
        ) from error


def _find_price(model: str, records: list[dict]) -> TokenPrice | None:
    requested_id = model.split("/", 1)[-1].lower()
    suffix = "/".join(requested_id.split("/")[-2:])
    record = next(
        (
            item
            for item in records
            if str(item.get("hugging_face_id") or "").lower() == requested_id
        ),
        None,
    )
    if record is None:
        record = next(
            (item for item in records if str(item.get("id") or "").lower() == suffix),
            None,
        )
    if record is None:
        return None
    rates = record.get("pricing", {})
    if "prompt" not in rates or "completion" not in rates:
        return None
    return TokenPrice(
        float(rates["prompt"]) * 1_000_000,
        float(rates["completion"]) * 1_000_000,
        "openrouter_reference",
    )


def resolve_prices(
    models,
    overrides,
    *,
    catalog_cache: Path | None = None,
    require_cost=True,
) -> dict[str, TokenPrice]:
    """Resolve explicit or OpenRouter reference USD-per-million-token rates."""
    if not require_cost:
        return {}
    prices = {}
    unresolved = []
    for model in sorted(models):
        rate = overrides.get(model)
        if rate is not None:
            if isinstance(rate, (int, float)):
                rate = {"input": rate, "output": rate}
            prices[model] = TokenPrice(
                float(rate["input"]), float(rate["output"]), "user_override"
            )
        else:
            unresolved.append(model)
    records = _fetch_openrouter(catalog_cache) if unresolved else []
    for model in unresolved:
        price = _find_price(model, records)
        if price is not None:
            prices[model] = price
    missing = sorted(set(models) - prices.keys())
    if missing:
        raise ValueError(
            f"No token price for {missing}; set price_per_million_tokens "
            f"overrides for: {missing}"
        )
    return prices
