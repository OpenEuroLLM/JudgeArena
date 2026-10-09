"""Explicit judge-token prices used by the tuning cost objective."""

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class TokenPrice:
    input: float
    output: float
    source: str


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
            if isinstance(cached, dict):
                return [
                    {
                        "id": model_id,
                        "pricing": {"prompt": price[0], "completion": price[1]},
                    }
                    for model_id, price in cached.items()
                ]
            if isinstance(cached, list):
                return cached
        raise ValueError(
            "Could not fetch OpenRouter pricing and no cached catalog is available; "
            "fetch the catalog on a login node or set "
            "tune_judge.price_per_million_tokens"
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
    if missing and require_cost:
        raise ValueError(
            f"No token price for {missing}; set tune_judge.price_per_million_tokens "
            f"overrides for: {missing}"
        )
    return prices
