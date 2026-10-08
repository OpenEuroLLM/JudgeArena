"""Resolve estimated judge-token prices from explicit reference sources."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

_TABLE = Path(__file__).with_name("prices.json")
_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
_LOCAL = {"vllm", "llamacpp", "dummy"}


@dataclass(frozen=True)
class TokenPrice:
    input: float
    output: float
    source: str


def _model_id(model: str) -> str:
    return model.split("/", 1)[-1] if "/" in model else model


def _hosted(model: str) -> bool:
    provider = model.split("/", 1)[0].lower() if "/" in model else ""
    return bool(provider and provider not in _LOCAL)


def _repo_snapshot(model_id: str) -> Path | None:
    path = Path(model_id).expanduser()
    if path.is_dir():
        return path
    try:
        config = try_to_load_from_cache(model_id, "config.json", repo_type="model")
    except (OSError, ValueError):
        config = None
    if isinstance(config, (str, Path)):
        return Path(config).parent
    return None


def _parameter_count(model_id: str) -> float | None:
    path = _repo_snapshot(model_id)
    if path is None:
        return None
    quant_config = path / "config.json"
    if quant_config.exists():
        config = json.loads(quant_config.read_text(encoding="utf-8"))
        if config.get("quantization_config"):
            return None
    index = path / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        shards = [p.name for p in path.glob("*.safetensors")]
    total = 0
    for shard in shards:
        with (path / shard).open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(size))
        total += sum(
            _shape_size(tensor["shape"])
            for name, tensor in header.items()
            if name != "__metadata__"
        )
    return total / 1e9 if total else None


def _shape_size(shape: list[int]) -> int:
    product = 1
    for dim in shape:
        product *= dim
    return product


def _tier_price(params_b: float, tiers: list) -> float | None:
    for upper, price in tiers:
        if upper is None or params_b <= upper:
            return float(price)
    return None


def _fetch_openrouter(cache_file: Path) -> list[dict]:
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if isinstance(cached, dict):
            return [
                {
                    "id": model_id,
                    "pricing": {"prompt": prices[0], "completion": prices[1]},
                }
                for model_id, prices in cached.items()
            ]
        return cached
    with urllib.request.urlopen(_OPENROUTER_URL, timeout=30) as response:
        records = json.load(response)["data"]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(records), encoding="utf-8")
    return records


def _hosted_price(model_id: str, records: list[dict]) -> TokenPrice | None:
    target = model_id.lower()
    suffix = "/".join(target.split("/")[-2:])
    match = next(
        (r for r in records if str(r.get("hugging_face_id") or "").lower() == target),
        None,
    )
    if match is None:
        match = next(
            (r for r in records if str(r.get("id") or "").lower() == suffix), None
        )
    if match is None:
        return None
    pricing = match.get("pricing", {})
    if "prompt" not in pricing or "completion" not in pricing:
        return None
    return TokenPrice(
        float(pricing["prompt"]) * 1e6,
        float(pricing["completion"]) * 1e6,
        "openrouter_reference",
    )


def resolve_prices(
    models: list[str] | set[str] | tuple[str, ...],
    overrides: dict[str, float | dict[str, float]],
    store_root: str | Path | None,
    require_cost: bool = True,
) -> dict[str, TokenPrice]:
    """Resolve USD/M-token estimates; never reports billed or measured cost."""
    table = json.loads(_TABLE.read_text(encoding="utf-8"))
    specs = table["models"]
    resolved: dict[str, TokenPrice] = {}
    pending = []
    for model in sorted(set(models)):
        override = overrides.get(model)
        if override is not None:
            pair = (
                {"input": override, "output": override}
                if isinstance(override, (int, float))
                else override
            )
            if "input" in pair and "output" in pair:
                resolved[model] = TokenPrice(
                    float(pair["input"]), float(pair["output"]), "user_override"
                )
                continue
        provider = model.split("/", 1)[0].lower() if "/" in model else ""
        model_id = _model_id(model)
        spec = specs.get(model_id)
        if spec and provider == "vllm":
            resolved[model] = TokenPrice(float(spec[0]), float(spec[1]), spec[2])
        elif _hosted(model):
            pending.append((model, model_id))
        elif provider == "vllm":
            count = _parameter_count(model_id)
            price = _tier_price(count, table["tiers"]) if count is not None else None
            if price is not None:
                resolved[model] = TokenPrice(
                    price, price, "historical-size-tier-reference"
                )
    if pending:
        if store_root is None:
            raise ValueError(
                "store_root is required to cache hosted model reference pricing"
            )
        records = _fetch_openrouter(Path(store_root) / "openrouter_pricing.json")
        for model, model_id in pending:
            price = _hosted_price(model_id, records)
            if price:
                resolved[model] = price
    missing = sorted(set(models) - set(resolved))
    if missing and require_cost:
        raise ValueError(f"No token price could be resolved for: {missing}")
    return resolved
