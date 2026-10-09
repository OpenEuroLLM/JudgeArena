import json
from urllib.error import URLError

import pytest

from judgearena import pricing
from judgearena.pricing import TokenPrice, reference_cost, resolve_prices
from judgearena.usage import RequestUsage


def test_overrides_do_not_fetch_catalog(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("all prices were overridden")

    monkeypatch.setattr(pricing, "urlopen", fail)
    result = resolve_prices(
        ["VLLM/local", "OpenRouter/hosted"],
        {"VLLM/local": 1.2, "OpenRouter/hosted": {"input": 0.2, "output": 0.7}},
    )
    assert result["VLLM/local"] == TokenPrice(1.2, 1.2, "user_override")
    assert result["OpenRouter/hosted"] == TokenPrice(0.2, 0.7, "user_override")


def test_huggingface_id_match_precedes_suffix_for_every_provider(tmp_path, monkeypatch):
    records = [
        {
            "id": "org/model",
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        },
        {
            "id": "elsewhere/different",
            "hugging_face_id": "ORG/Model",
            "pricing": {"prompt": "0.000003", "completion": "0.000004"},
        },
    ]
    monkeypatch.setattr(pricing, "urlopen", lambda *a, **k: _Response(records))
    cache = tmp_path / "openrouter_pricing.json"
    for model in ("VLLM/org/model", "Provider/org/model"):
        result = resolve_prices([model], {}, catalog_cache=cache)
        assert result[model] == TokenPrice(3, 4, "openrouter_reference")
        assert json.loads(cache.read_text()) == records


def test_suffix_match(monkeypatch):
    monkeypatch.setattr(
        pricing,
        "urlopen",
        lambda *a, **k: _Response(
            [
                {
                    "id": "org/model",
                    "pricing": {"prompt": "0.000005", "completion": "0.000009"},
                }
            ]
        ),
    )
    assert resolve_prices(["Hosted/org/model"], {})["Hosted/org/model"] == TokenPrice(
        5, 9, "openrouter_reference"
    )


def test_fetch_failure_uses_cached_catalog(tmp_path, monkeypatch):
    cache = tmp_path / "openrouter_pricing.json"
    cache.write_text(
        json.dumps(
            [
                {
                    "id": "org/model",
                    "pricing": {"prompt": "0.000002", "completion": "0.000006"},
                }
            ]
        )
    )
    monkeypatch.setattr(pricing, "urlopen", lambda *a, **k: _raise_url_error())
    price = resolve_prices(["Backend/org/model"], {}, catalog_cache=cache)
    assert price["Backend/org/model"] == TokenPrice(2, 6, "openrouter_reference")


def test_fetch_failure_without_cache_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pricing, "urlopen", lambda *a, **k: _raise_url_error())
    with pytest.raises(ValueError, match="login node.*price_per_million_tokens"):
        resolve_prices(
            ["Backend/org/model"], {}, catalog_cache=tmp_path / "catalog.json"
        )


def test_unpriced_model_fails_for_cost_objective(monkeypatch):
    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("agreement-only pricing must not fetch the catalog")

    monkeypatch.setattr(pricing, "urlopen", unexpected_fetch)
    assert resolve_prices(["VLLM/unknown"], {}, require_cost=False) == {}

    monkeypatch.setattr(pricing, "urlopen", lambda *a, **k: _Response([]))
    with pytest.raises(ValueError, match="unknown.*overrides"):
        resolve_prices(["VLLM/unknown"], {})


def test_reference_cost_requires_native_counts_and_uses_reported_zero():
    with pytest.raises(
        ValueError, match="missing native input_tokens and output_tokens"
    ):
        reference_cost(
            RequestUsage(stage="judging", reasoning_tokens=12),
            TokenPrice(1, 2, "ref"),
        )

    usage = RequestUsage(
        stage="judging", input_tokens=0, output_tokens=3, reasoning_tokens=100
    )
    assert reference_cost(usage, TokenPrice(1_000_000, 2_000_000, "ref")) == 6

    with pytest.raises(ValueError, match="missing native output_tokens"):
        reference_cost(
            RequestUsage(stage="judging", input_tokens=3), TokenPrice(1, 2, "ref")
        )


def test_fetch_failure_rejects_legacy_dictionary_catalog(tmp_path, monkeypatch):
    cache = tmp_path / "catalog.json"
    cache.write_text(json.dumps({"org/model": [1, 2]}))
    monkeypatch.setattr(pricing, "urlopen", lambda *a, **k: _raise_url_error())
    with pytest.raises(ValueError, match="unsupported format.*refresh.*overrides"):
        resolve_prices(["Backend/org/model"], {}, catalog_cache=cache)


class _Response:
    def __init__(self, records):
        self.records = records

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps({"data": self.records}).encode()


def _raise_url_error():
    raise URLError("offline")
