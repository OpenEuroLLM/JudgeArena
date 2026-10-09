import json
from contextlib import nullcontext
from io import BytesIO
from urllib.error import URLError

import pytest

from judgearena import pricing
from judgearena.pricing import TokenPrice, reference_cost, resolve_prices
from judgearena.usage import RequestUsage


@pytest.fixture
def catalog():
    return [
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


@pytest.mark.parametrize(
    ("model", "hf_match", "expected"),
    [
        ("VLLM/org/model", True, TokenPrice(3, 4, "openrouter_reference")),
        ("Hosted/org/model", True, TokenPrice(3, 4, "openrouter_reference")),
        ("Gateway/org/model", True, TokenPrice(3, 4, "openrouter_reference")),
        ("VLLM/org/model", False, TokenPrice(1, 2, "openrouter_reference")),
    ],
)
def test_reference_model_matching(
    model, hf_match, expected, catalog, monkeypatch, tmp_path
):
    records = catalog if hf_match else catalog[:1]
    monkeypatch.setattr(
        pricing,
        "urlopen",
        lambda *a, **k: nullcontext(BytesIO(json.dumps({"data": records}).encode())),
    )
    cache = tmp_path / "catalog.json"
    assert resolve_prices([model], {}, catalog_cache=cache) == {model: expected}
    assert json.loads(cache.read_text()) == records


@pytest.mark.parametrize(
    "cache_content,error",
    [
        ("catalog", None),
        (None, "login node.*price_per_million_tokens"),
        ({"org/model": [1, 2]}, "unsupported format.*refresh.*overrides"),
    ],
)
def test_offline_catalog_resolution(
    tmp_path, monkeypatch, catalog, cache_content, error
):
    cache = tmp_path / "catalog.json"
    if cache_content is not None:
        cache.write_text(
            json.dumps(catalog if cache_content == "catalog" else cache_content)
        )
    monkeypatch.setattr(pricing, "urlopen", lambda *a, **k: _raise_url_error())
    if error:
        with pytest.raises(ValueError, match=error):
            resolve_prices(["VLLM/org/model"], {}, catalog_cache=cache)
    else:
        assert resolve_prices(["VLLM/org/model"], {}, catalog_cache=cache) == {
            "VLLM/org/model": TokenPrice(3, 4, "openrouter_reference")
        }


def test_unpriced_model_fails_for_cost_objective(monkeypatch):
    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("agreement-only pricing must not fetch the catalog")

    monkeypatch.setattr(pricing, "urlopen", unexpected_fetch)
    assert resolve_prices(["VLLM/unknown"], {}, require_cost=False) == {}

    monkeypatch.setattr(
        pricing,
        "urlopen",
        lambda *a, **k: nullcontext(BytesIO(b'{"data": []}')),
    )
    with pytest.raises(ValueError, match="unknown.*overrides"):
        resolve_prices(["VLLM/unknown"], {})


@pytest.mark.parametrize(
    ("usage", "missing"),
    [
        (
            RequestUsage(stage="judging", reasoning_tokens=12),
            "input_tokens and output_tokens",
        ),
        (RequestUsage(stage="judging", input_tokens=3), "output_tokens"),
    ],
)
def test_reference_cost_requires_native_counts(usage, missing):
    with pytest.raises(ValueError, match=f"missing native {missing}"):
        reference_cost(usage, TokenPrice(1, 2, "ref"))


def test_reference_cost_uses_native_counts_including_zero():
    usage = RequestUsage(
        stage="judging", input_tokens=0, output_tokens=3, reasoning_tokens=100
    )
    assert reference_cost(usage, TokenPrice(1_000_000, 2_000_000, "ref")) == 6


def _raise_url_error():
    raise URLError("offline")
