import json
from contextlib import nullcontext
from io import BytesIO

import pytest

from judgearena import pricing
from judgearena.pricing import TokenPrice, reference_cost, resolve_prices
from judgearena.usage import RequestUsage


def test_override_and_agreement_only_skip_catalog(monkeypatch):
    monkeypatch.setattr(pricing, "urlopen", lambda *a, **k: pytest.fail("fetched"))
    assert resolve_prices(["VLLM/local"], {"VLLM/local": 1.2}) == {
        "VLLM/local": TokenPrice(1.2, 1.2, "user_override")
    }
    assert resolve_prices(["VLLM/local"], {}, require_cost=False) == {}


@pytest.mark.parametrize(
    "provider,hf_match", [("VLLM", True), ("OpenRouter", True), ("VLLM", False)]
)
def test_reference_matching_and_catalog_snapshot(
    provider, hf_match, monkeypatch, tmp_path
):
    records = [dict(id="org/model", pricing=dict(prompt="1e-6", completion="2e-6"))]
    if hf_match:
        records.append(
            dict(
                id="other/model",
                hugging_face_id="ORG/Model",
                pricing=dict(prompt="3e-6", completion="4e-6"),
            )
        )
    response = json.dumps({"data": records}).encode()
    monkeypatch.setattr(
        pricing, "urlopen", lambda *a, **k: nullcontext(BytesIO(response))
    )
    model, cache = f"{provider}/org/model", tmp_path / "prices.json"
    expected = (
        TokenPrice(3, 4, "openrouter_reference")
        if hf_match
        else TokenPrice(1, 2, "openrouter_reference")
    )
    assert resolve_prices([model], {}, catalog_cache=cache)[model] == expected
    assert json.loads(cache.read_text()) == records
    with pytest.raises(ValueError, match="overrides"):
        resolve_prices(["VLLM/unknown"], {}, catalog_cache=cache)


@pytest.mark.parametrize(
    "tokens,expected", [((0, 3), 6), ((2, 1), 4), ((None, 3), None)]
)
def test_native_counts_and_no_double_counted_reasoning(tokens, expected):
    usage = RequestUsage(
        stage="judging",
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        reasoning_tokens=100,
    )
    price = TokenPrice(1_000_000, 2_000_000, "ref")
    if expected is None:
        with pytest.raises(ValueError, match="missing native"):
            reference_cost(usage, price)
    else:
        assert reference_cost(usage, price) == expected
