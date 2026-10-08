import json

import pytest

from judgearena.tuning import pricing
from judgearena.tuning.pricing import TokenPrice, resolve_prices


def test_source_prices_exact_model_identifiers(tmp_path):
    prices = resolve_prices(
        [
            "VLLM/Qwen/Qwen2.5-7B-Instruct",
            "VLLM/google/gemma-2-9b-it",
            "VLLM/google/gemma-2-27b-it",
            "VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        ],
        {},
        tmp_path,
    )
    assert prices["VLLM/Qwen/Qwen2.5-7B-Instruct"] == TokenPrice(
        0.12, 0.12, "measured-runtime"
    )
    assert prices["VLLM/google/gemma-2-9b-it"].input == 0.14
    assert prices["VLLM/google/gemma-2-27b-it"].input == 0.30
    assert prices["VLLM/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"].input == 0.36


@pytest.mark.parametrize(
    "count, expected",
    [(3, 0.06), (4, 0.10), (8, 0.20), (21, 0.30), (41, 0.80), (70, 0.90), (72, 1.20)],
)
def test_parameter_tiers(count, expected):
    tiers = json.loads(pricing._TABLE.read_text())["tiers"]
    assert pricing._tier_price(count, tiers) == expected


def _write_safetensors(path, shape):
    header = {"weight": {"dtype": "I8", "shape": shape, "data_offsets": [0, 1]}}
    encoded = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(
        len(encoded).to_bytes(8, "little") + encoded
    )


def test_offline_hf_cache_snapshot_counts_shape_products(tmp_path, monkeypatch):
    snapshot = tmp_path / "models--org--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    _write_safetensors(snapshot, [2_000_000_000])
    monkeypatch.setattr(
        pricing, "try_to_load_from_cache", lambda *a, **k: snapshot / "config.json"
    )
    result = resolve_prices(["VLLM/org/model"], {}, tmp_path / "store")
    assert result["VLLM/org/model"].input == 0.06


def test_quantized_model_without_exact_entry_has_no_tier_price(tmp_path, monkeypatch):
    snapshot = tmp_path / "quantized"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"quantization_config": {"bits": 4}})
    )
    monkeypatch.setattr(
        pricing, "try_to_load_from_cache", lambda *a, **k: snapshot / "config.json"
    )
    with pytest.raises(ValueError, match="quantized-model"):
        resolve_prices(["VLLM/quantized-model"], {}, tmp_path)


def test_override_numeric_and_asymmetric(tmp_path, monkeypatch):
    def fail(_):
        raise AssertionError("override must avoid hosted pricing fetch")

    monkeypatch.setattr(pricing, "_fetch_openrouter", fail)
    result = resolve_prices(
        ["OpenRouter/org/a", "OpenAI/b"],
        {"OpenRouter/org/a": 0.4, "OpenAI/b": {"input": 0.2, "output": 0.9}},
        tmp_path,
    )
    assert result["OpenRouter/org/a"] == TokenPrice(0.4, 0.4, "user_override")
    assert result["OpenAI/b"] == TokenPrice(0.2, 0.9, "user_override")


def test_unknown_required_fails_but_agreement_only_omits(tmp_path):
    with pytest.raises(ValueError, match="unlisted"):
        resolve_prices(["VLLM/unlisted"], {}, tmp_path)
    assert resolve_prices(["VLLM/unlisted"], {}, tmp_path, require_cost=False) == {}


def test_openrouter_matches_hf_id_first_then_model_id(tmp_path, monkeypatch):
    records = [
        {
            "id": "provider/other",
            "hugging_face_id": "org/model",
            "pricing": {"prompt": "0.000002", "completion": "0.000007"},
        },
        {
            "id": "org/fallback",
            "pricing": {"prompt": "0.000003", "completion": "0.000009"},
        },
    ]
    monkeypatch.setattr(pricing, "_fetch_openrouter", lambda path: records)
    result = resolve_prices(
        ["OpenRouter/org/model", "Provider/org/fallback"], {}, tmp_path
    )
    assert result["OpenRouter/org/model"] == TokenPrice(
        2.0, 7.0, "openrouter_reference"
    )
    assert result["Provider/org/fallback"] == TokenPrice(
        3.0, 9.0, "openrouter_reference"
    )


def test_nonhosted_unlisted_model_does_not_fetch(tmp_path, monkeypatch):
    def fail(_):
        raise AssertionError("unexpected hosted pricing fetch")

    monkeypatch.setattr(pricing, "_fetch_openrouter", fail)
    assert resolve_prices(["VLLM/unlisted"], {}, tmp_path, require_cost=False) == {}


def test_legacy_openrouter_cache_mapping_is_supported(tmp_path):
    cache_file = tmp_path / "openrouter_pricing.json"
    cache_file.write_text(json.dumps({"org/model": [0.000002, 0.000007]}))
    records = pricing._fetch_openrouter(cache_file)
    assert pricing._hosted_price("org/model", records) == TokenPrice(
        2.0, 7.0, "openrouter_reference"
    )


def test_hf_cache_missing_sentinel_is_not_a_path(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pricing, "try_to_load_from_cache", lambda *a, **k: sentinel)
    assert pricing._repo_snapshot("org/missing") is None
