import json

import judgearena.openrouter_reference_pricing as pricing
from judgearena.repro import write_run_metadata
from judgearena.utils import do_inference


class CountingModel:
    def batch(self, inputs, **invoke_kwargs):
        return [f"output-{idx}" for idx, _input in enumerate(inputs)]

    def count_prompt_tokens_batch(self, inputs):
        return [len(str(input_item)) for input_item in inputs]

    def count_completion_tokens_batch(self, outputs):
        return [len(output) for output in outputs]


def test_do_inference_records_token_usage():
    tracker = pricing.OpenRouterReferencePricingTracker()
    model = CountingModel()

    outputs = do_inference(
        chat_model=model,
        inputs=["abc", "de"],
        usage_tracker=tracker,
        usage_phase="judge",
        usage_model_spec="VLLM/org/model",
    )

    assert outputs == ["output-0", "output-1"]
    assert tracker.records == [
        pricing.TokenUsageRecord(
            phase="judge",
            model_spec="VLLM/org/model",
            prompt_tokens=3,
            completion_tokens=8,
            requests=1,
        ),
        pricing.TokenUsageRecord(
            phase="judge",
            model_spec="VLLM/org/model",
            prompt_tokens=2,
            completion_tokens=8,
            requests=1,
        ),
    ]


class _FakeAIMessage:
    """Minimal AIMessage stand-in: .content + langchain-core .usage_metadata."""

    def __init__(self, content: str, usage_metadata: dict[str, int] | None) -> None:
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata: dict[str, object] = {}


class _OpenRouterShapeModel:
    """Mimics ChatOpenAI: returns AIMessage objects with usage_metadata, no
    count_*_batch helpers (so the fallback tokeniser path is unavailable)."""

    def __init__(self, usages: list[dict[str, int] | None]) -> None:
        self._usages = usages

    def batch(self, inputs, **invoke_kwargs):
        return [
            _FakeAIMessage(content=f"output-{idx}", usage_metadata=self._usages[idx])
            for idx, _ in enumerate(inputs)
        ]


def test_do_inference_records_usage_metadata_for_openrouter_shape_models():
    """For OpenRouter ChatOpenAI calls, the tracker must record API-reported
    token counts pulled from AIMessage.usage_metadata; ``record_batch_from_model``
    is a no-op because ChatOpenAI lacks ``count_*_batch`` helpers."""
    tracker = pricing.OpenRouterReferencePricingTracker()
    model = _OpenRouterShapeModel(
        usages=[
            {"input_tokens": 1234, "output_tokens": 17, "total_tokens": 1251},
            {"input_tokens": 800, "output_tokens": 42, "total_tokens": 842},
        ]
    )

    outputs = do_inference(
        chat_model=model,
        inputs=["prompt-1", "prompt-2"],
        usage_tracker=tracker,
        usage_phase="judge",
        usage_model_spec="OpenRouter/google/gemma-4-31b-it",
    )

    assert outputs == ["output-0", "output-1"]
    assert tracker.records == [
        pricing.TokenUsageRecord(
            phase="judge",
            model_spec="OpenRouter/google/gemma-4-31b-it",
            prompt_tokens=1234,
            completion_tokens=17,
            requests=1,
        ),
        pricing.TokenUsageRecord(
            phase="judge",
            model_spec="OpenRouter/google/gemma-4-31b-it",
            prompt_tokens=800,
            completion_tokens=42,
            requests=1,
        ),
    ]


def test_do_inference_falls_back_to_count_batch_when_usage_metadata_missing():
    """When AIMessage results carry no ``usage_metadata`` (e.g. local vLLM path),
    ``do_inference`` must fall back to ``record_batch_from_model`` and use the
    chat_model's ``count_*_batch`` helpers."""
    tracker = pricing.OpenRouterReferencePricingTracker()
    model = CountingModel()

    do_inference(
        chat_model=model,
        inputs=["abc", "de"],
        usage_tracker=tracker,
        usage_phase="generation_model_A",
        usage_model_spec="VLLM/org/model",
    )

    assert [(r.prompt_tokens, r.completion_tokens) for r in tracker.records] == [
        (3, 8),
        (2, 8),
    ]


def test_record_batch_from_usage_metadata_returns_false_on_all_none():
    """A batch where every entry is ``None`` must signal "no records added" so
    the caller can fall through to the tokeniser-based path."""
    tracker = pricing.OpenRouterReferencePricingTracker()

    recorded = tracker.record_batch_from_usage_metadata(
        phase="judge",
        model_spec="OpenRouter/google/gemma-4-31b-it",
        usages=[None, None, None],
    )

    assert recorded is False
    assert tracker.records == []


def test_record_batch_from_usage_metadata_accepts_openai_shape_keys():
    """OpenAI-shape keys (``prompt_tokens``/``completion_tokens``) appear on
    ``response_metadata.token_usage`` for older langchain-openai versions."""
    tracker = pricing.OpenRouterReferencePricingTracker()

    recorded = tracker.record_batch_from_usage_metadata(
        phase="judge",
        model_spec="OpenRouter/google/gemma-4-31b-it",
        usages=[{"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}],
    )

    assert recorded is True
    assert tracker.records == [
        pricing.TokenUsageRecord(
            phase="judge",
            model_spec="OpenRouter/google/gemma-4-31b-it",
            prompt_tokens=100,
            completion_tokens=25,
            requests=1,
        )
    ]


def test_build_reference_pricing_summary_uses_exact_match_and_reports_partial_cost(
    monkeypatch,
):
    catalog = pricing.parse_openrouter_catalog_payload(
        {
            "data": [
                {
                    "id": "openrouter/example-model",
                    "canonical_slug": "openrouter/example-model",
                    "hugging_face_id": "Org/Example-Model",
                    "name": "Example Model",
                    "pricing": {
                        "prompt": "0.001",
                        "completion": "0.002",
                        "request": "0.01",
                        "internal_reasoning": "0.5",
                    },
                }
            ],
            "fetched_at_utc": "2026-04-07T00:00:00+00:00",
        }
    )
    monkeypatch.setattr(
        pricing,
        "load_openrouter_price_catalog_with_fallback",
        lambda **kwargs: (catalog, None),
    )

    tracker = pricing.OpenRouterReferencePricingTracker()
    tracker._records.extend(
        [
            pricing.TokenUsageRecord(
                phase="generation_model_A",
                model_spec="VLLM/Org/Example-Model",
                prompt_tokens=100,
                completion_tokens=20,
            ),
            pricing.TokenUsageRecord(
                phase="generation_model_A",
                model_spec="VLLM/Org/Example-Model",
                prompt_tokens=50,
                completion_tokens=5,
            ),
            pricing.TokenUsageRecord(
                phase="generation_model_B",
                model_spec="VLLM/No/Match",
                prompt_tokens=10,
                completion_tokens=2,
            ),
        ]
    )

    summary = pricing.build_openrouter_reference_pricing_summary(
        tracker=tracker,
        phase_model_specs={
            "generation_model_A": "VLLM/Org/Example-Model",
            "generation_model_B": "VLLM/No/Match",
            "judge": "VLLM/No/Runtime",
        },
    )

    matched = summary["phases"]["generation_model_A"]
    assert matched["openrouter_model_id"] == "openrouter/example-model"
    assert matched["pricing_status"] == "matched_exact_openrouter_model_partial"
    assert matched["prompt_tokens"] == 150
    assert matched["completion_tokens"] == 25
    assert matched["request_count"] == 2
    assert matched["openrouter_reference_cost_usd"] == 0.22
    assert matched["ignored_pricing_components"] == ["internal_reasoning"]

    unmatched = summary["phases"]["generation_model_B"]
    assert unmatched["pricing_status"] == "no_exact_openrouter_match"
    assert unmatched["openrouter_reference_cost_usd"] is None

    no_runtime = summary["phases"]["judge"]
    assert no_runtime["pricing_status"] == "no_runtime_token_data"
    assert no_runtime["total_tokens"] == 0

    assert summary["total"]["openrouter_reference_cost_usd"] == 0.22


def test_build_reference_pricing_summary_matches_quantized_local_variant(
    monkeypatch,
):
    catalog = pricing.parse_openrouter_catalog_payload(
        {
            "data": [
                {
                    "id": "qwen/qwen3.5-27b",
                    "canonical_slug": "qwen/qwen3.5-27b",
                    "hugging_face_id": "Qwen/Qwen3.5-27B",
                    "name": "Qwen: Qwen3.5-27B",
                    "pricing": {
                        "prompt": "0.001",
                        "completion": "0.002",
                        "request": "0.01",
                    },
                }
            ],
            "fetched_at_utc": "2026-04-15T00:00:00+00:00",
        }
    )
    monkeypatch.setattr(
        pricing,
        "load_openrouter_price_catalog_with_fallback",
        lambda **kwargs: (catalog, None),
    )

    tracker = pricing.OpenRouterReferencePricingTracker()
    tracker._records.append(
        pricing.TokenUsageRecord(
            phase="generation_model_A",
            model_spec="VLLM/Qwen/Qwen3.5-27B-FP8",
            prompt_tokens=10,
            completion_tokens=5,
        )
    )

    summary = pricing.build_openrouter_reference_pricing_summary(
        tracker=tracker,
        phase_model_specs={
            "generation_model_A": "VLLM/Qwen/Qwen3.5-27B-FP8",
            "judge": "VLLM/No/Runtime",
        },
    )

    matched = summary["phases"]["generation_model_A"]
    assert (
        matched["pricing_status"]
        == "matched_openrouter_model_after_variant_normalization"
    )
    assert matched["openrouter_model_id"] == "qwen/qwen3.5-27b"
    assert matched["openrouter_reference_cost_usd"] == 0.03
    assert summary["exact_match_policy"]["fallback_normalizations"] == [
        "strip_common_local_quantization_suffixes"
    ]


def test_write_run_metadata_includes_pricing_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "judgearena.repro._get_dependency_versions",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr("judgearena.repro._get_git_hash", lambda *args, **kwargs: None)

    metadata_path = write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test",
        run={"dataset": "alpaca-eval"},
        pricing_reference={
            "pricing_model": "openrouter_reference",
            "total": {"openrouter_reference_cost_usd": 1.23},
        },
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["pricing_reference"]["pricing_model"] == "openrouter_reference"
    assert (
        metadata["pricing_reference"]["total"]["openrouter_reference_cost_usd"] == 1.23
    )
