"""Tests for the strip-thinking-before-char-cap fix applied to turn-1
answers when constructing the turn-2 prompt in MT-Bench generation.

Background: ``truncate_all_input_chars`` fires before the chat template
renders the turn-2 prompt. If the turn-1 answer contains a
``<think>...</think>`` block (Qwen3.5, SmolLM3 thinking mode) or a vLLM
forced-budget closer, a char cap landing inside the reasoning span would
destroy the ``</think>`` tag and force the full reasoning fragment into the
turn-2 context. Stripping the visible reasoning span first mirrors what
the Qwen3 chat template does natively for historical assistant turns and
keeps the cap on the visible answer.

These tests pin down the composition that
``judgearena.generate.generate_multiturn`` performs:
``strip_thinking_tags_with_metadata`` -> ``truncate_with_metadata``.
"""

from __future__ import annotations

from dataclasses import replace

from judgearena.cli_common import BaseCliArgs
from judgearena.generate_and_evaluate import CliArgs
from judgearena.mt_bench.mt_bench_utils import _mt_bench_generation_cache_name
from judgearena.utils import (
    VLLM_REASONING_END_STR,
    strip_thinking_tags_with_metadata,
    truncate_with_metadata,
)


def _strip_then_cap(
    answer: str, cap: int, *, strip: bool = True
) -> tuple[str, bool, bool]:
    """Reproduce the exact sequence inside ``generate_multiturn``'s turn-2 loop."""
    if strip:
        stripped_text, thinking_stripped = strip_thinking_tags_with_metadata(answer)
    else:
        stripped_text, thinking_stripped = answer, False
    truncated, was_truncated = truncate_with_metadata(stripped_text, max_len=cap)
    return truncated, was_truncated, thinking_stripped


def test_well_formed_think_block_is_stripped_before_cap():
    """Nominal Qwen3.5 case: a complete ``<think>...</think>`` wrapper sits
    in front of the visible answer. Stripping removes the whole span; the
    char cap then applies to the visible answer only."""
    reasoning = "so let me think through this... " * 400  # ~12K chars
    visible = "The capital of France is Paris."
    answer = f"<think>{reasoning}</think>\n\n{visible}"
    # Cap below the reasoning length but above the visible answer length
    # so the old behaviour would have clipped inside <think>.
    cap = 1024

    truncated, was_truncated, thinking_stripped = _strip_then_cap(answer, cap)

    assert thinking_stripped is True
    assert was_truncated is False
    assert truncated == visible
    assert "<think>" not in truncated
    assert "</think>" not in truncated


def test_vllm_forced_thinking_budget_closer_is_stripped():
    """When the thinking budget is exhausted, vLLM inserts a forced closer
    (``VLLM_REASONING_END_STR``) without a paired ``<think>`` opener. The
    strip helper treats everything up to and including that marker as
    reasoning and drops it before the cap fires."""
    forced_reasoning = "step 1... " * 500  # ~5K chars of runaway thought
    visible = "Final answer: 42."
    answer = f"<think>{forced_reasoning}{VLLM_REASONING_END_STR}{visible}"

    truncated, was_truncated, thinking_stripped = _strip_then_cap(answer, cap=256)

    assert thinking_stripped is True
    assert was_truncated is False
    assert truncated == visible


def test_dangling_closing_tag_is_stripped():
    """Qwen3.5 sometimes emits ``</think>`` without a preceding ``<think>``
    opener (e.g. when the opener was chopped off during generation rollover).
    The strip helper drops the preamble up to ``</think>`` and keeps the
    postamble. Without this, the cap would land inside the dangling
    preamble and the ``</think>`` closer would survive in the turn-2
    context, confusing the chat template."""
    preamble = "leftover reasoning fragment " * 100
    visible = "Answer: yes."
    answer = f"{preamble}</think>\n{visible}"

    truncated, was_truncated, thinking_stripped = _strip_then_cap(answer, cap=512)

    assert thinking_stripped is True
    assert was_truncated is False
    assert truncated == visible


def test_no_thinking_tags_passthrough():
    """Non-thinking models (e.g. EuroLLM, Apertus) produce answers without
    any ``<think>`` markers. Strip is a no-op; the cap behaves exactly as
    before the fix."""
    visible = "Paris is the capital of France. " * 50  # ~1.6K chars
    cap = 512

    truncated, was_truncated, thinking_stripped = _strip_then_cap(visible, cap)

    assert thinking_stripped is False
    assert was_truncated is True
    assert truncated == visible[:cap]


def test_unclosed_think_block_is_unfixable_by_stripping():
    """Pathological case: the model writes ``<think>`` and hits the
    generation limit before emitting ``</think>``. No ``</think>`` tag or
    vLLM closer appears anywhere in the output, so the strip helper
    returns the text unchanged and the cap still clips inside the
    reasoning span. Stripping cannot fix this; the escape hatch is a
    larger ``battle_thinking_token_budget``."""
    reasoning = "still reasoning " * 1000
    answer = f"<think>{reasoning}"

    truncated, was_truncated, thinking_stripped = _strip_then_cap(answer, cap=256)

    assert thinking_stripped is False
    assert was_truncated is True
    assert truncated.startswith("<think>")


def test_strip_disabled_reverts_to_pre_fix_behaviour():
    """With ``strip_thinking_in_turn_1_carryover=False`` (the pre-fix
    behaviour, kept as a reproduction knob), the cap clips inside the
    ``<think>`` block and the ``</think>`` closer is lost."""
    reasoning = "deep thinking " * 400
    visible = "Short answer."
    answer = f"<think>{reasoning}</think>\n{visible}"

    truncated, was_truncated, thinking_stripped = _strip_then_cap(
        answer, cap=1024, strip=False
    )

    assert thinking_stripped is False
    assert was_truncated is True
    assert truncated.startswith("<think>")
    assert "</think>" not in truncated


def test_default_flag_is_enabled_in_base_cli_args():
    """Guard the default value: the fix ships enabled so existing runs
    (including Phase A of the Gemma-4 benchmark) pick it up without a
    launcher change."""
    args = BaseCliArgs(judge_model="OpenRouter/google/gemma-4-31b-it")
    assert args.strip_thinking_in_turn_1_carryover is True


def _make_mt_bench_cli_args(**overrides) -> CliArgs:
    args = CliArgs(
        judge_model="OpenRouter/google/gemma-4-31b-it",
        dataset="mt-bench",
        model_A="VLLM/Qwen/Qwen3.5-9B",
        model_B="VLLM/Qwen/Qwen3.5-9B",
        n_instructions=3,
        truncate_all_input_chars=30000,
        max_out_tokens_models=49152,
        max_model_len=57344,
        battle_thinking_token_budget=32768,
    )
    return replace(args, **overrides) if overrides else args


def test_mt_bench_cache_key_changes_when_flag_flipped():
    """The flag participates in the MT-Bench generation cache key so that
    flipping it off to reproduce pre-fix behaviour does not silently reuse
    post-fix completions (and vice versa). Without this, a rerun with the
    same numeric knobs would reuse stale cache for multi-turn datasets."""
    args_on = _make_mt_bench_cli_args(strip_thinking_in_turn_1_carryover=True)
    args_off = _make_mt_bench_cli_args(strip_thinking_in_turn_1_carryover=False)

    key_on = _mt_bench_generation_cache_name(args_on, model_name="VLLM/Qwen/Qwen3.5-9B")
    key_off = _mt_bench_generation_cache_name(
        args_off, model_name="VLLM/Qwen/Qwen3.5-9B"
    )

    assert key_on != key_off
    assert key_on.startswith("mt-bench_VLLM/Qwen/Qwen3.5-9B_3_")
    assert key_off.startswith("mt-bench_VLLM/Qwen/Qwen3.5-9B_3_")
