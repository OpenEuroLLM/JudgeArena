"""Shared model setup for benchmark runners."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from judgearena.inference import CompletionInferenceCache, JudgementInferenceCache
from judgearena.models import (
    build_default_judge_model_kwargs,
    is_thinking_model,
    prepare_model,
)

if TYPE_CHECKING:
    from judgearena.config import RunConfig


def build_generation_kwargs(
    cfg: RunConfig, model_spec: str, *, role: Literal["A", "B"]
) -> dict[str, object]:
    """Resolve generation kwargs for an evaluated or baseline model."""
    if role == "A":
        generation_kwargs = cfg.model.evaluated_generation_kwargs()
    elif role == "B":
        generation_kwargs = cfg.model.baseline_generation_kwargs()
    else:  # guarded by the type; keeps untyped callers honest
        raise ValueError(f"Unknown generation role: {role!r}")

    provider, _, model_name = model_spec.partition("/")
    budget = cfg.judge.battle_thinking_token_budget
    if budget is not None and provider == "VLLM" and is_thinking_model(model_name):
        max_tokens = int(generation_kwargs.get("max_tokens", cfg.model.max_out_tokens))
        generation_kwargs["thinking_token_budget"] = min(int(budget), max_tokens)
    return generation_kwargs


def build_judge(cfg: RunConfig):
    """Construct the configured judge consistently across benchmark runners."""
    return prepare_model(
        model=cfg.judge.model,
        cache=build_judgement_cache(cfg),
        **build_default_judge_model_kwargs(
            cfg.judge.model,
            cfg.model.engine_kwargs,
            judge_engine_kwargs_override=cfg.judge.model_kwargs(
                fallback_chat_template=cfg.model.chat_template,
            ),
        ),
    )


def build_completion_cache(cfg: RunConfig) -> CompletionInferenceCache | None:
    if cfg.run.store_root is None:
        return None
    return CompletionInferenceCache(Path(cfg.run.store_root), cfg.task)


def build_judgement_cache(cfg: RunConfig) -> JudgementInferenceCache | None:
    if cfg.run.store_root is None:
        return None
    return JudgementInferenceCache(Path(cfg.run.store_root), cfg.task)
