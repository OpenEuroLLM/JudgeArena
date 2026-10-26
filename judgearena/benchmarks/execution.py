"""Shared model setup for benchmark runners."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from judgearena.cache.inference import (
    CompletionInferenceCache,
    JudgementInferenceCache,
)
from judgearena.models import (
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


def build_judge(cfg: RunConfig, *, default_temperature: float | None = None):
    """Prepare the judge and cache, retaining explicit settings over defaults."""
    kwargs = cfg.judge.model_kwargs(
        base_engine_kwargs=cfg.judge_engine_kwargs(),
        fallback_chat_template=cfg.model.chat_template,
    )
    if default_temperature is not None:
        kwargs.setdefault("temperature", default_temperature)
    return prepare_model(
        model=cfg.judge.model,
        cache=build_judgement_cache(cfg),
        **kwargs,
    )


def build_completion_cache(cfg: RunConfig) -> CompletionInferenceCache | None:
    if cfg.run.store_root is None:
        return None
    return CompletionInferenceCache(Path(cfg.run.store_root), cfg.task)


def build_judgement_cache(cfg: RunConfig) -> JudgementInferenceCache | None:
    if cfg.run.store_root is None:
        return None
    return JudgementInferenceCache(Path(cfg.run.store_root), cfg.task)


def complete_missing[Rows: (pd.Series, pd.DataFrame)](
    rows: Rows,
    existing: pd.Series | None,
    generate_missing: Callable[[Rows], pd.Series],
) -> pd.Series:
    """Fill missing answers by row ID; empty strings are already-complete answers."""
    if existing is not None and not existing.index.equals(rows.index):
        existing = existing.reindex(rows.index)
    completions = pd.Series(
        None if existing is None else existing.to_numpy(),
        index=rows.index,
        name="completion",
        dtype=object,
        copy=True,
    )
    missing = completions.isna().to_numpy()
    if missing.any():
        pending = rows.iloc[missing]
        generated = generate_missing(pending)
        # Repeated cache IDs can represent distinct rows already in input order.
        if not generated.index.equals(pending.index):
            generated = generated.loc[pending.index]
        completions.iloc[missing] = generated.to_numpy()
    return completions


def load_or_generate_completions(
    cfg: RunConfig,
    model: str,
    instructions: pd.Series,
    preloaded: pd.Series | None,
    *,
    role: Literal["A", "B"],
    generate: Callable[..., pd.DataFrame],
    use_tqdm: bool,
) -> pd.Series:
    """Reuse supplied rows and generate missing ones in the same input order."""
    if instructions.isna().any():
        raise ValueError("Cannot generate completions for missing instruction text.")

    def generate_missing(pending: pd.Series) -> pd.Series:
        return generate(
            instructions=pending,
            model=model,
            truncate_input_chars=cfg.generation.truncate_all_input_chars,
            use_tqdm=use_tqdm,
            inference_cache=build_completion_cache(cfg),
            **build_generation_kwargs(cfg, model, role=role),
        ).set_index("instruction_index")["completion"]

    # Elo's dataset row IDs and generation cache IDs can differ.
    existing = None if preloaded is None else preloaded.set_axis(instructions.index)
    return complete_missing(instructions, existing, generate_missing)
