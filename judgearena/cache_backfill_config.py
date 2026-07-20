"""Reconstruct model configurations from historical JudgeArena runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judgearena.config import CacheArgs, RunConfig, load_config
from judgearena.meta_eval.cli_args import CliMetaEvalArgs
from judgearena.model_adapters import PreparedModel
from judgearena.models import build_default_judge_model_kwargs, make_model
from judgearena.repro import METADATA_FILENAME


def _cache_relevant_config(cfg: RunConfig) -> dict[str, Any]:
    payload = cfg.model_dump(mode="json")
    payload.pop("cache", None)
    payload.pop("run", None)
    return payload


def _load_modern_run_config(run_dir: Path) -> RunConfig | None:
    candidates: list[tuple[str, RunConfig]] = []
    metadata_path = run_dir / METADATA_FILENAME
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        run = payload.get("run")
        if not isinstance(run, dict):
            raise ValueError(
                f"{METADATA_FILENAME} has no run config in {run_dir.name}."
            )
        candidates.append((METADATA_FILENAME, RunConfig(**run)))
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        candidates.append(("config.yaml", load_config(config_path)))
    if not candidates:
        return None

    expected = _cache_relevant_config(candidates[0][1])
    conflicting = [
        name
        for name, candidate in candidates[1:]
        if _cache_relevant_config(candidate) != expected
    ]
    if conflicting:
        sources = ", ".join([candidates[0][0], *conflicting])
        raise ValueError(f"Conflicting run configs in {run_dir.name}: {sources}")
    return candidates[0][1]


def _legacy_args_to_run_config(args: dict[str, Any]) -> RunConfig:
    judge_model = args.get("judge_model") or args.get("judge", {}).get("model")
    if not isinstance(judge_model, str):
        raise ValueError("Legacy args missing judge_model.")

    engine_kwargs = dict(args.get("engine_kwargs") or {})
    judge_engine_kwargs = dict(args.get("judge_engine_kwargs") or {})
    judge_engine_kwargs.update(engine_kwargs)
    truncate_all = args.get("truncate_all_input_chars", 8192)
    truncate_judge = args.get("truncate_judge_input_chars")
    if truncate_judge is None:
        truncate_judge = truncate_all

    return RunConfig(
        task=str(args["task"]),
        model={
            "name": args.get("model_A") or args.get("model", {}).get("name"),
            "baseline": args.get("model_B") or args.get("model", {}).get("baseline"),
            "max_out_tokens": args.get("max_out_tokens_models")
            or args.get("model", {}).get("max_out_tokens", 32768),
            "max_model_len": args.get("max_model_len"),
            "chat_template": args.get("chat_template"),
            "engine_kwargs": engine_kwargs,
        },
        judge={
            "model": judge_model,
            "max_out_tokens": args.get("max_out_tokens_judge")
            or args.get("judge", {}).get("max_out_tokens", 32768),
            "max_model_len": args.get("max_model_len_judge")
            or args.get("max_model_len"),
            "chat_template": args.get("chat_template_judge")
            or args.get("chat_template"),
            "engine_kwargs": judge_engine_kwargs,
            "provide_explanation": bool(args.get("provide_explanation", False)),
            "swap_mode": args.get("swap_mode", "fixed"),
            "prompt_preset": args.get("prompt_preset"),
            "system_prompt_file": args.get("judge_system_prompt_file"),
            "user_prompt_file": args.get("judge_user_prompt_file"),
            "strip_thinking_before_judging": bool(
                args.get("strip_thinking_before_judging", False)
            ),
        },
        generation={
            "n_instructions": args.get("n_instructions"),
            "truncate_all_input_chars": truncate_all,
            "truncate_judge_input_chars": truncate_judge,
        },
        run={
            "result_folder": str(args.get("result_folder", "results")),
            "seed": args.get("seed", 0),
        },
    )


def load_gae_run_config(run_dir: Path) -> RunConfig:
    modern_cfg = _load_modern_run_config(run_dir)
    if modern_cfg is not None:
        return modern_cfg
    args_paths = sorted(run_dir.glob("args-*.json"))
    if not args_paths:
        raise ValueError(f"No reconstructable config found under {run_dir.name}.")
    if len(args_paths) > 1:
        names = ", ".join(path.name for path in args_paths)
        raise ValueError(
            f"Ambiguous legacy args files under {run_dir.name}; "
            f"expected one args-*.json or config/metadata: {names}"
        )
    args = json.loads(args_paths[0].read_text(encoding="utf-8"))
    return _legacy_args_to_run_config(args)


def load_meta_args(run_dir: Path) -> CliMetaEvalArgs:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise ValueError(f"Meta-eval run missing args.json: {run_dir.name}")
    payload = json.loads(args_path.read_text(encoding="utf-8"))
    cache_payload = payload.pop("cache", {})
    payload.pop("ignore_cache", None)
    if isinstance(cache_payload, dict):
        cache_payload.pop("ignore_cache", None)
    payload["cache"] = CacheArgs(**cache_payload) if cache_payload else CacheArgs()
    return CliMetaEvalArgs(**payload)


def build_gae_judge_model(cfg: RunConfig) -> PreparedModel:
    return make_model(
        model=cfg.judge.model,
        **build_default_judge_model_kwargs(
            cfg.judge.model,
            cfg.model.engine_kwargs,
            judge_engine_kwargs_override=cfg.judge.model_kwargs(
                fallback_chat_template=cfg.model.chat_template,
            ),
        ),
    )


def build_mt_judge_model(cfg: RunConfig, *, delegated: bool) -> PreparedModel:
    judge_model_kwargs = cfg.judge.model_kwargs(
        base_engine_kwargs=cfg.model.engine_kwargs,
        fallback_chat_template=cfg.model.chat_template,
    )
    if delegated and cfg.judge.temperature is None:
        judge_model_kwargs.setdefault("temperature", 0.0)
    return make_model(model=cfg.judge.model, **judge_model_kwargs)


def build_meta_judge_model(args: CliMetaEvalArgs) -> PreparedModel:
    return make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
        **args.engine_kwargs,
    )
