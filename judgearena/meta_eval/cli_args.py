"""CLI argument dataclass for meta-evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from judgearena.cli_common import (
    DEFAULT_MAX_OUT_TOKENS_MODELS,
    BaseCliArgs,
    parse_engine_kwargs,
    resolve_verbosity,
)

PROMPT_MODES = (
    "standard",
    "arena-hard",
    "alpaca-eval",
    "alpaca-eval-pair-score",
)


@dataclass
class CliMetaEvalArgs(BaseCliArgs):
    """CLI arguments for judge meta-evaluation."""

    reference_arena: str = "LMArena-140k"
    prompt_mode: str = "standard"
    top_models: int = 20
    battles_per_model: int = 50
    batch_size: int = 50
    languages: list[str] | None = None
    n_bootstraps: int = 20
    seed: int = 0
    elo_gap_battles: list[int] | None = None
    elo_gap_seeds: int = 10
    exclude_human_ties: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"Unsupported prompt_mode {self.prompt_mode!r}; "
                f"expected one of {PROMPT_MODES}."
            )
        if self.elo_gap_battles is None:
            self.elo_gap_battles = [10, 20, 30, 40, 50]


def add_meta_eval_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference_arena",
        default="LMArena-140k",
        help="[meta-eval] Human-labeled reference arena to sample battles from.",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=list(PROMPT_MODES),
        default="standard",
        help="[meta-eval] Named judge prompt mode.",
    )
    parser.add_argument(
        "--top_models",
        type=int,
        default=20,
        help="[meta-eval] Number of top models by battle count to include.",
    )
    parser.add_argument(
        "--battles_per_model",
        type=int,
        default=50,
        help="[meta-eval] Battles sampled per top model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="[meta-eval] Annotation batch size.",
    )
    parser.add_argument(
        "--elo_gap_battles",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50],
        help="[meta-eval] Battle counts for ELO-gap analysis.",
    )
    parser.add_argument(
        "--elo_gap_seeds",
        type=int,
        default=10,
        help="[meta-eval] Random seeds for ELO-gap subsampling.",
    )
    parser.add_argument(
        "--include_human_ties",
        action="store_true",
        help="[meta-eval] Include human-labeled ties in agreement metrics.",
    )


def build_meta_eval_args(args: argparse.Namespace) -> CliMetaEvalArgs:
    if args.model_A is not None or args.model_B is not None:
        raise SystemExit(
            "--model_A/--model_B are not used for meta-eval; only --judge_model is required."
        )
    if args.n_instructions is not None:
        raise SystemExit(
            "--n_instructions is not used for meta-eval; use --top_models and "
            "--battles_per_model to control the sample."
        )
    if args.max_out_tokens_models != DEFAULT_MAX_OUT_TOKENS_MODELS:
        raise SystemExit(
            "--max_out_tokens_models is not used for meta-eval because no model "
            "completions are generated."
        )
    return CliMetaEvalArgs(
        reference_arena=args.reference_arena,
        prompt_mode=args.prompt_mode,
        top_models=args.top_models,
        battles_per_model=args.battles_per_model,
        batch_size=args.batch_size,
        languages=args.languages,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        elo_gap_battles=args.elo_gap_battles,
        elo_gap_seeds=args.elo_gap_seeds,
        exclude_human_ties=not args.include_human_ties,
        judge_model=args.judge_model,
        n_instructions=args.n_instructions,
        provide_explanation=args.provide_explanation,
        swap_mode=args.swap_mode,
        ignore_cache=args.ignore_cache,
        truncate_all_input_chars=args.truncate_all_input_chars,
        max_out_tokens_models=args.max_out_tokens_models,
        max_out_tokens_judge=args.max_out_tokens_judge,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
        result_folder=args.result_folder,
        engine_kwargs=parse_engine_kwargs(args.engine_kwargs),
        verbosity=resolve_verbosity(args),
        log_file=args.log_file,
        no_log_file=args.no_log_file,
    )
