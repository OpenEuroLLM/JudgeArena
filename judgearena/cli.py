"""Unified CLI entrypoint for judgearena.

Dispatches to ``generate_and_evaluate.main`` or ``estimate_elo_ratings.main``
based on the value of ``--task``. Task names prefixed with ``elo-`` run the
ELO rating flow; anything else runs the generate-and-judge flow.
"""

from __future__ import annotations

import argparse
import warnings

from judgearena.cli_common import (
    ELO_TASK_PREFIX,
    ELO_TASK_TO_ARENA,
    add_common_arguments,
    parse_engine_kwargs,
    resolve_verbosity,
)
from judgearena.config import RunConfig, load_config
from judgearena.estimate_elo_ratings import CliEloArgs
from judgearena.estimate_elo_ratings import main as main_elo
from judgearena.generate_and_evaluate import CliArgs, native_pairwise_baseline
from judgearena.generate_and_evaluate import main as main_generate_and_evaluate
from judgearena.log import configure_logging, get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="judgearena",
        description=(
            "Run a judge-based evaluation. Use `--task <name>` for generate+judge "
            "benchmarks (e.g. alpaca-eval, arena-hard-v2.0, mt-bench) or "
            "`--task elo-<arena>` for ELO rating (e.g. elo-lmarena-140k, elo-comparia)."
        ),
    )
    parser.add_argument(
        "--config_path",
        default=None,
        help="Path to a YAML run config. When set, all other run options come "
        "from the file.",
    )
    parser.add_argument(
        "--task",
        help=(
            "Task to run. Generate+judge tasks: `alpaca-eval`, `arena-hard-v0.1`, "
            "`arena-hard-v2.0`, `m-arena-hard`, `m-arena-hard-{lang}`, `m-arena-hard-EU`, "
            "`mt-bench`, `fluency-{lang}`. ELO tasks: `elo-lmarena-100k`, `elo-lmarena-140k`, "
            "`elo-lmarena`, `elo-comparia`."
        ),
    )
    parser.add_argument(
        "--dataset",
        help="[DEPRECATED] Use `--task` instead.",
    )
    parser.add_argument(
        "--arena",
        help="[DEPRECATED] Use `--task elo-<arena>` instead.",
    )
    parser.add_argument(
        "--model_A",
        help=(
            "Model under evaluation. For pairwise tasks, this is Model A (paired with "
            "--model_B). For elo tasks, this is the single model rated against arena opponents."
        ),
    )
    parser.add_argument(
        "--model_B",
        help="Model B for generate+judge tasks (not yet supported for elo tasks).",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="[generate+judge] Use tqdm (not compatible with vLLM).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="[elo] Language codes to evaluate, e.g. `en fr de`.",
    )
    parser.add_argument(
        "--n_instructions_per_language",
        type=int,
        default=None,
        help="[elo] Cap battles per language.",
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=20,
        help="[elo] Bootstrap samples for ELO confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="[elo] Random seed for reproducibility.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="[elo] Model anchored at 1000 ELO (ratings are reported relative to it).",
    )
    add_common_arguments(parser)
    return parser


def _resolve_task(args: argparse.Namespace) -> str:
    """Return the task value from --task, or a deprecated --dataset / --arena."""
    set_flags = [
        name
        for name, value in (
            ("--task", args.task),
            ("--dataset", args.dataset),
            ("--arena", args.arena),
        )
        if value is not None
    ]
    if len(set_flags) > 1:
        raise SystemExit(
            f"Specify exactly one of --task/--dataset/--arena, got {set_flags}."
        )
    if not set_flags:
        raise SystemExit("One of --task/--dataset/--arena is required.")

    if args.task is not None:
        return args.task
    if args.dataset is not None:
        warnings.warn(
            "--dataset is deprecated; use --task instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return args.dataset
    # --arena was historically case-sensitive (e.g. "LMArena-140k").  Lowercase it
    # here so the deprecated path lands on a valid ELO_TASK_TO_ARENA key without
    # asking users to relearn the arena names.
    lower_arena = args.arena.lower()
    warnings.warn(
        f"--arena is deprecated; use --task {ELO_TASK_PREFIX}{lower_arena} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return f"{ELO_TASK_PREFIX}{lower_arena}"


def _runconfig_from_args(args: argparse.Namespace, task: str) -> RunConfig:
    """Single argparse-Namespace -> RunConfig mapping (shared by both builders)."""
    elo_kwargs = None
    if task.startswith(ELO_TASK_PREFIX):
        elo_kwargs = {
            "baseline_model": args.baseline_model,
            "n_bootstraps": args.n_bootstraps,
            "languages": args.languages,
            "n_instructions_per_language": args.n_instructions_per_language,
        }
    return RunConfig(
        task=task,
        model={
            "path": args.model_A,
            "path_b": args.model_B,
            "max_out_tokens": args.max_out_tokens_models,
            "max_model_len": args.max_model_len,
            "chat_template": args.chat_template,
            "engine_kwargs": parse_engine_kwargs(args.engine_kwargs),
        },
        judge={
            "model": args.judge_model,
            "max_out_tokens": args.max_out_tokens_judge,
            "max_model_len": args.max_judge_model_len,
            "engine_kwargs": parse_engine_kwargs(args.judge_engine_kwargs),
            "provide_explanation": args.provide_explanation,
            "swap_mode": args.swap_mode,
        },
        generation={
            "n_instructions": args.n_instructions,
            "truncate_all_input_chars": args.truncate_all_input_chars,
            "truncate_judge_input_chars": args.truncate_judge_input_chars,
        },
        elo=elo_kwargs,
        run={
            "seed": args.seed,
            "result_folder": args.result_folder,
            "ignore_cache": args.ignore_cache,
            "use_tqdm": args.use_tqdm,
            "verbosity": resolve_verbosity(args),
            "log_file": args.log_file,
            "no_log_file": args.no_log_file,
        },
    )


def _build_elo_args(
    args: argparse.Namespace, task: str, model_a: str | None
) -> CliEloArgs:
    if model_a is None:
        raise SystemExit(
            "--model_A is required for elo tasks (use `--task elo-<arena> --model_A <model>`)."
        )
    if args.model_B is not None:
        raise SystemExit(
            "--model_B is not yet supported for elo tasks; only --model_A is used."
        )
    return _runconfig_from_args(args, task).to_flat_args()


def _build_generate_and_evaluate_args(
    args: argparse.Namespace, task: str, model_a: str | None
) -> CliArgs:
    if model_a is None or (
        args.model_B is None and native_pairwise_baseline(task) is None
    ):
        raise SystemExit(f"--model_A and --model_B are required for task {task!r}.")
    return _runconfig_from_args(args, task).to_flat_args()


def cli(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.config_path is not None:
        cfg = load_config(args.config_path)
        configure_logging(cfg.run.verbosity, log_file=cfg.run.log_file)
        flat = cfg.to_flat_args()
        logger.debug("Running with config args: %s", flat.__dict__)
        if isinstance(flat, CliEloArgs):
            main_elo(flat)
        else:
            main_generate_and_evaluate(flat)
        return

    if args.judge_model is None:
        parser.error("the following arguments are required: --judge/--judge_model")
    configure_logging(resolve_verbosity(args), log_file=args.log_file)
    task = _resolve_task(args)
    if task.startswith(ELO_TASK_PREFIX):
        if task not in ELO_TASK_TO_ARENA:
            raise SystemExit(
                f"Unknown elo task {task!r}; expected one of {list(ELO_TASK_TO_ARENA)}."
            )
        elo_args = _build_elo_args(args, task=task, model_a=args.model_A)
        logger.debug("Running with CLI args: %s", elo_args.__dict__)
        main_elo(elo_args)
    else:
        ge_args = _build_generate_and_evaluate_args(
            args, task=task, model_a=args.model_A
        )
        logger.debug("Running with CLI args: %s", ge_args.__dict__)
        main_generate_and_evaluate(ge_args)


if __name__ == "__main__":
    cli()
