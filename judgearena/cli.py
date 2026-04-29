"""Unified CLI entrypoint for judgearena.

Dispatches to ``generate_and_evaluate.main`` or ``estimate_elo_ratings.main``
based on the value of ``--task``. Task names prefixed with ``elo-`` run the
ELO rating flow; anything else runs the generate-and-judge flow.
"""

from __future__ import annotations

import argparse
import warnings

from judgearena.cli_common import (
    add_common_arguments,
    resolve_generation_configs,
    resolve_verbosity,
)
from judgearena.estimate_elo_ratings import CliEloArgs
from judgearena.estimate_elo_ratings import main as main_elo
from judgearena.generate_and_evaluate import CliArgs
from judgearena.generate_and_evaluate import main as main_generate_and_evaluate
from judgearena.log import configure_logging, get_logger

logger = get_logger(__name__)

ELO_TASK_PREFIX = "elo-"

# Lowercase CLI task name -> canonical arena identifier used inside
# ``judgearena.arenas_utils.KNOWN_ARENAS`` and the ``benchmark`` column of
# saved battle dataframes.  The CLI stays lowercase (matching ``alpaca-eval``
# conventions) while internal identifiers keep their original casing.
ELO_TASK_TO_ARENA: dict[str, str] = {
    "elo-lmarena-100k": "LMArena-100k",
    "elo-lmarena-140k": "LMArena-140k",
    "elo-lmarena": "LMArena",
    "elo-comparia": "ComparIA",
}


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
        "--model",
        help="[DEPRECATED] Use `--model_A` instead.",
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


def _resolve_model_a(args: argparse.Namespace) -> str | None:
    """Collapse the deprecated --model flag into --model_A."""
    if args.model is not None and args.model_A is not None:
        raise SystemExit(
            "Specify exactly one of --model_A/--model; --model is a deprecated alias."
        )
    if args.model is not None:
        warnings.warn(
            "--model is deprecated; use --model_A instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return args.model
    return args.model_A


def _common_base_kwargs(args: argparse.Namespace) -> dict:
    """Build the kwargs shared by every CLI dataclass from a parsed Namespace.

    Centralised here so that ``CliArgs`` and ``CliEloArgs`` see the same
    per-role :class:`GenerationConfig` instances and the same shared
    fields without each call site repeating the field-by-field forwarding.
    """
    gen_configs = resolve_generation_configs(args)
    return {
        "judge_model": args.judge_model,
        "n_instructions": args.n_instructions,
        "provide_explanation": args.provide_explanation,
        "swap_mode": args.swap_mode,
        "ignore_cache": args.ignore_cache,
        "truncate_all_input_chars": args.truncate_all_input_chars,
        "result_folder": args.result_folder,
        "verbosity": resolve_verbosity(args),
        "log_file": args.log_file,
        "no_log_file": args.no_log_file,
        "gen_A": gen_configs["A"],
        "gen_B": gen_configs["B"],
        "gen_judge": gen_configs["judge"],
    }


def _build_elo_args(
    args: argparse.Namespace, arena: str, model_a: str | None
) -> CliEloArgs:
    if model_a is None:
        raise SystemExit(
            "--model_A is required for elo tasks (use `--task elo-<arena> --model_A <model>`)."
        )
    if args.model_B is not None:
        raise SystemExit(
            "--model_B is not yet supported for elo tasks; only --model_A is used."
        )
    return CliEloArgs(
        arena=arena,
        model=model_a,
        n_instructions_per_language=args.n_instructions_per_language,
        languages=args.languages,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        baseline_model=args.baseline_model,
        **_common_base_kwargs(args),
    )


def _build_generate_and_evaluate_args(
    args: argparse.Namespace, task: str, model_a: str | None
) -> CliArgs:
    if model_a is None or args.model_B is None:
        raise SystemExit(f"--model_A and --model_B are required for task {task!r}.")
    return CliArgs(
        task=task,
        model_A=model_a,
        model_B=args.model_B,
        use_tqdm=args.use_tqdm,
        **_common_base_kwargs(args),
    )


def cli(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(resolve_verbosity(args), log_file=args.log_file)
    task = _resolve_task(args)
    model_a = _resolve_model_a(args)
    if task.startswith(ELO_TASK_PREFIX):
        if task not in ELO_TASK_TO_ARENA:
            raise SystemExit(
                f"Unknown elo task {task!r}; expected one of {list(ELO_TASK_TO_ARENA)}."
            )
        elo_args = _build_elo_args(args, arena=ELO_TASK_TO_ARENA[task], model_a=model_a)
        logger.debug("Running with CLI args: %s", elo_args.__dict__)
        main_elo(elo_args)
    else:
        ge_args = _build_generate_and_evaluate_args(args, task=task, model_a=model_a)
        logger.debug("Running with CLI args: %s", ge_args.__dict__)
        main_generate_and_evaluate(ge_args)


if __name__ == "__main__":
    cli()
