"""Shared CLI configuration for judgearena entrypoints.

Houses the base dataclass fields and argparse definitions that are common
to both ``judgearena`` (generate_and_evaluate) and ``judgearena-elo``
(estimate_elo_ratings) CLI tools.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field


@dataclass
class BaseCliArgs:
    """Fields shared by every judgearena CLI entrypoint."""

    judge_model: str

    n_instructions: int | None = None
    # Judge-prompt selection (see ``judgearena.prompts.registry``).
    # ``judge_prompt_preset`` picks a named preset; the ``_file`` overrides
    # take a path on disk and win over the preset.  ``provide_explanation``
    # is kept for backward compatibility and is equivalent to setting the
    # preset to ``default_with_explanation``.
    judge_prompt_preset: str | None = None
    judge_system_prompt_file: str | None = None
    judge_user_prompt_file: str | None = None
    provide_explanation: bool = False
    swap_mode: str = "fixed"
    ignore_cache: bool = False
    truncate_all_input_chars: int = 8192
    max_out_tokens_models: int = 32768
    max_out_tokens_judge: int = 32768
    max_model_len: int | None = None
    chat_template: str | None = None
    result_folder: str = "results"
    engine_kwargs: dict = field(default_factory=dict)
    verbosity: int = 0
    log_file: str | None = None
    no_log_file: bool = False

    def __post_init__(self):
        supported_modes = ["fixed", "both"]
        assert self.swap_mode in supported_modes, (
            f"Only {supported_modes} modes are supported but got {self.swap_mode}."
        )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the CLI flags shared by all judgearena entrypoints."""
    parser.add_argument(
        "--judge",
        "--judge_model",
        dest="judge_model",
        required=True,
        help=(
            "Name of the LLM to use as judge, for instance "
            "`Together/meta-llama/Meta-Llama-3-70B-Instruct-Turbo`, "
            "`VLLM/meta-llama/Meta-Llama-3-70B-Instruct-Turbo`, "
            "`LlamaCpp/path/to/model.gguf` etc"
        ),
    )
    parser.add_argument(
        "--n_instructions",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--judge_prompt_preset",
        type=str,
        required=False,
        default=None,
        help=(
            "Name of a judge-prompt preset registered in "
            "``judgearena.prompts.registry`` (e.g. ``default``, "
            "``default_with_explanation``, ``fluency``, "
            "``fastchat-pairwise``). When omitted, the per-task default "
            "is used."
        ),
    )
    parser.add_argument(
        "--judge_system_prompt_file",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a custom judge system prompt; takes precedence over "
            "--judge_prompt_preset.  Must be combined with "
            "--judge_user_prompt_file."
        ),
    )
    parser.add_argument(
        "--judge_user_prompt_file",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to a custom judge user-prompt template; takes precedence "
            "over --judge_prompt_preset.  Must be combined with "
            "--judge_system_prompt_file."
        ),
    )
    parser.add_argument(
        "--provide_explanation",
        action="store_true",
        help=(
            "Equivalent to --judge_prompt_preset default_with_explanation. "
            "If specified, judge will provide explanation before making a "
            "judgement. Does not necessarily improve the accuracy of the judge "
            "but enables some result interpretation."
        ),
    )
    parser.add_argument(
        "--swap_mode",
        type=str,
        choices=["fixed", "both"],
        default="fixed",
        help=(
            "Model comparison order mode. 'fixed': always use model order A-B. "
            "'both': correct for model order bias by evaluating each instruction "
            "twice, once as A-B and once as B-A, and concatenating the results. "
            "This helps account for judge position bias. Default is 'fixed'."
        ),
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="If specified, ignore cache of previous completions.",
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        required=False,
        default="results",
        help=(
            "The folder to save the results. Defaults to `results`. Evaluation "
            "results will be saved in `[result_folder]/[evaluation_name]`."
        ),
    )
    parser.add_argument(
        "--truncate_all_input_chars",
        type=int,
        required=False,
        default=8192,
        help=(
            "Character-level truncation applied before tokenization: truncates "
            "each instruction before model A/B generation and truncates each "
            "completion before judge evaluation."
        ),
    )
    parser.add_argument(
        "--max_out_tokens_models",
        type=int,
        required=False,
        default=32768,
        help=(
            "Generation token budget for each model A/B response. For VLLM, "
            "keep this <= --max_model_len (if provided)."
        ),
    )
    parser.add_argument(
        "--max_out_tokens_judge",
        type=int,
        required=False,
        default=32768,
        help=(
            "Generation token budget for the judge response (reasoning + scores). "
            "For VLLM, keep this <= --max_model_len (if provided)."
        ),
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        required=False,
        default=None,
        help=(
            "Optional total context window for VLLM models (prompt + generation). "
            "This is independent from --max_out_tokens_models/--max_out_tokens_judge, "
            "which only cap generated tokens. This is useful on smaller GPUs to "
            "avoid OOM."
        ),
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        required=False,
        default=None,
        help=(
            "Jinja2 chat template string to use instead of the model's tokenizer "
            "template. If not provided, ChatML is used as fallback for models "
            "without a chat template."
        ),
    )
    parser.add_argument(
        "--engine_kwargs",
        type=str,
        required=False,
        default="{}",
        help=(
            "JSON dict of engine-specific kwargs forwarded to the underlying "
            "engine. Example for vLLM: "
            '\'{"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9}\'.'
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity. Use -v for DEBUG output.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress all output except warnings and errors.",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        type=str,
        default=None,
        help=(
            "Write the full DEBUG log to this file in addition to the "
            "console output. By default a timestamped run-*.log is saved "
            "automatically in the result folder."
        ),
    )
    parser.add_argument(
        "--no-log-file",
        dest="no_log_file",
        action="store_true",
        default=False,
        help="Disable automatic file logging in the result folder.",
    )


def parse_engine_kwargs(raw: str) -> dict:
    """Parse and validate a JSON string into an engine-kwargs dict."""
    try:
        engine_kwargs = json.loads(raw) if raw else {}
        if not isinstance(engine_kwargs, dict):
            raise ValueError("engine_kwargs must be a JSON object")
    except Exception as e:
        raise SystemExit(f"Failed to parse --engine_kwargs: {e}") from e
    return engine_kwargs


def resolve_verbosity(args: argparse.Namespace) -> int:
    """Derive a single verbosity int from ``-v`` / ``-q`` flags.

    Returns ``-1`` for quiet, ``0`` for default (INFO), ``1+`` for verbose.
    """
    if getattr(args, "quiet", False):
        return -1
    return getattr(args, "verbose", 0)
