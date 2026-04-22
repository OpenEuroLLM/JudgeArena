"""Shared CLI configuration for judgearena entrypoints.

Houses the base dataclass fields and argparse definitions that are common
to both ``judgearena`` (generate_and_evaluate) and ``judgearena-elo``
(estimate_elo_ratings) CLI tools.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from judgearena.judge_prompt_presets import JUDGE_PROMPT_PRESETS


@dataclass
class BaseCliArgs:
    """Fields shared by every judgearena CLI entrypoint."""

    judge_model: str

    n_instructions: int | None = None
    provide_explanation: bool = False
    swap_mode: str = "fixed"
    ignore_cache: bool = False
    judge_prompt_preset: str = "default"
    battle_thinking_token_budget: int | None = None
    strip_thinking_before_judging: bool = False
    truncate_all_input_chars: int = 8192
    truncate_judge_input_chars: int | None = None
    max_out_tokens_models: int = 32768
    max_out_tokens_judge: int = 32768
    max_model_len: int | None = None
    max_judge_model_len: int | None = None
    chat_template: str | None = None
    result_folder: str = "results"
    engine_kwargs: dict = field(default_factory=dict)
    judge_engine_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        supported_modes = ["fixed", "both"]
        assert self.swap_mode in supported_modes, (
            f"Only {supported_modes} modes are supported but got {self.swap_mode}."
        )

    def effective_judge_truncation(self) -> int:
        """Character cap applied to judge-side inputs (completions, reference, etc.).

        Falls back to the generation-side ``truncate_all_input_chars`` when a
        dedicated judge cap is not configured.
        """
        if self.truncate_judge_input_chars is not None:
            return int(self.truncate_judge_input_chars)
        return int(self.truncate_all_input_chars)

    def effective_judge_max_model_len(self) -> int | None:
        """Total context window for the judge vLLM instance.

        Falls back to the generation-side ``max_model_len`` when a dedicated
        judge context window is not configured.
        """
        if self.max_judge_model_len is not None:
            return int(self.max_judge_model_len)
        return self.max_model_len


def parse_optional_bool(raw: str | None) -> bool:
    if raw is None:
        return True
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{raw}'.")


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
        "--provide_explanation",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
        help=(
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
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
        help="If specified, ignore cache of previous completions.",
    )
    parser.add_argument(
        "--judge_prompt_preset",
        type=str,
        choices=JUDGE_PROMPT_PRESETS,
        default="default",
        help=(
            "Judge prompt preset to use. 'default' preserves the existing score-first "
            "JudgeArena prompts, while 'skywork' enables an optional Skywork-style "
            "verdict-first preset."
        ),
    )
    parser.add_argument(
        "--battle_thinking_token_budget",
        type=int,
        required=False,
        default=None,
        help=(
            "Optional reasoning-token sub-budget for battle-model generation. "
            "This stays inside --max_out_tokens_models."
        ),
    )
    parser.add_argument(
        "--strip_thinking_before_judging",
        nargs="?",
        const=True,
        default=False,
        type=parse_optional_bool,
        help=(
            "If specified, strip visible reasoning traces from model completions "
            "before sending them to the judge."
        ),
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
            "Character-level truncation applied to generation-side inputs: "
            "truncates each instruction before model A/B generation. When "
            "--truncate_judge_input_chars is not set, this value also caps the "
            "judge-side inputs (completions, reference, etc.)."
        ),
    )
    parser.add_argument(
        "--truncate_judge_input_chars",
        type=int,
        required=False,
        default=None,
        help=(
            "Character cap applied to judge-side inputs (completions, "
            "reference, instruction) before judge evaluation. Falls back to "
            "--truncate_all_input_chars when not specified. Set much higher "
            "than the generation cap to avoid cutting model completions before "
            "they reach the judge."
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
            "Optional total context window for the battle-generation VLLM "
            "instances (prompt + generation). Independent from "
            "--max_out_tokens_models/--max_out_tokens_judge, which only cap "
            "generated tokens. When --max_judge_model_len is not set, this "
            "value also sizes the judge instance."
        ),
    )
    parser.add_argument(
        "--max_judge_model_len",
        type=int,
        required=False,
        default=None,
        help=(
            "Optional total context window for the judge VLLM instance. Falls "
            "back to --max_model_len when not specified. Set higher than the "
            "battle model_len when the judge needs to see longer prompts "
            "(e.g. long completions from both A and B) than the battle "
            "generator can fit."
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
        "--judge_engine_kwargs",
        type=str,
        required=False,
        default="{}",
        help=(
            "Optional JSON dict of engine-specific kwargs that override "
            "``--engine_kwargs`` only for the judge model. Useful when the "
            "judge needs a different tensor-parallel or quantization config "
            "than the battle models, e.g. a 70B judge on TP=2 while the "
            "battle models run on TP=1 to dodge compile-time deadlocks."
        ),
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
