"""Shared CLI configuration for judgearena entrypoints.

Houses the base dataclass fields and argparse definitions that are common
to both ``judgearena`` (generate_and_evaluate) and ``judgearena-elo``
(estimate_elo_ratings) CLI tools.

The CLI exposes per-role generation parameters (``_A``, ``_B``,
``_judge``) so that every knob that can affect the generated text -
temperature, top_p, top_k, seed, max_tokens, max_model_len, chat_template,
engine_kwargs - is recorded explicitly in the run.  Older flags like
``--max_out_tokens_models`` keep working as deprecated fan-out aliases.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field

ROLE_NAMES: tuple[str, ...] = ("A", "B", "judge")
"""The three roles a JudgeArena run distinguishes for generation settings.

``A`` and ``B`` are the two battle models; ``judge`` is the LLM judge.
"""


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling and inference configuration applied to a single role.

    Every field is optional (``None``/empty defaults mean "let the backend
    pick").  The cache layer hashes this dataclass, so any change to a
    field invalidates the cached completions for that role.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    max_tokens: int = 32768
    max_model_len: int | None = None
    chat_template: str | None = None
    engine_kwargs: dict = field(default_factory=dict)


@dataclass
class BaseCliArgs:
    """Fields shared by every judgearena CLI entrypoint."""

    judge_model: str

    n_instructions: int | None = None
    swap_mode: str = "fixed"
    ignore_cache: bool = False
    truncate_all_input_chars: int = 8192
    result_folder: str = "results"
    verbosity: int = 0
    log_file: str | None = None
    no_log_file: bool = False

    # Per-role generation configuration.  Built by ``resolve_generation_configs``
    # from the CLI flags or set programmatically by API users.
    gen_A: GenerationConfig = field(default_factory=GenerationConfig)
    gen_B: GenerationConfig = field(default_factory=GenerationConfig)
    gen_judge: GenerationConfig = field(default_factory=GenerationConfig)

    provide_explanation: bool = False

    def __post_init__(self):
        supported_modes = ["fixed", "both"]
        assert self.swap_mode in supported_modes, (
            f"Only {supported_modes} modes are supported but got {self.swap_mode}."
        )


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _add_per_role_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the 8 sampling flags x 3 roles (24 flags total)."""
    for role in ROLE_NAMES:
        suffix = role
        parser.add_argument(
            f"--temperature_{suffix}",
            type=float,
            default=None,
            help=f"Sampling temperature for the {role!r} role.",
        )
        parser.add_argument(
            f"--top_p_{suffix}",
            type=float,
            default=None,
            help=f"Nucleus-sampling top-p for the {role!r} role.",
        )
        parser.add_argument(
            f"--top_k_{suffix}",
            type=int,
            default=None,
            help=f"Top-k sampling for the {role!r} role.",
        )
        parser.add_argument(
            f"--seed_{suffix}",
            type=int,
            default=None,
            help=(
                f"Random seed forwarded to the {role!r} role's backend. "
                "Hosted providers honour this on a best-effort basis."
            ),
        )
        parser.add_argument(
            f"--max_out_tokens_{suffix}",
            type=int,
            default=None,
            help=(
                f"Generation token budget for the {role!r} role. "
                "For VLLM, keep this <= --max_model_len_* (if provided)."
            ),
        )
        parser.add_argument(
            f"--max_model_len_{suffix}",
            type=int,
            default=None,
            help=(
                f"Optional total context window for the {role!r} role's "
                "vLLM model (prompt + generation)."
            ),
        )
        parser.add_argument(
            f"--chat_template_{suffix}",
            type=str,
            default=None,
            help=(
                f"Jinja2 chat template string used for the {role!r} role "
                "instead of the model tokenizer's template."
            ),
        )
        parser.add_argument(
            f"--engine_kwargs_{suffix}",
            type=str,
            default=None,
            help=(
                f"JSON dict of engine-specific kwargs forwarded to the "
                f"{role!r} role's backend."
            ),
        )


def _add_deprecated_aliases(parser: argparse.ArgumentParser) -> None:
    """Register the legacy non-role-aware flags that fan out to A/B/(judge)."""
    parser.add_argument(
        "--max_out_tokens_models",
        type=int,
        default=None,
        help=(
            "[DEPRECATED] Use --max_out_tokens_A and --max_out_tokens_B. "
            "Sets both A and B when neither role-specific flag is provided."
        ),
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help=(
            "[DEPRECATED] Use --max_model_len_{A,B,judge}. Fans out to all "
            "three roles when no role-specific flag is provided."
        ),
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help=(
            "[DEPRECATED] Use --chat_template_{A,B,judge}. Fans out to all "
            "three roles when no role-specific flag is provided."
        ),
    )
    parser.add_argument(
        "--engine_kwargs",
        type=str,
        default=None,
        help=(
            "[DEPRECATED] Use --engine_kwargs_{A,B,judge}. Fans out to all "
            "three roles when no role-specific flag is provided."
        ),
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
        "--provide_explanation",
        action="store_true",
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

    _add_per_role_generation_arguments(parser)
    _add_deprecated_aliases(parser)

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


def parse_engine_kwargs(raw: str | None) -> dict:
    """Parse and validate a JSON string into an engine-kwargs dict.

    ``None`` and empty strings both resolve to ``{}``.
    """
    if raw is None or raw == "":
        return {}
    try:
        engine_kwargs = json.loads(raw)
        if not isinstance(engine_kwargs, dict):
            raise ValueError("engine_kwargs must be a JSON object")
    except Exception as e:
        raise SystemExit(f"Failed to parse engine_kwargs: {e}") from e
    return engine_kwargs


def resolve_verbosity(args: argparse.Namespace) -> int:
    """Derive a single verbosity int from ``-v`` / ``-q`` flags.

    Returns ``-1`` for quiet, ``0`` for default (INFO), ``1+`` for verbose.
    """
    if getattr(args, "quiet", False):
        return -1
    return getattr(args, "verbose", 0)


# ---------------------------------------------------------------------------
# Resolver: argparse Namespace -> per-role GenerationConfig dict
# ---------------------------------------------------------------------------


_LEGACY_DEFAULT_MAX_TOKENS = 32768


def _warn_deprecated_alias(flag: str, replacements: tuple[str, ...]) -> None:
    """Emit a single ``DeprecationWarning`` pointing users at the new flags."""
    warnings.warn(
        f"{flag} is deprecated; use {', '.join(replacements)} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def resolve_generation_configs(
    args: argparse.Namespace,
) -> dict[str, GenerationConfig]:
    """Build ``{A, B, judge}`` :class:`GenerationConfig` from a parsed Namespace.

    Per-role flags take precedence; deprecated aliases fan out as fallbacks
    and emit a ``DeprecationWarning``.
    """
    legacy_max_tokens_models = getattr(args, "max_out_tokens_models", None)
    legacy_max_model_len = getattr(args, "max_model_len", None)
    legacy_chat_template = getattr(args, "chat_template", None)
    legacy_engine_kwargs_raw = getattr(args, "engine_kwargs", None)
    legacy_engine_kwargs = (
        parse_engine_kwargs(legacy_engine_kwargs_raw)
        if legacy_engine_kwargs_raw is not None
        else None
    )

    if legacy_max_tokens_models is not None:
        _warn_deprecated_alias(
            "--max_out_tokens_models",
            ("--max_out_tokens_A", "--max_out_tokens_B"),
        )
    if legacy_max_model_len is not None:
        _warn_deprecated_alias(
            "--max_model_len",
            ("--max_model_len_A", "--max_model_len_B", "--max_model_len_judge"),
        )
    if legacy_chat_template is not None:
        _warn_deprecated_alias(
            "--chat_template",
            ("--chat_template_A", "--chat_template_B", "--chat_template_judge"),
        )
    if legacy_engine_kwargs_raw is not None:
        _warn_deprecated_alias(
            "--engine_kwargs",
            ("--engine_kwargs_A", "--engine_kwargs_B", "--engine_kwargs_judge"),
        )

    configs: dict[str, GenerationConfig] = {}
    for role in ROLE_NAMES:
        explicit_max_tokens = getattr(args, f"max_out_tokens_{role}", None)
        if explicit_max_tokens is not None:
            max_tokens = explicit_max_tokens
        elif role in ("A", "B") and legacy_max_tokens_models is not None:
            max_tokens = legacy_max_tokens_models
        else:
            max_tokens = _LEGACY_DEFAULT_MAX_TOKENS

        explicit_max_model_len = getattr(args, f"max_model_len_{role}", None)
        max_model_len = (
            explicit_max_model_len
            if explicit_max_model_len is not None
            else legacy_max_model_len
        )

        explicit_chat_template = getattr(args, f"chat_template_{role}", None)
        chat_template = (
            explicit_chat_template
            if explicit_chat_template is not None
            else legacy_chat_template
        )

        explicit_engine_kwargs_raw = getattr(args, f"engine_kwargs_{role}", None)
        if explicit_engine_kwargs_raw is not None:
            engine_kwargs = parse_engine_kwargs(explicit_engine_kwargs_raw)
        elif legacy_engine_kwargs is not None:
            engine_kwargs = dict(legacy_engine_kwargs)
        else:
            engine_kwargs = {}

        configs[role] = GenerationConfig(
            temperature=getattr(args, f"temperature_{role}", None),
            top_p=getattr(args, f"top_p_{role}", None),
            top_k=getattr(args, f"top_k_{role}", None),
            seed=getattr(args, f"seed_{role}", None),
            max_tokens=max_tokens,
            max_model_len=max_model_len,
            chat_template=chat_template,
            engine_kwargs=engine_kwargs,
        )
    return configs


def gen_config_to_invoke_kwargs(gen: GenerationConfig) -> dict:
    """Flatten a :class:`GenerationConfig` to the kwargs accepted by ``make_model``.

    Backends ignore unknown kwargs (vLLM-only fields are stripped inside
    :func:`judgearena.utils.make_model` for hosted providers).
    """
    kwargs: dict[str, object] = {"max_tokens": gen.max_tokens}
    if gen.temperature is not None:
        kwargs["temperature"] = gen.temperature
    if gen.top_p is not None:
        kwargs["top_p"] = gen.top_p
    if gen.top_k is not None:
        kwargs["top_k"] = gen.top_k
    if gen.seed is not None:
        kwargs["seed"] = gen.seed
    if gen.max_model_len is not None:
        kwargs["max_model_len"] = gen.max_model_len
    if gen.chat_template is not None:
        kwargs["chat_template"] = gen.chat_template
    kwargs.update(gen.engine_kwargs)
    return kwargs
