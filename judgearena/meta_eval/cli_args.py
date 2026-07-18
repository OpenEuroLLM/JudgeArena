"""Runtime arguments for meta-evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from judgearena.config import RunConfig
from judgearena.models import build_default_judge_model_kwargs

PROMPT_MODES = (
    "standard",
    "arena-hard",
    "alpaca-eval",
    "alpaca-eval-pair-score",
)


@dataclass
class CliMetaEvalArgs:
    """Resolved arguments consumed by the meta-evaluation pipeline."""

    judge_model: str
    reference_arena: str = "LMArena-140k"
    prompt_mode: str = "standard"
    top_models: int = 20
    battles_per_model: int = 50
    batch_size: int = 50
    languages: list[str] | None = None
    n_bootstraps: int = 20
    seed: int = 0
    elo_gap_battles: list[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    elo_gap_seeds: int = 10
    exclude_human_ties: bool = True
    provide_explanation: bool = False
    swap_mode: str = "fixed"
    ignore_cache: bool = False
    truncate_judge_input_chars: int | None = None
    max_out_tokens_judge: int = 32768
    max_model_len: int | None = None
    chat_template: str | None = None
    result_folder: str = "results"
    engine_kwargs: dict[str, object] = field(default_factory=dict)
    no_log_file: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.swap_mode not in {"fixed", "both"}:
            raise ValueError(f"Unsupported swap_mode {self.swap_mode!r}.")
        if self.prompt_mode not in PROMPT_MODES:
            raise ValueError(
                f"Unsupported prompt_mode {self.prompt_mode!r}; "
                f"expected one of {PROMPT_MODES}."
            )


def meta_eval_args_from_config(cfg: RunConfig) -> CliMetaEvalArgs:
    """Map the shared hierarchical run config to meta-eval runtime arguments."""
    if cfg.meta_eval is None:
        raise ValueError("meta_eval config is required for the meta-eval task.")

    judge_kwargs = build_default_judge_model_kwargs(
        cfg.judge.model,
        {},
        judge_engine_kwargs_override=cfg.judge.model_kwargs(),
    )
    max_out_tokens_judge = int(judge_kwargs.pop("max_tokens"))
    max_model_len = judge_kwargs.pop("max_model_len", None)
    chat_template = judge_kwargs.pop("chat_template", None)

    return CliMetaEvalArgs(
        judge_model=cfg.judge.model,
        reference_arena=cfg.meta_eval.reference_arena,
        prompt_mode=cfg.meta_eval.prompt_mode,
        top_models=cfg.meta_eval.top_models,
        battles_per_model=cfg.meta_eval.battles_per_model,
        batch_size=cfg.meta_eval.batch_size,
        languages=cfg.meta_eval.languages,
        n_bootstraps=cfg.meta_eval.n_bootstraps,
        seed=cfg.run.seed,
        elo_gap_battles=list(cfg.meta_eval.elo_gap_battles),
        elo_gap_seeds=cfg.meta_eval.elo_gap_seeds,
        exclude_human_ties=not cfg.meta_eval.include_human_ties,
        provide_explanation=cfg.judge.provide_explanation,
        swap_mode=cfg.judge.swap_mode,
        ignore_cache=cfg.run.ignore_cache,
        truncate_judge_input_chars=cfg.generation.truncate_judge_input_chars,
        max_out_tokens_judge=max_out_tokens_judge,
        max_model_len=max_model_len,
        chat_template=chat_template,
        result_folder=cfg.run.result_folder,
        engine_kwargs=judge_kwargs,
        no_log_file=cfg.run.no_log_file,
    )
