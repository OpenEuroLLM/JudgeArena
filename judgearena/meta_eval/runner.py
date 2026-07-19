"""Main entrypoint for judge meta-evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from judgearena.log import attach_file_handler, get_logger, make_run_log_path
from judgearena.meta_eval.annotate import annotate_sample
from judgearena.meta_eval.cli_args import CliMetaEvalArgs
from judgearena.meta_eval.metrics import (
    compute_agreement_metrics,
    compute_elo_gap_summary,
    format_metric,
    summarize_language_splits,
)
from judgearena.meta_eval.prompts import resolve_prompt_mode
from judgearena.meta_eval.sampling import (
    MetaEvalSamplingError,
    load_reference_arena_battles,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.models import make_model
from judgearena.repro import _to_jsonable, write_run_metadata

logger = get_logger(__name__)


def _result_folder_name(args: CliMetaEvalArgs, started_at: datetime) -> str:
    name = (
        f"meta-eval-{args.reference_arena}-{args.prompt_mode}-"
        f"{args.judge_model}-{args.swap_mode}"
    )
    return name.replace("/", "_") + f"-{started_at.strftime('%Y%m%d_%H%M%S')}"


def _build_summary_csv(
    language_summary: dict[str, dict[str, str | int]],
) -> pd.DataFrame:
    rows = []
    for split, metrics in language_summary.items():
        rows.append({"split": split, **metrics})
    return pd.DataFrame(rows)


def _agreement_view(
    metrics: dict[str, float | int],
    *,
    exclude_human_ties: bool,
) -> dict[str, float | int | str]:
    suffix = "_nt" if exclude_human_ties else ""
    n_key = "n_nt" if exclude_human_ties else "n"
    accuracy = float(metrics[f"accuracy{suffix}"])
    accuracy_se = float(metrics[f"acc_se{suffix}"])
    kappa = float(metrics[f"kappa{suffix}"])
    kappa_se = float(metrics[f"kappa_se{suffix}"])
    return {
        "n": int(metrics[n_key]),
        "accuracy": accuracy,
        "accuracy_se": accuracy_se,
        "kappa": kappa,
        "kappa_se": kappa_se,
        "accuracy_formatted": format_metric(accuracy, accuracy_se, digits=3),
        "kappa_formatted": format_metric(kappa, kappa_se, digits=3),
    }


def _annotation_telemetry(
    df_ann: pd.DataFrame,
    *,
    swap_mode: str,
) -> dict[str, object]:
    costs = pd.to_numeric(df_ann["cost_usd"], errors="coerce").dropna()
    sources = df_ann["cost_source"].dropna()
    total_cost = float(costs.sum()) if not costs.empty else None
    cost_per_1k = float(costs.mean() * 1000) if not costs.empty else None
    if costs.empty:
        logger.warning(
            "OpenRouter reference pricing is unavailable; cost fields will be null."
        )

    return {
        "judge_passes_per_battle": 2 if swap_mode == "both" else 1,
        "judgement_count": len(df_ann),
        "estimated_input_tokens": int(
            pd.to_numeric(df_ann["estimated_input_tokens"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "estimated_output_tokens": int(
            pd.to_numeric(df_ann["estimated_output_tokens"], errors="coerce")
            .fillna(0)
            .sum()
        ),
        "token_count_source": "estimated_chars_div_4",
        "total_cost_usd": total_cost,
        "cost_per_1k_judgements_usd": cost_per_1k,
        "cost_source_counts": {
            str(source): int(count) for source, count in sources.value_counts().items()
        },
    }


def _compute_results(
    *,
    args: CliMetaEvalArgs,
    top_models: list[str],
    df_top: pd.DataFrame,
    df_sample: pd.DataFrame,
    df_ann: pd.DataFrame,
) -> dict:
    agreement_metrics = compute_agreement_metrics(
        df_ann["winner"].tolist(),
        df_ann["winner_llm"].tolist(),
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )
    primary_view = "no_human_ties" if args.exclude_human_ties else "all"
    agreement = {
        "primary_view": primary_view,
        "all": _agreement_view(
            agreement_metrics,
            exclude_human_ties=False,
        ),
        "no_human_ties": _agreement_view(
            agreement_metrics,
            exclude_human_ties=True,
        ),
    }
    ranking_annotations = df_ann[df_ann["orientation"] == "forward"].copy()
    language_summary = summarize_language_splits(
        ranking_annotations,
        exclude_human_ties=args.exclude_human_ties,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )
    elo_gap_all = compute_elo_gap_summary(
        df_top,
        ranking_annotations,
        top_models,
        n_battles_list=args.elo_gap_battles or [],
        n_seeds=args.elo_gap_seeds,
        seed=args.seed,
        exclude_ties=False,
    )
    elo_gap_no_tie = compute_elo_gap_summary(
        df_top,
        ranking_annotations,
        top_models,
        n_battles_list=args.elo_gap_battles or [],
        n_seeds=args.elo_gap_seeds,
        seed=args.seed + 1000,
        exclude_ties=True,
    )
    return {
        "task": "meta-eval",
        "reference_arena": args.reference_arena,
        "prompt_mode": args.prompt_mode,
        "judge_model": args.judge_model,
        "top_models": top_models,
        "sample_size": len(df_sample),
        "ranking_annotation_count": len(ranking_annotations),
        "agreement": agreement,
        "language_summary": language_summary,
        "elo_gap_all": elo_gap_all.to_dict(orient="records"),
        "elo_gap_exclude_ties": elo_gap_no_tie.to_dict(orient="records"),
        **_annotation_telemetry(df_ann, swap_mode=args.swap_mode),
    }


def main(args: CliMetaEvalArgs) -> dict:
    started_at = datetime.now(UTC)
    res_folder = Path(args.result_folder) / _result_folder_name(args, started_at)
    res_folder.mkdir(parents=True, exist_ok=True)

    if not args.no_log_file:
        attach_file_handler(make_run_log_path(res_folder))

    with open(res_folder / "args.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(args), handle, indent=2)

    prompt_spec = resolve_prompt_mode(
        args.prompt_mode,
        provide_explanation=args.provide_explanation,
    )
    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
        **args.engine_kwargs,
    )

    logger.info("Loading reference arena %s", args.reference_arena)
    df = load_reference_arena_battles(
        args.reference_arena,
        languages=args.languages,
    )
    top_models, df_top = select_top_models(df, top_models=args.top_models)
    df_sample = sample_battles_per_model(
        df_top,
        top_models,
        battles_per_model=args.battles_per_model,
        seed=args.seed,
    )

    logger.info(
        "Meta-eval sample: %d battles among top %d models",
        len(df_sample),
        len(top_models),
    )

    df_ann = annotate_sample(
        df_sample,
        args,
        judge_chat_model=judge_chat_model,
        prompt_spec=prompt_spec,
    )
    df_ann.to_parquet(res_folder / "annotations.parquet", index=False)

    results = _compute_results(
        args=args,
        top_models=top_models,
        df_top=df_top,
        df_sample=df_sample,
        df_ann=df_ann,
    )

    with open(res_folder / "results.json", "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(results), handle, indent=2, allow_nan=False)

    summary_csv = _build_summary_csv(results["language_summary"])
    summary_csv.to_csv(res_folder / "summary.csv", index=False)

    write_run_metadata(
        output_dir=res_folder,
        entrypoint="judgearena.meta_eval.runner",
        run=asdict(args),
        results=results,
        input_payloads={"question_id": df_sample["question_id"].astype(str).tolist()},
        judge_system_prompt=prompt_spec.system_prompt,
        judge_user_prompt_template=prompt_spec.user_prompt_template,
        started_at_utc=started_at,
    )

    logger.info("Meta-eval results saved to %s", res_folder)
    return results


def run_or_exit(args: CliMetaEvalArgs) -> dict:
    try:
        return main(args)
    except MetaEvalSamplingError as exc:
        raise SystemExit(str(exc)) from exc
