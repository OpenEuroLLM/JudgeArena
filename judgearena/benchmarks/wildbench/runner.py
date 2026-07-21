"""Registered WildBench V2 generation and evaluation runner."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from judgearena.artifacts import (
    prepare_run_directory,
    slugify,
    write_run_metadata_safely,
)
from judgearena.benchmarks.execution import build_generation_kwargs, build_judge
from judgearena.benchmarks.wildbench.prompting import (
    WildBenchPrompt,
    render_wildbench_pairwise_prompt,
    render_wildbench_score_prompt,
    resolve_wildbench_prompt,
)
from judgearena.benchmarks.wildbench.report import WildBenchReport
from judgearena.benchmarks.wildbench.scoring import (
    WildBenchMetrics,
    WildBenchScorer,
    apply_wildbench_length_penalty,
    choice_to_candidate_reward,
    resolve_wildbench_scorer,
)
from judgearena.datasets.registry import resolve_dataset_adapter
from judgearena.log import get_logger
from judgearena.models import do_inference, make_model
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import (
    HuggingFaceDatasetSource,
    OfficialOutputsBaseline,
    ResolvedTaskSpec,
    WildBenchProtocol,
)
from judgearena.utils import (
    cache_function_dataframe,
    data_root,
    generation_cache_token,
    strip_thinking_tags,
    truncate,
)

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def _conversation_prompt(
    messages: list[dict[str, str]], max_chars: int | None
) -> ChatPromptValue:
    prompt_messages = [SystemMessage(content="You are a helpful assistant.")]
    for message in messages:
        content = truncate(message["content"], max_len=max_chars)
        cls = HumanMessage if message["role"] == "user" else AIMessage
        prompt_messages.append(cls(content=content))
    return ChatPromptValue(messages=prompt_messages)


def generate_wildbench_outputs(
    examples: pd.DataFrame,
    model: str,
    *,
    truncate_input_chars: int | None,
    use_tqdm: bool,
    generation_kwargs: dict[str, object],
) -> pd.DataFrame:
    """Continue each normalized WildBench conversation with one model."""
    chat_model = make_model(model=model, **generation_kwargs)
    inputs = [
        _conversation_prompt(messages, truncate_input_chars)
        for messages in examples["conversation_input"]
    ]
    completions = do_inference(chat_model, inputs, use_tqdm=use_tqdm)
    return pd.DataFrame(
        {
            "instruction_index": examples.index.astype(str),
            "completion": completions,
        }
    )


def _align_outputs(
    outputs: pd.DataFrame, examples: pd.DataFrame, model_name: str
) -> pd.Series:
    indexed = outputs.copy()
    indexed["instruction_index"] = indexed["instruction_index"].astype(str)
    indexed = indexed.drop_duplicates("instruction_index").set_index(
        "instruction_index"
    )
    expected = pd.Index(examples.index.astype(str))
    missing = expected.difference(indexed.index)
    if not missing.empty:
        raise ValueError(
            f"WildBench outputs for {model_name!r} are missing {len(missing)} "
            f"session(s); first missing id: {missing[0]}."
        )
    output_column = "completion" if "completion" in indexed.columns else "output"
    return indexed.loc[expected, output_column].fillna("").astype(str)


def _official_outputs_for_model(
    model_outputs: pd.DataFrame | None,
    model_name: str,
) -> pd.DataFrame | None:
    if model_outputs is None:
        return None
    outputs = model_outputs.loc[
        model_outputs["model"] == model_name,
        ["instruction_index", "output"],
    ]
    return outputs if not outputs.empty else None


def _load_or_generate_outputs(
    cfg: RunConfig,
    examples: pd.DataFrame,
    model_name: str,
    *,
    role: Literal["A", "B"],
    official_outputs: pd.DataFrame | None = None,
) -> pd.Series:
    if role == "B":
        preloaded = _official_outputs_for_model(official_outputs, model_name)
        if preloaded is not None:
            return _align_outputs(preloaded, examples, model_name)

    generation_kwargs = build_generation_kwargs(cfg, model_name, role=role)
    sampling_token = generation_cache_token(generation_kwargs)
    session_token = hashlib.sha256(
        "\n".join(examples.index.astype(str)).encode("utf-8")
    ).hexdigest()[:12]
    generated = cache_function_dataframe(
        lambda: generate_wildbench_outputs(
            examples,
            model_name,
            truncate_input_chars=cfg.generation.truncate_all_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            generation_kwargs=generation_kwargs,
        ),
        ignore_cache=cfg.run.ignore_cache,
        cache_name=(
            f"wildbench-v2/{model_name}/{role.lower()}_{session_token}_{sampling_token}"
        ),
    )
    return _align_outputs(generated, examples, model_name)


def _run_judge_prompts(
    judge_model: object, prompts: list[str], *, use_tqdm: bool
) -> list[str]:
    if not prompts:
        return []
    return do_inference(judge_model, prompts, use_tqdm=use_tqdm)


def _score_annotations(
    cfg: RunConfig,
    protocol: WildBenchProtocol,
    prompt: WildBenchPrompt,
    scorer: WildBenchScorer,
    examples: pd.DataFrame,
    candidate_outputs: pd.Series,
) -> tuple[pd.DataFrame, int]:
    prompts = []
    pending_ids = []
    records = []
    for session_id, example in examples.iterrows():
        output = candidate_outputs.loc[str(session_id)]
        rendered = render_wildbench_score_prompt(
            prompt,
            example,
            output,
            max_words=cfg.wildbench.max_words_to_eval,
            max_chars=cfg.generation.truncate_judge_input_chars,
        )
        if output.strip():
            prompts.append(rendered)
            pending_ids.append(str(session_id))
        else:
            records.append(
                {
                    "session_id": str(session_id),
                    "prompt": rendered,
                    "judge_completion": json.dumps(
                        {
                            "strengths": "N/A",
                            "weaknesses": "The model output is empty.",
                            "score": "1",
                        }
                    ),
                    "score": 1.0,
                }
            )

    judge_outputs = _run_judge_prompts(
        build_judge(cfg), prompts, use_tqdm=cfg.run.use_tqdm
    )
    for session_id, rendered, judge_output in zip(
        pending_ids, prompts, judge_outputs, strict=True
    ):
        parsed = scorer.parse(judge_output)
        records.append(
            {
                "session_id": session_id,
                "prompt": rendered,
                "judge_completion": judge_output,
                "score": parsed if isinstance(parsed, float) else None,
            }
        )
    return pd.DataFrame(records).sort_values("session_id"), len(prompts)


def _reward_annotations(
    cfg: RunConfig,
    protocol: WildBenchProtocol,
    prompt: WildBenchPrompt,
    scorer: WildBenchScorer,
    examples: pd.DataFrame,
    candidate_outputs: pd.Series,
    baseline_outputs: dict[str, pd.Series],
    *,
    length_penalty_chars: int | None,
) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(cfg.run.seed)
    prompt_records = []
    pending_prompts = []
    pending_indices = []

    for baseline_model, outputs in baseline_outputs.items():
        for session_id, example in examples.iterrows():
            sid = str(session_id)
            candidate = candidate_outputs.loc[sid]
            baseline = outputs.loc[sid]
            if cfg.judge.strip_thinking_before_judging:
                candidate = strip_thinking_tags(candidate)
                baseline = strip_thinking_tags(baseline)

            positions = (
                [True, False]
                if cfg.judge.swap_mode == "both"
                else [bool(rng.integers(0, 2))]
            )
            for pass_index, candidate_is_a in enumerate(positions):
                completion_a = candidate if candidate_is_a else baseline
                completion_b = baseline if candidate_is_a else candidate
                rendered = render_wildbench_pairwise_prompt(
                    prompt,
                    example,
                    completion_a,
                    completion_b,
                    max_words=cfg.wildbench.max_words_to_eval,
                    max_chars=cfg.generation.truncate_judge_input_chars,
                )
                record = {
                    "session_id": sid,
                    "baseline_model": baseline_model,
                    "pass_index": pass_index,
                    "candidate_is_a": candidate_is_a,
                    "candidate_output": candidate,
                    "baseline_output": baseline,
                    "prompt": rendered,
                }
                if not candidate.strip() and not baseline.strip():
                    choice = "A=B"
                elif not candidate.strip():
                    choice = "B++" if candidate_is_a else "A++"
                elif not baseline.strip():
                    choice = "A++" if candidate_is_a else "B++"
                else:
                    choice = None

                if choice is None:
                    pending_indices.append(len(prompt_records))
                    pending_prompts.append(rendered)
                    record["judge_completion"] = None
                    record["choice"] = None
                else:
                    record["judge_completion"] = json.dumps({"choice": choice})
                    record["choice"] = choice
                prompt_records.append(record)

    judge_outputs = _run_judge_prompts(
        build_judge(cfg), pending_prompts, use_tqdm=cfg.run.use_tqdm
    )
    for record_index, judge_output in zip(pending_indices, judge_outputs, strict=True):
        parsed = scorer.parse(judge_output)
        prompt_records[record_index]["judge_completion"] = judge_output
        prompt_records[record_index]["choice"] = (
            parsed if isinstance(parsed, str) else None
        )

    for record in prompt_records:
        choice = record["choice"]
        if not isinstance(choice, str):
            record["raw_reward"] = np.nan
            record["reward"] = np.nan
            continue
        raw_reward = choice_to_candidate_reward(
            choice, candidate_is_a=bool(record["candidate_is_a"])
        )
        record["raw_reward"] = raw_reward
        record["reward"] = apply_wildbench_length_penalty(
            raw_reward,
            str(record["candidate_output"]),
            str(record["baseline_output"]),
            length_penalty_chars,
        )
    return pd.DataFrame(prompt_records), len(pending_prompts)


def _source_repo_id(task: ResolvedTaskSpec, name: str) -> str:
    source = task.spec.dataset.sources[name]
    if not isinstance(source, HuggingFaceDatasetSource):
        raise ValueError(f"WildBench source {name!r} is not a Hugging Face dataset.")
    return source.repo_id


def _build_report(
    *,
    cfg: RunConfig,
    task: ResolvedTaskSpec,
    protocol: WildBenchProtocol,
    prompt: WildBenchPrompt,
    metrics: WildBenchMetrics,
    baseline_models: list[str],
    annotations: pd.DataFrame,
    num_judgments: int,
    num_examples: int,
    length_penalty_chars: int | None,
) -> WildBenchReport:
    metadata: dict[str, object] = {
        "dataset": _source_repo_id(task, "examples"),
        "prompt_sha256": hashlib.sha256(prompt.template.encode("utf-8")).hexdigest(),
        "paper": task.spec.metadata.paper or "https://arxiv.org/abs/2406.04770",
    }
    if protocol.mode == "score":
        metadata["metric_scale"] = "published (-80 to 100)"
        return WildBenchReport(
            task=cfg.task,
            mode="score",
            model_name=cfg.model.name,
            judge_model=cfg.judge.model,
            baseline_models=[],
            num_examples=num_examples,
            num_judgments=num_judgments,
            num_missing=int(annotations["score"].isna().sum()),
            wb_score=metrics.value,
            raw_mean_score=metrics.raw_mean,
            task_macro_score=metrics.task_macro,
            per_category=metrics.per_category,
            per_baseline={},
            metadata=metadata,
        )

    metadata.update(
        {
            "baseline_outputs_dataset": _source_repo_id(task, "official_outputs"),
            "length_penalty_chars": length_penalty_chars,
        }
    )
    return WildBenchReport(
        task=cfg.task,
        mode="reward",
        model_name=cfg.model.name,
        judge_model=cfg.judge.model,
        baseline_models=baseline_models,
        num_examples=num_examples,
        num_judgments=num_judgments,
        num_missing=int(annotations["reward"].isna().sum()),
        wb_reward=metrics.value,
        task_macro_reward=metrics.task_macro,
        per_category=metrics.per_category,
        per_baseline=metrics.per_baseline,
        metadata=metadata,
    )


def _save_run(
    cfg: RunConfig,
    examples: pd.DataFrame,
    candidate_outputs: pd.Series,
    baseline_outputs: dict[str, pd.Series],
    annotations: pd.DataFrame,
    report: WildBenchReport,
    *,
    started_at: datetime,
    prompt_template: str,
) -> Path:
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{cfg.task}-{slugify(cfg.model.name)}-{slugify(cfg.judge.model)}-{timestamp}"
    )
    res_dir = prepare_run_directory(cfg, Path(cfg.run.result_folder) / run_name)
    annotations.to_parquet(res_dir / "annotations.parquet", index=False)

    output_records = [
        {
            "session_id": sid,
            "model": cfg.model.name,
            "completion": completion,
        }
        for sid, completion in candidate_outputs.items()
    ]
    for baseline, outputs in baseline_outputs.items():
        output_records.extend(
            {
                "session_id": sid,
                "model": baseline,
                "completion": completion,
            }
            for sid, completion in outputs.items()
        )
    pd.DataFrame(output_records).to_parquet(
        res_dir / "model_outputs.parquet", index=False
    )

    result_path = report.save(res_dir / "results.json")
    write_run_metadata_safely(
        output_dir=res_dir,
        entrypoint="judgearena.benchmarks.wildbench.runner.run_wildbench",
        run=cfg.model_dump(),
        results=report.to_dict(),
        input_payloads={"instruction_index": examples.index.astype(str).tolist()},
        judge_user_prompt_template=prompt_template,
        started_at_utc=started_at,
    )
    return result_path


def run_wildbench(
    cfg: RunConfig, resolved_task: ResolvedTaskSpec | None = None
) -> dict[str, object]:
    """Run a registered WB-Score or WB-Reward task."""
    resolved_task = resolved_task or get_packaged_task(cfg.task)
    if resolved_task is None or not isinstance(
        resolved_task.spec.protocol, WildBenchProtocol
    ):
        raise ValueError(f"Task {cfg.task!r} does not use the WildBench protocol.")
    protocol = resolved_task.spec.protocol
    if cfg.wildbench is None or cfg.model.name is None:
        raise ValueError("WildBench runtime settings and model.name are required.")
    if cfg.wildbench.max_words_to_eval is None:
        raise ValueError("WildBench max_words_to_eval was not resolved.")

    started_at = datetime.now(UTC)
    dataset_adapter = resolve_dataset_adapter(resolved_task.spec.dataset.adapter)
    examples = dataset_adapter.load_instructions(resolved_task, data_root / "tables")
    if cfg.generation.n_instructions is not None:
        examples = examples.head(cfg.generation.n_instructions)
    if examples.empty:
        raise ValueError("WildBench selection contains no examples.")
    if "instruction_index" in examples.columns:
        examples = examples.set_index("instruction_index")
    examples = examples.copy()
    examples.index = examples.index.astype(str)

    prompt = resolve_wildbench_prompt(protocol.judge.default_prompt, mode=protocol.mode)
    scorer = resolve_wildbench_scorer(protocol.scoring.adapter)
    if scorer.mode != protocol.mode:
        raise ValueError(
            f"WildBench {protocol.mode} mode cannot use {scorer.mode} scorer "
            f"{scorer.name!r}."
        )

    logger.info("Generating WildBench completions with %s.", cfg.model.name)
    candidate_outputs = _load_or_generate_outputs(
        cfg, examples, cfg.model.name, role="A"
    )

    baseline_outputs: dict[str, pd.Series] = {}
    baseline_models: list[str] = []
    if protocol.mode == "score":
        annotations, num_judgments = _score_annotations(
            cfg, protocol, prompt, scorer, examples, candidate_outputs
        )
    else:
        if not isinstance(protocol.baseline, OfficialOutputsBaseline):
            raise ValueError("WildBench reward mode requires official outputs.")
        baseline_models = (
            [cfg.model.baseline]
            if cfg.model.baseline is not None
            else list(protocol.baseline.references)
        )
        logger.info("Using WildBench baselines: %s", ", ".join(baseline_models))
        official_outputs = dataset_adapter.load_model_outputs(
            resolved_task, data_root / "tables"
        )
        baseline_outputs = {
            baseline: _load_or_generate_outputs(
                cfg,
                examples,
                baseline,
                role="B",
                official_outputs=official_outputs,
            )
            for baseline in baseline_models
        }
        annotations, num_judgments = _reward_annotations(
            cfg,
            protocol,
            prompt,
            scorer,
            examples,
            candidate_outputs,
            baseline_outputs,
            length_penalty_chars=cfg.wildbench.length_penalty_chars,
        )

    metrics = scorer.aggregate(
        examples,
        annotations,
        baseline_models=baseline_models,
    )
    report = _build_report(
        cfg=cfg,
        task=resolved_task,
        protocol=protocol,
        prompt=prompt,
        metrics=metrics,
        baseline_models=baseline_models,
        annotations=annotations,
        num_judgments=num_judgments,
        num_examples=len(examples),
        length_penalty_chars=cfg.wildbench.length_penalty_chars,
    )
    report.render()
    result_path = _save_run(
        cfg,
        examples,
        candidate_outputs,
        baseline_outputs,
        annotations,
        report,
        started_at=started_at,
        prompt_template=prompt.template,
    )
    return {**report.to_dict(), "result_path": str(result_path)}
