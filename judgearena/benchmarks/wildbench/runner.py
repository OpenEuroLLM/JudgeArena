"""WildBench V2 generation, judging, and reporting."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue

from judgearena.artifacts import (
    prepare_run_directory,
    safe_filename,
    to_jsonable,
    write_run_metadata_safely,
)
from judgearena.benchmarks.execution import build_generation_kwargs, build_judge
from judgearena.benchmarks.wildbench.parsing import resolve_wildbench_parser
from judgearena.benchmarks.wildbench.prompting import render_score_prompt
from judgearena.benchmarks.wildbench.scoring import resolve_wildbench_scorer
from judgearena.datasets.registry import resolve_dataset_adapter
from judgearena.log import get_logger
from judgearena.models import do_inference, make_model
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import ResolvedTaskSpec, WildBenchProtocol
from judgearena.utils import (
    cache_function_dataframe,
    data_root,
    generation_cache_token,
    truncate,
)

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def _conversation_prompt(
    messages: list[dict[str, str]], max_chars: int | None
) -> ChatPromptValue:
    prompt_messages = []
    for message in messages:
        message_type = HumanMessage if message["role"] == "user" else AIMessage
        prompt_messages.append(
            message_type(content=truncate(message["content"], max_len=max_chars))
        )
    return ChatPromptValue(messages=prompt_messages)


def generate_outputs(
    examples: pd.DataFrame,
    model_name: str,
    *,
    generation_kwargs: dict[str, object],
    max_input_chars: int | None,
    use_tqdm: bool,
) -> pd.DataFrame:
    """Generate the final assistant turn for each WildBench conversation."""
    model = make_model(model=model_name, **generation_kwargs)
    prompts = [
        _conversation_prompt(messages, max_input_chars)
        for messages in examples["conversation_input"]
    ]
    outputs = do_inference(model, prompts, use_tqdm=use_tqdm)
    return pd.DataFrame(
        {
            "instruction_index": examples["instruction_index"].astype(str),
            "completion": outputs,
        }
    )


def _load_or_generate_outputs(cfg: RunConfig, examples: pd.DataFrame) -> pd.Series:
    generation_kwargs = build_generation_kwargs(cfg, cfg.model.name, role="A")
    sampling_token = generation_cache_token(generation_kwargs)
    session_token = hashlib.sha256(
        "\n".join(examples["instruction_index"].astype(str)).encode()
    ).hexdigest()[:16]
    generated = cache_function_dataframe(
        lambda: generate_outputs(
            examples,
            cfg.model.name,
            generation_kwargs=generation_kwargs,
            max_input_chars=cfg.generation.truncate_all_input_chars,
            use_tqdm=cfg.run.use_tqdm,
        ),
        cache_name=(
            f"wildbench-v2/{safe_filename(cfg.model.name)}/"
            f"{session_token}_{sampling_token}"
        ),
        ignore_cache=cfg.run.ignore_cache,
    )
    generated["instruction_index"] = generated["instruction_index"].astype(str)
    indexed = generated.drop_duplicates("instruction_index").set_index(
        "instruction_index"
    )
    expected = pd.Index(examples["instruction_index"].astype(str))
    missing = expected.difference(indexed.index)
    if not missing.empty:
        raise ValueError(
            f"Generated outputs are missing {len(missing)} WildBench sessions; "
            f"first missing session: {missing[0]}."
        )
    return indexed.loc[expected, "completion"].fillna("").astype(str)


def _score_annotations(
    cfg: RunConfig,
    protocol: WildBenchProtocol,
    examples: pd.DataFrame,
    outputs: pd.Series,
    prompt_template: str,
) -> tuple[pd.DataFrame, int]:
    parser = resolve_wildbench_parser(protocol.judge.parser)
    records: list[dict[str, object]] = []
    judge_inputs: list[ChatPromptValue] = []
    pending_record_indices: list[int] = []

    for (_, example), output in zip(examples.iterrows(), outputs, strict=True):
        rendered = render_score_prompt(
            prompt_template,
            example,
            output,
            max_words=protocol.judge.max_words_to_eval,
            max_chars=cfg.generation.truncate_judge_input_chars,
        )
        record: dict[str, object] = {
            "session_id": str(example["instruction_index"]),
            "model_output": output,
            "prompt": rendered,
        }
        if output.strip():
            pending_record_indices.append(len(records))
            judge_inputs.append(
                ChatPromptValue(messages=[HumanMessage(content=rendered)])
            )
            record.update({"judge_completion": None, "score": None})
        else:
            judge_completion = json.dumps(
                {
                    "strengths": "N/A",
                    "weaknesses": "The model output is empty.",
                    "score": "1",
                }
            )
            record.update({"judge_completion": judge_completion, "score": 1.0})
        records.append(record)

    judge_outputs = (
        do_inference(build_judge(cfg), judge_inputs, use_tqdm=cfg.run.use_tqdm)
        if judge_inputs
        else []
    )
    for record_index, judge_output in zip(
        pending_record_indices, judge_outputs, strict=True
    ):
        records[record_index]["judge_completion"] = judge_output
        records[record_index]["score"] = parser(judge_output)
    return pd.DataFrame(records), len(judge_inputs)


def _render_results(results: dict[str, object]) -> None:
    print("\n" + "=" * 60)
    print("WILDBENCH V2 SCORE".center(60))
    print(f"Task: {results['task']}")
    print(f"Model: {results['model_name']}")
    print(f"Judge: {results['judge_model']}")
    print(f"WB-Score: {results['wb_score']:.2f}")
    print(f"Leaderboard scale: {results['wb_score_leaderboard']:.2f}")
    print(
        f"Scored: {results['num_scored']}/{results['num_examples']} "
        f"(missing: {results['num_missing']})"
    )
    print(f"Results: {results['result_folder']}")
    print("=" * 60 + "\n")


def run_wildbench(
    cfg: RunConfig, resolved_task: ResolvedTaskSpec | None = None
) -> dict[str, object]:
    """Run the packaged official WB-Score task."""
    run_started_at = datetime.now(UTC)
    resolved_task = resolved_task or get_packaged_task(cfg.task)
    if resolved_task is None or not isinstance(
        resolved_task.spec.protocol, WildBenchProtocol
    ):
        raise ValueError(f"Task {cfg.task!r} does not use the WildBench protocol.")
    if cfg.judge.prompt is not None or cfg.judge.prompt_preset is not None:
        raise ValueError(
            "WildBench currently uses the task-shipped official prompt; runtime "
            "prompt overrides are not supported."
        )
    protocol = resolved_task.spec.protocol
    if resolved_task.prompt_text is None:
        raise ValueError(f"Task {cfg.task!r} did not load its judge prompt.")
    prompt_template = resolved_task.prompt_text

    adapter = resolve_dataset_adapter(resolved_task.spec.dataset.adapter)
    examples = adapter.load_instructions(resolved_task, data_root / "tables")
    if cfg.generation.n_instructions is not None:
        examples = examples.head(cfg.generation.n_instructions)
    if examples.empty:
        raise ValueError("WildBench selection contains no examples.")

    logger.info("Generating WildBench completions with %s.", cfg.model.name)
    outputs = _load_or_generate_outputs(cfg, examples)
    annotations, num_judgments = _score_annotations(
        cfg, protocol, examples, outputs, prompt_template
    )
    scorer = resolve_wildbench_scorer(protocol.scoring.adapter)
    metrics = scorer(examples, annotations)

    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    run_name = "-".join(
        [
            safe_filename(cfg.task),
            safe_filename(cfg.model.name),
            safe_filename(cfg.judge.model),
            timestamp,
        ]
    )
    result_folder = prepare_run_directory(cfg, Path(cfg.run.result_folder) / run_name)
    annotations.to_parquet(result_folder / "annotations.parquet", index=False)
    pd.DataFrame(
        {
            "instruction_index": examples["instruction_index"].astype(str),
            "model": cfg.model.name,
            "completion": outputs.to_numpy(),
        }
    ).to_parquet(result_folder / "model_outputs.parquet", index=False)

    results = {
        "task": cfg.task,
        "mode": protocol.mode,
        "model_name": cfg.model.name,
        "judge_model": cfg.judge.model,
        "reference_judge": protocol.judge.reference_judge,
        "num_judgments": num_judgments,
        **metrics,
        "result_folder": str(result_folder),
    }
    result_path = result_folder / "results.json"
    result_path.write_text(
        json.dumps(to_jsonable(results), indent=2) + "\n", encoding="utf-8"
    )
    write_run_metadata_safely(
        output_dir=result_folder,
        entrypoint="judgearena.benchmarks.wildbench.runner.run_wildbench",
        run=cfg.model_dump(),
        results=results,
        input_payloads={
            "instruction_index": examples["instruction_index"].astype(str).tolist()
        },
        judge_user_prompt_template=prompt_template,
        started_at_utc=run_started_at,
    )
    _render_results(results)
    return {**results, "result_path": str(result_path)}
