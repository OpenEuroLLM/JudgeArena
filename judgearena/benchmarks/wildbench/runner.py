"""WildBench V2 generation, judging, and reporting."""

from __future__ import annotations

import hashlib
import json
import random
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
from judgearena.benchmarks.wildbench.prompting import (
    render_reward_prompt,
    render_score_prompt,
)
from judgearena.benchmarks.wildbench.scoring import (
    apply_length_penalty,
    candidate_reward,
    resolve_wildbench_scorer,
)
from judgearena.datasets.registry import resolve_dataset_adapter
from judgearena.log import get_logger
from judgearena.models import do_inference, make_model
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import (
    OfficialOutputsBaseline,
    ResolvedTaskSpec,
    WildBenchProtocol,
)
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


def _reward_references(cfg: RunConfig, protocol: WildBenchProtocol) -> tuple[str, ...]:
    baseline = protocol.baseline
    if not isinstance(baseline, OfficialOutputsBaseline):
        raise ValueError("WB-Reward requires official reference outputs.")
    if cfg.model.baseline is None:
        return baseline.references
    if cfg.model.baseline not in baseline.references:
        raise ValueError(
            f"Unknown WB-Reward reference {cfg.model.baseline!r}; choose from "
            f"{list(baseline.references)}."
        )
    return (cfg.model.baseline,)


def _aligned_reference_outputs(
    official_outputs: pd.DataFrame,
    examples: pd.DataFrame,
    reference: str,
) -> pd.Series:
    selected = official_outputs.loc[official_outputs["model"] == reference].copy()
    selected["instruction_index"] = selected["instruction_index"].astype(str)
    indexed = selected.drop_duplicates("instruction_index").set_index(
        "instruction_index"
    )
    expected = pd.Index(examples["instruction_index"].astype(str))
    missing = expected.difference(indexed.index)
    if not missing.empty:
        raise ValueError(
            f"Official outputs for {reference!r} are missing {len(missing)} "
            f"WildBench sessions; first missing session: {missing[0]}."
        )
    return indexed.loc[expected, "output"].fillna("").astype(str)


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


def _reward_annotations(
    cfg: RunConfig,
    protocol: WildBenchProtocol,
    examples: pd.DataFrame,
    candidate_outputs: pd.Series,
    reference_outputs: dict[str, pd.Series],
    prompt_template: str,
) -> tuple[pd.DataFrame, int]:
    parser = resolve_wildbench_parser(protocol.judge.parser)
    if protocol.judge.assignment_seed is None:
        raise ValueError("WB-Reward requires an assignment seed.")
    if protocol.judge.max_words_to_eval is None:
        raise ValueError("WB-Reward requires a judge word limit.")
    if cfg.wildbench is None:
        raise ValueError("WB-Reward runtime settings were not initialized.")

    records: list[dict[str, object]] = []
    judge_inputs: list[ChatPromptValue] = []
    pending_record_indices: list[int] = []
    for reference, baseline_outputs in reference_outputs.items():
        rng = random.Random(protocol.judge.assignment_seed)
        for (_, example), candidate, baseline in zip(
            examples.iterrows(),
            candidate_outputs,
            baseline_outputs,
            strict=True,
        ):
            candidate_is_a = rng.random() < 0.5
            rendered = render_reward_prompt(
                prompt_template,
                example,
                candidate,
                baseline,
                candidate_is_a=candidate_is_a,
                max_words=protocol.judge.max_words_to_eval,
                max_chars=cfg.generation.truncate_judge_input_chars,
            )
            record: dict[str, object] = {
                "session_id": str(example["instruction_index"]),
                "baseline_model": reference,
                "candidate_is_a": candidate_is_a,
                "candidate_output": rendered.candidate_output,
                "baseline_output": rendered.baseline_output,
                "prompt": rendered.text,
                "judge_completion": None,
                "choice": None,
                "raw_reward": None,
                "reward": None,
            }

            if not candidate.strip() and not baseline.strip():
                choice = "A=B"
                reason = "Both responses are empty."
            elif not candidate.strip():
                choice = "B++" if candidate_is_a else "A++"
                reason = "The candidate response is empty."
            elif not baseline.strip():
                choice = "A++" if candidate_is_a else "B++"
                reason = "The reference response is empty."
            else:
                choice = None
                reason = None

            if choice is None:
                pending_record_indices.append(len(records))
                judge_inputs.append(
                    ChatPromptValue(messages=[HumanMessage(content=rendered.text)])
                )
            else:
                record["judge_completion"] = json.dumps(
                    {"reason": reason, "choice": choice}
                )
                record["choice"] = choice
            records.append(record)

    judge_outputs = (
        do_inference(build_judge(cfg), judge_inputs, use_tqdm=cfg.run.use_tqdm)
        if judge_inputs
        else []
    )
    for record_index, judge_output in zip(
        pending_record_indices, judge_outputs, strict=True
    ):
        parsed = parser(judge_output)
        records[record_index]["judge_completion"] = judge_output
        records[record_index]["choice"] = parsed if isinstance(parsed, str) else None

    for record in records:
        choice = record["choice"]
        if not isinstance(choice, str):
            continue
        raw_reward = candidate_reward(
            choice, candidate_is_a=bool(record["candidate_is_a"])
        )
        record["raw_reward"] = raw_reward
        record["reward"] = apply_length_penalty(
            raw_reward,
            str(record["candidate_output"]),
            str(record["baseline_output"]),
            cfg.wildbench.length_penalty_chars,
        )
    return pd.DataFrame(records), len(judge_inputs)


def _render_results(results: dict[str, object]) -> None:
    print("\n" + "=" * 60)
    heading = (
        "WILDBENCH V2 REWARD" if results["mode"] == "reward" else "WILDBENCH V2 SCORE"
    )
    print(heading.center(60))
    print(f"Task: {results['task']}")
    print(f"Model: {results['model_name']}")
    print(f"Judge: {results['judge_model']}")
    if results["mode"] == "reward":
        print(f"WB-Reward-Mix: {results['wb_reward']:.2f}")
        print(f"References: {', '.join(results['reference_models'])}")
        print(
            f"Scored: {results['num_scored']}/{results['num_annotations']} "
            f"(missing: {results['num_missing']})"
        )
    else:
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
    """Run a packaged official WildBench V2 task."""
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
    reference_outputs: dict[str, pd.Series] = {}
    if protocol.mode == "reward":
        official_outputs = adapter.load_model_outputs(
            resolved_task, data_root / "tables"
        )
        if official_outputs is None:
            raise ValueError("WB-Reward task did not load official reference outputs.")
        for reference in _reward_references(cfg, protocol):
            reference_outputs[reference] = _aligned_reference_outputs(
                official_outputs, examples, reference
            )
        annotations, num_judgments = _reward_annotations(
            cfg,
            protocol,
            examples,
            outputs,
            reference_outputs,
            prompt_template,
        )
    else:
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
    output_frames = [
        pd.DataFrame(
            {
                "instruction_index": examples["instruction_index"].astype(str),
                "model": cfg.model.name,
                "completion": outputs.to_numpy(),
            }
        )
    ]
    for reference, reference_output in reference_outputs.items():
        output_frames.append(
            pd.DataFrame(
                {
                    "instruction_index": examples["instruction_index"].astype(str),
                    "model": reference,
                    "completion": reference_output.to_numpy(),
                }
            )
        )
    pd.concat(output_frames, ignore_index=True).to_parquet(
        result_folder / "model_outputs.parquet", index=False
    )

    results = {
        "task": cfg.task,
        "mode": protocol.mode,
        "model_name": cfg.model.name,
        "judge_model": cfg.judge.model,
        "reference_judge": protocol.judge.reference_judge,
        "num_judgments": num_judgments,
        **(
            {"reference_models": list(reference_outputs)}
            if protocol.mode == "reward"
            else {}
        ),
        "num_examples": int(len(examples)),
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
