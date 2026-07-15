"""WildBench V2 generation and checklist-based evaluation.

The judge templates and metric definitions follow the official Apache-2.0
implementation at https://github.com/allenai/WildBench.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from judgearena.constants import WILDBENCH_REWARD_TASK, WILDBENCH_SCORE_TASK
from judgearena.generate_and_evaluate import _build_generation_kwargs
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.wildbench import (
    OFFICIAL_WILDBENCH_BASELINES,
    WILDBENCH_MODEL_OUTPUTS_REPO_ID,
    WILDBENCH_REPO_ID,
    WILDBENCH_TASK_WEIGHTS,
    load_official_wildbench_baseline,
)
from judgearena.log import attach_file_handler, get_logger, make_run_log_path
from judgearena.models import (
    build_default_judge_model_kwargs,
    do_inference,
    make_model,
)
from judgearena.repro import write_run_metadata
from judgearena.utils import (
    cache_function_dataframe,
    generation_cache_token,
    strip_thinking_tags,
    truncate,
)
from judgearena.utils.eval import Report

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)

_PROMPT_PACKAGE = "judgearena.prompts"
_SCORE_PROMPT_FILE = "wildbench/score-v2.txt"
_PAIRWISE_PROMPT_FILE = "wildbench/pairwise-v2.txt"
_CHOICE_TO_A_REWARD = {
    "A++": 1.0,
    "A+": 0.5,
    "A=B": 0.0,
    "B+": -0.5,
    "B++": -1.0,
}


class WildBenchReport(Report):
    """Typed report shared by WB-Score and WB-Reward runs."""

    task: str
    mode: Literal["score", "reward"]
    model_name: str
    judge_model: str
    baseline_models: list[str]
    num_examples: int
    num_judgments: int
    num_missing: int
    wb_score: float | None = None
    raw_mean_score: float | None = None
    task_macro_score: float | None = None
    wb_reward: float | None = None
    task_macro_reward: float | None = None
    per_category: dict[str, float]
    per_baseline: dict[str, float]
    metadata: dict[str, object]

    def render(self) -> None:
        print(f"\n=== WildBench V2 {self.mode.title()} for {self.model_name} ===")
        if self.mode == "score":
            print(f"WB-Score: {self.wb_score:.2f}")
            print(f"Raw mean score: {self.raw_mean_score:.3f}/10")
            print(f"Task-macro WB-Score: {self.task_macro_score:.2f}")
        else:
            print(f"WB-Reward: {self.wb_reward:.2f}")
            print(f"Task-macro WB-Reward: {self.task_macro_reward:.2f}")
            for baseline, reward in self.per_baseline.items():
                print(f"  {baseline}: {reward:.2f}")
        print(
            f"Examples: {self.num_examples} | Judgments: {self.num_judgments} | "
            f"Missing parses: {self.num_missing}"
        )
        if self.per_category:
            print("Per category:")
            for category, value in sorted(self.per_category.items()):
                print(f"  {category}: {value:.2f}")


def _load_prompt_template(relative_path: str) -> str:
    return files(_PROMPT_PACKAGE).joinpath(relative_path).read_text(encoding="utf-8")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shorten_words(text: object, max_words: int) -> str:
    value = "" if text is None else str(text)
    words = value.split(" ")
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "... (truncated)"
    return value


def _prompt_field(text: object, *, max_words: int, max_chars: int | None) -> str:
    return truncate(_shorten_words(text, max_words), max_len=max_chars)


def _checklist_markdown(checklist: object) -> str:
    if not isinstance(checklist, list):
        return ""
    return "".join(f"- {item}\n" for item in checklist)


def render_wildbench_score_prompt(
    example: pd.Series,
    model_output: str,
    *,
    max_words: int,
    max_chars: int | None,
) -> str:
    template = _load_prompt_template(_SCORE_PROMPT_FILE)
    replacements = {
        "$HISTORY": _prompt_field(
            example["history"], max_words=max_words, max_chars=max_chars
        ),
        "$USER_QUERY": _prompt_field(
            example["instruction"], max_words=max_words, max_chars=max_chars
        ),
        "$MODEL_OUTPUT": _prompt_field(
            model_output, max_words=max_words, max_chars=max_chars
        ),
        "$CHECKLIST": _checklist_markdown(example["checklist"]),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def render_wildbench_pairwise_prompt(
    example: pd.Series,
    completion_a: str,
    completion_b: str,
    *,
    max_words: int,
    max_chars: int | None,
) -> str:
    template = _load_prompt_template(_PAIRWISE_PROMPT_FILE)
    replacements = {
        "$HISTORY": _prompt_field(
            example["history"], max_words=max_words, max_chars=max_chars
        ),
        "$USER_QUERY": _prompt_field(
            example["instruction"], max_words=max_words, max_chars=max_chars
        ),
        "$CANDIDATE_A": _prompt_field(
            completion_a or "[This model response is empty.]",
            max_words=max_words,
            max_chars=max_chars,
        ),
        "$CANDIDATE_B": _prompt_field(
            completion_b or "[This model response is empty.]",
            max_words=max_words,
            max_chars=max_chars,
        ),
        "$CHECKLIST": _checklist_markdown(example["checklist"]),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def _parse_json_object(text: str) -> dict[str, object] | None:
    cleaned = strip_thinking_tags(text).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, json.JSONDecodeError):
        pass

    decoder = json.JSONDecoder()
    for start in [i for i, char in enumerate(cleaned) if char == "{"]:
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_wildbench_score(text: str) -> float | None:
    payload = _parse_json_object(text)
    value = payload.get("score") if payload is not None else None
    if value is None:
        matches = re.findall(
            r'["\']?score["\']?\s*:\s*["\']?(-?\d+(?:\.\d+)?)',
            strip_thinking_tags(text),
            flags=re.IGNORECASE,
        )
        value = matches[-1] if matches else None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if np.isfinite(score) and 1 <= score <= 10 else None


def parse_wildbench_choice(text: str) -> str | None:
    payload = _parse_json_object(text)
    value = payload.get("choice") if payload is not None else None
    if isinstance(value, str):
        normalized = value.strip().upper().replace(" ", "")
        if normalized in _CHOICE_TO_A_REWARD:
            return normalized
    matches = re.findall(r"A\+\+|B\+\+|A=B|A\+|B\+", strip_thinking_tags(text))
    return matches[-1] if matches else None


def choice_to_candidate_reward(choice: str, *, candidate_is_a: bool) -> float:
    reward_a = _CHOICE_TO_A_REWARD[choice]
    return reward_a if candidate_is_a else -reward_a


def apply_wildbench_length_penalty(
    reward: float,
    candidate_output: str,
    baseline_output: str,
    length_penalty_chars: int | None,
) -> float:
    """Convert only a length-advantaged slight win/loss into a tie."""
    if length_penalty_chars is None or abs(reward) != 0.5:
        return reward
    if (
        reward > 0
        and len(candidate_output) > len(baseline_output) + length_penalty_chars
    ):
        return 0.0
    if (
        reward < 0
        and len(baseline_output) > len(candidate_output) + length_penalty_chars
    ):
        return 0.0
    return reward


def _conversation_prompt(messages: list[dict[str, str]], max_chars: int | None):
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
    return indexed.loc[expected, "completion"].fillna("").astype(str)


def _load_or_generate_outputs(
    cfg: RunConfig,
    examples: pd.DataFrame,
    model_name: str,
    *,
    role: Literal["A", "B"],
) -> pd.Series:
    if role == "B":
        official = load_official_wildbench_baseline(model_name)
        if official is not None:
            return _align_outputs(official, examples, model_name)

    generation_kwargs = _build_generation_kwargs(cfg, model_name, role=role)
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


def _make_judge(cfg: RunConfig):
    return make_model(
        model=cfg.judge.model,
        **build_default_judge_model_kwargs(
            cfg.judge.model,
            cfg.model.engine_kwargs,
            judge_engine_kwargs_override=cfg.judge.model_kwargs(
                fallback_chat_template=cfg.model.chat_template,
            ),
        ),
    )


def _run_judge_prompts(judge_model, prompts: list[str], *, use_tqdm: bool) -> list[str]:
    if not prompts:
        return []
    return do_inference(judge_model, prompts, use_tqdm=use_tqdm)


def _weighted_task_macro(per_category: dict[str, float]) -> float:
    weighted = [
        (value, WILDBENCH_TASK_WEIGHTS[category])
        for category, value in per_category.items()
        if category in WILDBENCH_TASK_WEIGHTS and np.isfinite(value)
    ]
    if not weighted:
        return float("nan")
    return float(
        sum(value * weight for value, weight in weighted)
        / sum(weight for _, weight in weighted)
    )


def _categories_for(examples: pd.DataFrame, session_id: str) -> list[str]:
    categories = examples.loc[session_id, "task_categories"]
    return categories if isinstance(categories, list) else []


def _score_metrics(
    examples: pd.DataFrame, annotations: pd.DataFrame
) -> tuple[float, float, dict[str, float]]:
    valid = annotations.dropna(subset=["score"])
    raw_mean = float(valid["score"].mean()) if not valid.empty else float("nan")
    category_values: dict[str, list[float]] = {}
    for row in valid.itertuples(index=False):
        for category in _categories_for(examples, row.session_id):
            category_values.setdefault(category, []).append(float(row.score))
    per_category = {
        category: (float(np.mean(values)) - 5.0) * 20.0
        for category, values in category_values.items()
    }
    return raw_mean, (raw_mean - 5.0) * 20.0, per_category


def _reward_metrics(
    examples: pd.DataFrame, canonical: pd.DataFrame, baseline_models: list[str]
) -> tuple[float, dict[str, float], dict[str, float]]:
    per_baseline = {}
    for baseline in baseline_models:
        values = canonical.loc[
            canonical["baseline_model"] == baseline, "reward"
        ].dropna()
        per_baseline[baseline] = (
            float(values.mean()) * 100.0 if not values.empty else float("nan")
        )
    valid_baselines = [value for value in per_baseline.values() if np.isfinite(value)]
    wb_reward = float(np.mean(valid_baselines)) if valid_baselines else float("nan")

    category_baseline_values: dict[str, dict[str, list[float]]] = {}
    for row in canonical.dropna(subset=["reward"]).itertuples(index=False):
        for category in _categories_for(examples, row.session_id):
            category_baseline_values.setdefault(category, {}).setdefault(
                row.baseline_model, []
            ).append(float(row.reward))
    per_category = {}
    for category, values_by_baseline in category_baseline_values.items():
        baseline_means = [
            float(np.mean(values)) for values in values_by_baseline.values() if values
        ]
        per_category[category] = float(np.mean(baseline_means)) * 100.0
    return wb_reward, per_baseline, per_category


def _score_annotations(
    cfg: RunConfig,
    examples: pd.DataFrame,
    candidate_outputs: pd.Series,
) -> tuple[pd.DataFrame, int]:
    prompts = []
    pending_ids = []
    records = []
    for session_id, example in examples.iterrows():
        output = candidate_outputs.loc[str(session_id)]
        prompt = render_wildbench_score_prompt(
            example,
            output,
            max_words=cfg.wildbench.max_words_to_eval,
            max_chars=cfg.generation.truncate_judge_input_chars,
        )
        if output.strip():
            prompts.append(prompt)
            pending_ids.append(str(session_id))
        else:
            records.append(
                {
                    "session_id": str(session_id),
                    "prompt": prompt,
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
        _make_judge(cfg), prompts, use_tqdm=cfg.run.use_tqdm
    )
    for session_id, prompt, judge_output in zip(
        pending_ids, prompts, judge_outputs, strict=True
    ):
        records.append(
            {
                "session_id": session_id,
                "prompt": prompt,
                "judge_completion": judge_output,
                "score": parse_wildbench_score(judge_output),
            }
        )
    return pd.DataFrame(records).sort_values("session_id"), len(prompts)


def _reward_annotations(
    cfg: RunConfig,
    examples: pd.DataFrame,
    candidate_outputs: pd.Series,
    baseline_outputs: dict[str, pd.Series],
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

            if cfg.judge.swap_mode == "both":
                positions = [True, False]
            else:
                positions = [bool(rng.integers(0, 2))]

            for pass_index, candidate_is_a in enumerate(positions):
                completion_a = candidate if candidate_is_a else baseline
                completion_b = baseline if candidate_is_a else candidate
                prompt = render_wildbench_pairwise_prompt(
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
                    "prompt": prompt,
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
                    pending_prompts.append(prompt)
                    record["judge_completion"] = None
                    record["choice"] = None
                else:
                    record["judge_completion"] = json.dumps({"choice": choice})
                    record["choice"] = choice
                prompt_records.append(record)

    judge_outputs = _run_judge_prompts(
        _make_judge(cfg), pending_prompts, use_tqdm=cfg.run.use_tqdm
    )
    for record_index, judge_output in zip(pending_indices, judge_outputs, strict=True):
        prompt_records[record_index]["judge_completion"] = judge_output
        prompt_records[record_index]["choice"] = parse_wildbench_choice(judge_output)

    for record in prompt_records:
        choice = record["choice"]
        if choice is None:
            record["raw_reward"] = np.nan
            record["reward"] = np.nan
            continue
        raw_reward = choice_to_candidate_reward(
            choice, candidate_is_a=record["candidate_is_a"]
        )
        record["raw_reward"] = raw_reward
        record["reward"] = apply_wildbench_length_penalty(
            raw_reward,
            record["candidate_output"],
            record["baseline_output"],
            cfg.wildbench.length_penalty_chars,
        )
    return pd.DataFrame(prompt_records), len(pending_prompts)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "model"


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
        f"{cfg.task}-{_slugify(cfg.model.name)}-{_slugify(cfg.judge.model)}-{timestamp}"
    )
    res_dir = Path(cfg.run.result_folder) / run_name
    res_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.run.no_log_file:
        attach_file_handler(make_run_log_path(res_dir))

    from judgearena.config import dump_config

    dump_config(cfg, res_dir / "config.yaml")
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
    write_run_metadata(
        output_dir=res_dir,
        entrypoint="judgearena.wildbench.main",
        run=cfg.model_dump(),
        results=report.to_dict(),
        input_payloads={"instruction_index": examples.index.astype(str).tolist()},
        judge_user_prompt_template=prompt_template,
        started_at_utc=started_at,
    )
    return result_path


def main(cfg: RunConfig) -> dict[str, object]:
    """Run WB-Score or WB-Reward using the official WildBench V2 contract."""
    assert cfg.wildbench is not None
    assert cfg.model.name is not None
    started_at = datetime.now(UTC)
    examples = load_instructions(cfg.task, n_instructions=cfg.generation.n_instructions)
    if examples.empty:
        raise ValueError("WildBench selection contains no examples.")
    examples = examples.copy()
    examples.index = examples.index.astype(str)

    logger.info("Generating WildBench completions with %s.", cfg.model.name)
    candidate_outputs = _load_or_generate_outputs(
        cfg, examples, cfg.model.name, role="A"
    )

    baseline_outputs: dict[str, pd.Series] = {}
    if cfg.task == WILDBENCH_SCORE_TASK:
        annotations, num_judgments = _score_annotations(
            cfg, examples, candidate_outputs
        )
        raw_mean, wb_score, per_category = _score_metrics(examples, annotations)
        prompt_template = _load_prompt_template(_SCORE_PROMPT_FILE)
        report = WildBenchReport(
            task=cfg.task,
            mode="score",
            model_name=cfg.model.name,
            judge_model=cfg.judge.model,
            baseline_models=[],
            num_examples=len(examples),
            num_judgments=num_judgments,
            num_missing=int(annotations["score"].isna().sum()),
            wb_score=wb_score,
            raw_mean_score=raw_mean,
            task_macro_score=_weighted_task_macro(per_category),
            per_category=per_category,
            per_baseline={},
            metadata={
                "dataset": WILDBENCH_REPO_ID,
                "metric_scale": "published (-80 to 100)",
                "prompt_sha256": _sha256(prompt_template),
                "paper": "https://arxiv.org/abs/2406.04770",
            },
        )
    elif cfg.task == WILDBENCH_REWARD_TASK:
        baseline_models = (
            [cfg.model.baseline]
            if cfg.model.baseline is not None
            else list(OFFICIAL_WILDBENCH_BASELINES)
        )
        logger.info("Using WildBench baselines: %s", ", ".join(baseline_models))
        baseline_outputs = {
            baseline: _load_or_generate_outputs(cfg, examples, baseline, role="B")
            for baseline in baseline_models
        }
        annotations, num_judgments = _reward_annotations(
            cfg, examples, candidate_outputs, baseline_outputs
        )
        canonical = (
            annotations.groupby(
                ["session_id", "baseline_model"], as_index=False, sort=False
            )["reward"]
            .mean()
            .reset_index(drop=True)
        )
        wb_reward, per_baseline, per_category = _reward_metrics(
            examples, canonical, baseline_models
        )
        prompt_template = _load_prompt_template(_PAIRWISE_PROMPT_FILE)
        report = WildBenchReport(
            task=cfg.task,
            mode="reward",
            model_name=cfg.model.name,
            judge_model=cfg.judge.model,
            baseline_models=baseline_models,
            num_examples=len(examples),
            num_judgments=num_judgments,
            num_missing=int(annotations["reward"].isna().sum()),
            wb_reward=wb_reward,
            task_macro_reward=_weighted_task_macro(per_category),
            per_category=per_category,
            per_baseline=per_baseline,
            metadata={
                "dataset": WILDBENCH_REPO_ID,
                "baseline_outputs_dataset": WILDBENCH_MODEL_OUTPUTS_REPO_ID,
                "length_penalty_chars": cfg.wildbench.length_penalty_chars,
                "prompt_sha256": _sha256(prompt_template),
                "paper": "https://arxiv.org/abs/2406.04770",
            },
        )
    else:  # guarded by RunConfig; keeps direct callers honest
        raise ValueError(f"Unsupported WildBench task: {cfg.task!r}")

    report.render()
    result_path = _save_run(
        cfg,
        examples,
        candidate_outputs,
        baseline_outputs,
        annotations,
        report,
        started_at=started_at,
        prompt_template=prompt_template,
    )
    return {**report.to_dict(), "result_path": str(result_path)}
