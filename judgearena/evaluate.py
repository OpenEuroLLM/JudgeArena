import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate

from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.judge_prompt_presets import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    resolve_pairwise_judge_prompt,
)
from judgearena.openrouter_reference_pricing import (
    OpenRouterReferencePricingTracker,
    build_openrouter_reference_pricing_summary,
    format_openrouter_reference_pricing_summary,
)
from judgearena.repro import _to_jsonable, write_run_metadata
from judgearena.utils import (
    LimitEventTracker,
    compute_pref_summary,
    data_root,
    do_inference,
    download_hf,
    infer_model_spec_from_instance,
    read_df,
    strip_thinking_tags,
    strip_thinking_tags_with_metadata,
    truncate_with_metadata,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_PREFLIGHT_MAX_ITERATIONS = 3
_PREFLIGHT_RESERVED_TOKENS = 256
_PREFLIGHT_MIN_COMPLETION_CHARS = 512


class PairScore:
    def __init__(self, *, parser_mode: str = "score"):
        super(PairScore).__init__()
        self.temperature = 0.3
        self.parser_mode = parser_mode

    def preference_from_scores(self, score_a: float, score_b: float) -> float:
        return 1 - np.exp(self.temperature * score_a) / (
            np.exp(self.temperature * np.array([score_a, score_b])).sum()
        )

    def parse_model_raw(self, judge_completion: str) -> float | None:
        judge_completion = strip_thinking_tags(judge_completion)
        if self.parser_mode == "verdict":
            return self._parse_bracketed_verdict(judge_completion)
        if self.parser_mode == "score":
            return self._parse_numeric_scores(judge_completion)
        raise ValueError(f"Unsupported parser_mode '{self.parser_mode}'.")

    def _parse_numeric_scores(self, judge_completion: str) -> float | None:
        lowered = judge_completion.lower()
        score_a = self.get_regexp_match(lowered, r'score.*?a[": *\n]*(-?\d+)')
        score_b = self.get_regexp_match(lowered, r'score.*?b[": *\n]*(-?\d+)')
        if score_a is None or score_b is None:
            return None
        return float(self.preference_from_scores(score_a, score_b))

    def _parse_bracketed_verdict(self, judge_completion: str) -> float | None:
        verdict_match = re.search(r"\[\[\s*([ABCabc])\s*\]\]", judge_completion)
        if verdict_match is None:
            return None
        bracketed_verdict = verdict_match.group(1).lower()
        return {
            "a": 0.0,
            "b": 1.0,
            "c": 0.5,
        }[bracketed_verdict]

    def get_regexp_match(self, s: str, regex: str, group_index: int = 1):
        m = re.search(re.compile(regex), s)
        if m is None:
            return None
        else:
            return float(m.group(group_index).strip(" "))


def load_judge_system_and_user_prompt(
    provide_explanation: bool = True,
    multi_turn: bool = False,
) -> tuple[str, str]:
    resolved = resolve_pairwise_judge_prompt(
        prompt_preset=DEFAULT_JUDGE_PROMPT_PRESET,
        provide_explanation=provide_explanation,
        multi_turn=multi_turn,
    )
    return resolved.system_prompt or "", resolved.user_prompt_template


def resolve_judge_prompts(
    *,
    provide_explanation: bool,
    multi_turn: bool = False,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> ResolvedJudgePrompt:
    return resolve_pairwise_judge_prompt(
        prompt_preset=prompt_preset,
        provide_explanation=provide_explanation,
        multi_turn=multi_turn,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )


def evaluate_completions(
    dataset: str = "alpaca-eval",
    judge_chat_model: LLM = None,
    method_A: str = "gpt4_1106_preview",
    method_B: str = "llama-2-70b-chat-hf",
    num_annotations: int | None = 50,
    use_tqdm: bool = False,
    truncate_input_chars: int | None = 8192,
    provide_explanation: bool = False,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    strip_thinking_before_judging: bool = False,
):
    """
    :param dataset:
    :param judge_chat_model:
    :param method_A: one method to evaluate, can be a method existing in `dataset` or a local path to the completion
    of a local method. The path should be a dataframe ending with ".csv.zip" or ".parquet", have columns
    "instruction_index" and "output" and should contains all the instruction of `dataset`.
    :param method_B: another method to evaluate against `method_A`
    :param num_annotations: if specified will do at most `num_annotations` annotations
    :param use_tqdm:
    :param truncate_input_chars: if specified, truncates the length of completion, useful to save cost and avoid
    exceeding context limit
    :return:
    """
    run_started_at = datetime.now(UTC)
    local_path_tables = data_root / "tables"
    if is_arena_hard_dataset(dataset):
        download_arena_hard(dataset=dataset, local_tables_path=local_path_tables)
    else:
        download_hf(name=dataset, local_path=local_path_tables)

    instructions = load_instructions(
        dataset=dataset,
    ).loc[:, "instruction"]

    # Only loads if the per-dataset local path exists; some datasets (e.g.
    # language slices of m-arena-hard for which no baseline has been written
    # yet) may not ship a local completions file.
    dataset_output_path = local_path_tables / "model_outputs" / f"{dataset}.csv.zip"
    if dataset_output_path.exists():
        df_outputs = read_df(dataset_output_path)
        # empty strings are encoded as Nan in csv
        df_outputs.loc[:, "output"] = df_outputs.loc[:, "output"].fillna("")
        df_outputs = df_outputs.pivot_table(
            index="instruction_index", columns="model", values="output", aggfunc="last"
        ).sort_index()
        df_outputs = df_outputs.loc[instructions.index]
    else:
        df_outputs = None

    def get_output(df_outputs: pd.DataFrame, dataset: str, method: str):
        if Path(method).exists():
            print(f"Path {method} exists, loads local model completions.")
            df = read_df(Path(method)).set_index("instruction_index").sort_index()
            print(f"Loaded {len(df)} completions.")
            df.loc[:, "output"] = df.loc[:, "output"].fillna("")
            return df.loc[:, "output"]
        else:
            print(f"Loading {method} from {dataset} dataset.")
            assert method in df_outputs.columns, (
                f"Method {method} not present, pick among {df_outputs.columns.tolist()}"
            )
            return df_outputs.loc[:, method].sort_index()

    completions_A = get_output(df_outputs=df_outputs, dataset=dataset, method=method_A)
    completions_B = get_output(df_outputs=df_outputs, dataset=dataset, method=method_B)
    if num_annotations is not None:
        instructions = instructions.head(num_annotations)
        completions_A = completions_A.head(num_annotations)
        completions_B = completions_B.head(num_annotations)
    assert completions_A.index.tolist() == completions_B.index.tolist(), (
        f"Index mismatch between methods {method_A} and {method_B}."
    )

    if judge_chat_model is None:
        from langchain_together.llms import Together

        judge_chat_model = Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    judge_model_spec = infer_model_spec_from_instance(judge_chat_model)
    usage_tracker = OpenRouterReferencePricingTracker()
    limit_event_tracker = LimitEventTracker()

    unique_string = dataset + "-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = data_root / "judge-evals" / unique_string
    print(f"Saving results in {output_folder}")
    output_folder.mkdir(parents=True, exist_ok=True)
    resolved_prompt = resolve_judge_prompts(
        provide_explanation=provide_explanation,
        prompt_preset=prompt_preset,
    )

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions.tolist(),
        completions_A=completions_A.loc[instructions.index].tolist(),
        completions_B=completions_B.loc[instructions.index].tolist(),
        case_ids=instructions.index.tolist(),
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        use_tqdm=use_tqdm,
        truncate_input_chars=truncate_input_chars,
        provide_explanation=provide_explanation,
        strip_thinking_before_judging=strip_thinking_before_judging,
        usage_tracker=usage_tracker,
        usage_phase="judge",
        usage_model_spec=judge_model_spec,
        limit_event_tracker=limit_event_tracker,
    )

    # Pairwise judge results
    score_parser = PairScore(parser_mode=resolved_prompt.parser_mode)
    prefs = pd.Series(
        [
            score_parser.parse_model_raw(annotation.judge_completion)
            for annotation in annotations
        ]
    )
    results = {
        **compute_pref_summary(prefs),
        "judge_prompt_preset": resolved_prompt.preset_name,
        "limit_events": limit_event_tracker.build_summary(),
    }
    pd.DataFrame(annotations).to_csv(output_folder / "annotations.csv", index=False)

    print(f"{method_A} against {method_B}:\n{results}")
    with open(output_folder / "results.json", "w") as f:
        json.dump(_to_jsonable(results), f, allow_nan=False)
    pricing_reference = None
    if judge_model_spec is not None:
        pricing_reference = build_openrouter_reference_pricing_summary(
            tracker=usage_tracker,
            phase_model_specs={"judge": judge_model_spec},
        )
        print(format_openrouter_reference_pricing_summary(pricing_reference))

    run_metadata = {
        "dataset": dataset,
        "method_A": method_A,
        "method_B": method_B,
        "num_annotations": num_annotations,
        "n_annotations": len(instructions),
        "use_tqdm": use_tqdm,
        "truncate_input_chars": truncate_input_chars,
        "provide_explanation": provide_explanation,
        "judge_prompt_preset": resolved_prompt.preset_name,
        "strip_thinking_before_judging": strip_thinking_before_judging,
    }

    try:
        write_run_metadata(
            output_dir=output_folder,
            entrypoint="judgearena.evaluate.evaluate_completions",
            run=run_metadata,
            results=results,
            input_payloads={
                "instruction_index": instructions.index.tolist(),
                "instructions": instructions.tolist(),
                "completions_A": completions_A.loc[instructions.index].tolist(),
                "completions_B": completions_B.loc[instructions.index].tolist(),
            },
            judge_system_prompt=resolved_prompt.system_prompt,
            judge_user_prompt_template=resolved_prompt.user_prompt_template,
            started_at_utc=run_started_at,
            pricing_reference=pricing_reference,
        )
    except OSError as e:
        print(f"Warning: failed to write run metadata: {e}")


@dataclass
class JudgeAnnotation:
    instruction: str  # instruction from the user
    completion_A: str  # completion of the first model
    completion_B: str  # completion of the second model
    judge_completion: str  # output of the judge
    judge_input: str | None = None  # input that was passed to the judge
    completion_A_for_judge: str | None = None
    completion_B_for_judge: str | None = None
    completion_A_reasoning_stripped: bool = False
    completion_B_reasoning_stripped: bool = False
    completion_A_truncated_for_judge: bool = False
    completion_B_truncated_for_judge: bool = False


def annotate_battles(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    case_ids: list[object] | None = None,
    system_prompt: str | None = None,
    user_prompt_template: str = None,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
    provide_explanation: bool = False,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    strip_thinking_before_judging: bool = False,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    usage_model_spec: str | None = None,
    limit_event_tracker: LimitEventTracker | None = None,
    judge_tokenizer: "PreTrainedTokenizerBase | None" = None,
    max_judge_model_len: int | None = None,
    max_out_tokens_judge: int | None = None,
) -> list[JudgeAnnotation]:
    """
    Directly evaluate from list of instructions and completions
    Can also pass custom LLM judge prompts, if not passed uses defaults
    `system_prompt, user_prompt_template = load_judge_system_and_user_prompt()`
    Example usage:
    ```python
    annotations = annotate_battles(
        # can be any langchain ChatModel, supports OpenAI, Together, vLLM, ...
        judge_chat_model=Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        # the instructions we want to evaluate
        user_prompts=["Write numbers between 1 and 5."],
        # the completions we want to evaluate for the first model
        completions_A=["1 2 3 4 5."],
        # the completions we want to evaluate for the second model
        completions_B=["No"],
    )
    ```
    :param provide_explanation:
    :param judge_chat_model:
    :param instructions:
    :param completions_A:
    :param completions_B:
    :param system_prompt:
    :param user_prompt_template:
    :param truncate_input_chars: Max characters to truncate completions before sending to judge.
    :param use_tqdm:
    :param judge_tokenizer: Optional HF tokenizer matching the judge model; when
        supplied together with ``max_judge_model_len`` triggers a preflight
        tokenize-and-retry pass that shrinks per-completion character caps until
        the rendered prompt fits the judge context window. Converts the hard
        ``VLLMValidationError`` class into a soft ``judge_input_token_truncation``
        limit event.
    :param max_judge_model_len: Judge-side ``max_model_len``; required for the
        preflight pass to be active.
    :param max_out_tokens_judge: Judge-side output budget subtracted from
        ``max_judge_model_len`` to derive the per-request prompt budget.
    :return:
    """
    # alternatively pass list of tuples
    assert len(instructions) == len(completions_A) == len(completions_B)
    if case_ids is None:
        case_ids = [None] * len(instructions)
    assert len(case_ids) == len(instructions)

    resolved_prompt = resolve_judge_prompts(
        provide_explanation=provide_explanation,
        prompt_preset=prompt_preset,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )
    message_templates: list[tuple[str, str]] = []
    if resolved_prompt.system_prompt is not None:
        message_templates.append(("system", resolved_prompt.system_prompt))
    message_templates.append(("user", resolved_prompt.user_prompt_template))

    prompt_template = ChatPromptTemplate.from_messages(message_templates)
    truncated_completion_count = 0
    input_payloads = []
    annotation_input_metadata: list[dict[str, object]] = []
    for case_id, user_prompt, completion_A, completion_B in zip(
        case_ids, instructions, completions_A, completions_B, strict=True
    ):
        raw_completion_A = completion_A if isinstance(completion_A, str) else ""
        raw_completion_B = completion_B if isinstance(completion_B, str) else ""
        completion_A_for_judge = raw_completion_A
        completion_B_for_judge = raw_completion_B
        stripped_A = False
        stripped_B = False
        if strip_thinking_before_judging:
            completion_A_for_judge, stripped_A = strip_thinking_tags_with_metadata(
                completion_A_for_judge
            )
            completion_B_for_judge, stripped_B = strip_thinking_tags_with_metadata(
                completion_B_for_judge
            )
            if stripped_A and limit_event_tracker is not None:
                limit_event_tracker.record(
                    "thinking_trace_stripped_before_judging",
                    stage="judge_input",
                    field="completion_A",
                    case_id=case_id,
                    original_length=len(raw_completion_A),
                    final_length=len(completion_A_for_judge),
                )
            if stripped_B and limit_event_tracker is not None:
                limit_event_tracker.record(
                    "thinking_trace_stripped_before_judging",
                    stage="judge_input",
                    field="completion_B",
                    case_id=case_id,
                    original_length=len(raw_completion_B),
                    final_length=len(completion_B_for_judge),
                )
        truncated_completion_A, truncated_A = truncate_with_metadata(
            completion_A_for_judge,
            max_len=truncate_input_chars,
            tracker=limit_event_tracker,
            kind="judge_input_char_truncation",
            stage="judge_input",
            field="completion_A",
            case_id=case_id,
        )
        truncated_completion_B, truncated_B = truncate_with_metadata(
            completion_B_for_judge,
            max_len=truncate_input_chars,
            tracker=limit_event_tracker,
            kind="judge_input_char_truncation",
            stage="judge_input",
            field="completion_B",
            case_id=case_id,
        )
        truncated_completion_count += int(truncated_A)
        truncated_completion_count += int(truncated_B)
        input_payloads.append(
            {
                "user_prompt": user_prompt,
                "completion_A": truncated_completion_A,
                "completion_B": truncated_completion_B,
            }
        )
        annotation_input_metadata.append(
            {
                "completion_A_for_judge": truncated_completion_A,
                "completion_B_for_judge": truncated_completion_B,
                "completion_A_reasoning_stripped": stripped_A,
                "completion_B_reasoning_stripped": stripped_B,
                "completion_A_truncated_for_judge": truncated_A,
                "completion_B_truncated_for_judge": truncated_B,
            }
        )
    if truncated_completion_count:
        print(
            "Warning: truncated "
            f"{truncated_completion_count} judge inputs to "
            f"{truncate_input_chars} characters before evaluation."
        )
    inputs = prompt_template.batch(input_payloads)

    if judge_tokenizer is not None and max_judge_model_len:
        inputs = _preflight_shrink_to_judge_budget(
            prompt_template=prompt_template,
            inputs=inputs,
            input_payloads=input_payloads,
            annotation_input_metadata=annotation_input_metadata,
            case_ids=case_ids,
            judge_tokenizer=judge_tokenizer,
            max_judge_model_len=max_judge_model_len,
            max_out_tokens_judge=max_out_tokens_judge,
            limit_event_tracker=limit_event_tracker,
        )

    print(f"Start LLM judge annotation ({len(inputs)} annotations).")
    judge_completions = do_inference(
        chat_model=judge_chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        usage_tracker=usage_tracker,
        usage_phase=usage_phase,
        usage_model_spec=usage_model_spec,
    )

    annotations = []
    for (
        judge_input,
        judge_completion,
        instruction,
        completion_A,
        completion_B,
        annotation_input_metadata_row,
    ) in zip(
        inputs,
        judge_completions,
        instructions,
        completions_A,
        completions_B,
        annotation_input_metadata,
        strict=True,
    ):
        annotations.append(
            JudgeAnnotation(
                judge_input=judge_input,
                judge_completion=judge_completion,
                instruction=instruction,
                completion_A=completion_A,
                completion_B=completion_B,
                **annotation_input_metadata_row,
            )
        )
    return annotations


def judge_and_parse_prefs(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    case_ids: list[object] | None = None,
    swap_mode: str = "fixed",
    provide_explanation: bool = False,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    parser_mode: str = "score",
    strip_thinking_before_judging: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    truncate_input_chars: int = 8192,
    use_tqdm: bool = False,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    usage_model_spec: str | None = None,
    limit_event_tracker: LimitEventTracker | None = None,
    judge_tokenizer: "PreTrainedTokenizerBase | None" = None,
    max_judge_model_len: int | None = None,
    max_out_tokens_judge: int | None = None,
) -> tuple[list[JudgeAnnotation], list[JudgeAnnotation] | None, pd.Series]:
    """Run judge annotation and parse preferences, handling swap_mode='both'.

    Returns:
        annotations: original-order JudgeAnnotations
        annotations_reversed: reversed-order JudgeAnnotations (None if swap_mode != "both")
        prefs: pd.Series of floats (0=A wins, 0.5=tie, 1=B wins, None=unparseable),
               already combined for swap_mode="both"
    """
    if swap_mode == "both":
        print("Correction for judge bias towards a certain model position is set.")
        print(
            f"Evaluating completions with models reversed with judge {judge_chat_model}."
        )

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        case_ids=case_ids,
        provide_explanation=provide_explanation,
        prompt_preset=prompt_preset,
        strip_thinking_before_judging=strip_thinking_before_judging,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=use_tqdm,
        usage_tracker=usage_tracker,
        usage_phase=usage_phase,
        usage_model_spec=usage_model_spec,
        limit_event_tracker=limit_event_tracker,
        judge_tokenizer=judge_tokenizer,
        max_judge_model_len=max_judge_model_len,
        max_out_tokens_judge=max_out_tokens_judge,
    )

    annotations_reversed = None
    if swap_mode == "both":
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions,
            completions_A=completions_B,
            completions_B=completions_A,
            case_ids=case_ids,
            provide_explanation=provide_explanation,
            prompt_preset=prompt_preset,
            strip_thinking_before_judging=strip_thinking_before_judging,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=usage_model_spec,
            limit_event_tracker=limit_event_tracker,
            judge_tokenizer=judge_tokenizer,
            max_judge_model_len=max_judge_model_len,
            max_out_tokens_judge=max_out_tokens_judge,
        )

    def _none_to_nan(x):
        return float("nan") if x is None else x

    score_parser = PairScore(parser_mode=parser_mode)
    prefs = pd.Series(
        [score_parser.parse_model_raw(a.judge_completion) for a in annotations]
    )

    if swap_mode == "both":
        prefs = prefs.apply(_none_to_nan)
        prefs_reversed = pd.Series(
            [
                score_parser.parse_model_raw(a.judge_completion)
                for a in annotations_reversed
            ]
        ).apply(_none_to_nan)
        prefs = pd.concat([prefs, (1 - prefs_reversed)]).reset_index(drop=True)

    return annotations, annotations_reversed, prefs


_LC_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def _count_chat_tokens(prompt_value: Any, tokenizer: Any) -> int:
    """Count tokens the way vLLM's ``llm.chat()`` tokenizes after applying the
    tokenizer's chat template. Falls back to raw-string encoding for tokenizers
    without a chat template or if template application raises."""
    if hasattr(prompt_value, "to_messages"):
        messages = [
            {
                "role": _LC_ROLE_MAP.get(msg.type, msg.type),
                "content": msg.content,
            }
            for msg in prompt_value.to_messages()
        ]
        try:
            return len(tokenizer.apply_chat_template(messages, tokenize=True))
        except Exception:
            pass
    if hasattr(prompt_value, "to_string"):
        text = prompt_value.to_string()
    else:
        text = str(prompt_value)
    return len(tokenizer.encode(text))


def _find_token_overflows(
    inputs: list[Any], tokenizer: Any, safe_budget: int
) -> list[tuple[int, int]]:
    """Return ``(index, token_count)`` for inputs whose tokenized length exceeds
    ``safe_budget``."""
    overflows: list[tuple[int, int]] = []
    for idx, item in enumerate(inputs):
        token_count = _count_chat_tokens(item, tokenizer)
        if token_count > safe_budget:
            overflows.append((idx, token_count))
    return overflows


def _chars_per_token(text: str, tokenizer: Any) -> float:
    """Return a conservative char-to-token ratio for ``text``, floored at 1.0.

    Short/empty inputs yield a low ratio, which under-truncates rather than
    overflowing - the safe direction for the preflight shrink loop.
    """
    text = text if isinstance(text, str) else ""
    if not text:
        return 1.0
    token_count = max(1, len(tokenizer.encode(text)))
    return max(1.0, len(text) / token_count)


def _render_with_empty_completions(
    prompt_template: ChatPromptTemplate, user_prompt: str
) -> Any:
    """Render the prompt template with empty completions so the fixed template
    + user-prompt overhead can be measured per case. ``ChatPromptTemplate``
    uses ``str.format()`` on each message, so empty strings substitute cleanly
    for both completion slots."""
    return prompt_template.invoke(
        {
            "user_prompt": user_prompt,
            "completion_A": "",
            "completion_B": "",
        }
    )


def _preflight_shrink_to_judge_budget(
    *,
    prompt_template: ChatPromptTemplate,
    inputs: list[Any],
    input_payloads: list[dict[str, str]],
    annotation_input_metadata: list[dict[str, object]],
    case_ids: list[object],
    judge_tokenizer: Any,
    max_judge_model_len: int,
    max_out_tokens_judge: int | None,
    limit_event_tracker: LimitEventTracker | None,
) -> list[Any]:
    """Bounded shrink-and-re-render loop that converts judge-context overflows
    into soft ``judge_input_token_truncation`` limit events instead of a hard
    ``VLLMValidationError`` at request time.

    The per-completion budget subtracts the case-specific template + user-prompt
    overhead so that one iteration typically suffices; the 3-iteration bound is
    a genuine safety net for the rare pathological case where the char-to-token
    ratio shifts after truncation (e.g. dropping multi-byte glyphs).
    """
    safe_budget = (
        max_judge_model_len - (max_out_tokens_judge or 0) - _PREFLIGHT_RESERVED_TOKENS
    )
    for _ in range(_PREFLIGHT_MAX_ITERATIONS):
        overflows = _find_token_overflows(inputs, judge_tokenizer, safe_budget)
        if not overflows:
            return inputs
        for idx, _token_count in overflows:
            payload = input_payloads[idx]
            fixed_tokens = _count_chat_tokens(
                _render_with_empty_completions(prompt_template, payload["user_prompt"]),
                judge_tokenizer,
            )
            per_completion_budget = max(256, (safe_budget - fixed_tokens) // 2)
            ratio_A = _chars_per_token(payload["completion_A"], judge_tokenizer)
            ratio_B = _chars_per_token(payload["completion_B"], judge_tokenizer)
            new_cap_A = max(
                _PREFLIGHT_MIN_COMPLETION_CHARS,
                int(per_completion_budget * ratio_A * 0.9),
            )
            new_cap_B = max(
                _PREFLIGHT_MIN_COMPLETION_CHARS,
                int(per_completion_budget * ratio_B * 0.9),
            )
            payload["completion_A"], shrunk_A = truncate_with_metadata(
                payload["completion_A"],
                max_len=new_cap_A,
                tracker=limit_event_tracker,
                kind="judge_input_token_truncation",
                stage="judge_input",
                field="completion_A",
                case_id=case_ids[idx],
            )
            payload["completion_B"], shrunk_B = truncate_with_metadata(
                payload["completion_B"],
                max_len=new_cap_B,
                tracker=limit_event_tracker,
                kind="judge_input_token_truncation",
                stage="judge_input",
                field="completion_B",
                case_id=case_ids[idx],
            )
            metadata_row = annotation_input_metadata[idx]
            metadata_row["completion_A_for_judge"] = payload["completion_A"]
            metadata_row["completion_B_for_judge"] = payload["completion_B"]
            if shrunk_A:
                metadata_row["completion_A_truncated_for_judge"] = True
            if shrunk_B:
                metadata_row["completion_B_truncated_for_judge"] = True
        inputs = prompt_template.batch(input_payloads)

    final_overflows = _find_token_overflows(inputs, judge_tokenizer, safe_budget)
    for idx, token_count in final_overflows:
        if limit_event_tracker is not None:
            limit_event_tracker.record(
                "judge_input_token_truncation_failed",
                stage="judge_input",
                case_id=case_ids[idx],
                original_length=token_count,
                final_length=safe_budget,
                note=(
                    f"{_PREFLIGHT_MAX_ITERATIONS} shrink iterations did not "
                    f"bring tokens under {safe_budget}; falling through to "
                    "vLLM validation."
                ),
            )
    return inputs
