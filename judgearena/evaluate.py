import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from scipy.optimize import minimize_scalar

from judgearena.log import get_logger
from judgearena.models import InferenceResult, do_inference
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    resolve_judge_prompt,
)
from judgearena.prompts.registry import (
    resolve_run_judge_prompt as _resolve_run_judge_prompt,
)
from judgearena.utils import strip_thinking_tags, truncate

logger = get_logger(__name__)

# Re-exported for existing callers; implementations live with the presets.
from judgearena.prompts.parsing import (  # noqa: E402
    JudgeParser,
    PairScore,
    resolve_judge_parser,
)


def calibrate_temperature(
    delta_s: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float] = (-10.0, 10.0),
) -> float:
    """Find the MLE temperature T* for the model P(A>B) = σ(T·Δs).

    The log-likelihood is:

        L(T) = Σ_i [ y_i·log σ(T·Δs_i) + (1−y_i)·log σ(−T·Δs_i) ]
               = Σ_i log σ(T · (2y_i − 1) · Δs_i)

    This is concave in T (single global maximum) so ``minimize_scalar`` with
    the 'bounded' method is guaranteed to converge.

    Args:
        delta_s: Score differences ``s_A − s_B`` for each battle, shape (N,).
        y: Observed hard labels (1 = A was preferred, 0 = B was preferred,
           0.5 = tie).  Ties contribute zero gradient and are skipped.
        bounds: Search interval for T (default −10 to +10).

    Returns:
        The calibrated temperature T*.
    """
    delta_s = np.asarray(delta_s, dtype=float)
    y = np.asarray(y, dtype=float)

    # Skip ties (y == 0.5) — they carry no directional information.
    non_tie = y != 0.5
    delta_s = delta_s[non_tie]
    y = y[non_tie]

    if len(delta_s) == 0:
        raise ValueError(
            "No non-tie observations available for temperature calibration."
        )

    # z_i = (2y_i − 1) · Δs_i  (positive when the score difference agrees with the outcome)
    z = (2 * y - 1) * delta_s

    def neg_log_likelihood(T: float) -> float:
        # log σ(T·z) = −log(1 + exp(−T·z)) = −logaddexp(0, −T·z)
        return float(np.sum(np.logaddexp(0.0, -T * z)))

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def load_judge_system_and_user_prompt(
    multi_turn: bool = False,
) -> tuple[str, str]:
    resolved = resolve_judge_prompt(
        preset=DEFAULT_JUDGE_PROMPT_PRESET,
        multi_turn=multi_turn,
    )
    return resolved.system_prompt or "", resolved.user_prompt_template


def resolve_judge_prompts(
    *,
    multi_turn: bool = False,
    prompt_preset: str | None = None,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    task: str | None = None,
    system_file: str | None = None,
    user_file: str | None = None,
    parser: str | None = None,
) -> ResolvedJudgePrompt:
    if system_prompt is not None and user_prompt_template is not None:
        return ResolvedJudgePrompt(
            preset_name=prompt_preset or "custom",
            parser=resolve_judge_parser(parser or "score"),
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            source="override",
        )
    if system_prompt is not None or user_prompt_template is not None:
        raise ValueError(
            "Both system_prompt and user_prompt_template must be provided together."
        )

    resolved = resolve_judge_prompt(
        task=task,
        preset=prompt_preset,
        system_file=system_file,
        user_file=user_file,
        multi_turn=multi_turn,
        parser=parser,
    )
    if resolved.delegated:
        raise ValueError(
            f"Judge prompt preset '{resolved.preset_name}' is delegated and cannot "
            "be used for generic pairwise judging."
        )
    return resolved


def resolve_run_judge_prompt(
    task: str | None,
    cli_args,
    *,
    multi_turn: bool = False,
) -> ResolvedJudgePrompt:
    return _resolve_run_judge_prompt(task, cli_args, multi_turn=multi_turn)


@dataclass
class JudgeAnnotation:
    instruction: str  # instruction from the user
    completion_A: str  # completion of the first model
    completion_B: str  # completion of the second model
    judge_completion: str  # output of the judge
    judge_input: str | None = None  # input that was passed to the judge
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET
    # first-token top logprobs, only collected for logprob-weighted presets
    judge_top_logprobs: dict[str, float] | None = None


def annotate_battles(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    system_prompt: str | None = None,
    user_prompt_template: str = None,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    strip_thinking_before_judging: bool = False,
    collect_top_logprobs: bool = False,
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
    :param judge_chat_model:
    :param instructions:
    :param completions_A:
    :param completions_B:
    :param system_prompt:
    :param user_prompt_template:
    :param truncate_input_chars: Max characters to truncate completions before sending to judge.
    :param use_tqdm:
    :return:
    """
    # alternatively pass list of tuples
    assert len(instructions) == len(completions_A) == len(completions_B)

    resolved_prompt = resolve_judge_prompts(
        prompt_preset=prompt_preset,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )

    message_templates: list[tuple[str, str]] = []
    if resolved_prompt.system_prompt is not None:
        message_templates.append(("system", resolved_prompt.system_prompt))
    message_templates.append(("user", resolved_prompt.user_prompt_template))
    prompt_template = ChatPromptTemplate.from_messages(message_templates)
    if strip_thinking_before_judging:
        completions_A = [strip_thinking_tags(c) for c in completions_A]
        completions_B = [strip_thinking_tags(c) for c in completions_B]

    prompt_inputs = []
    for user_prompt, completion_A, completion_B in zip(
        instructions, completions_A, completions_B, strict=True
    ):
        completion_A = truncate(completion_A, max_len=truncate_input_chars)
        completion_B = truncate(completion_B, max_len=truncate_input_chars)
        prompt_inputs.append(
            {
                "user_prompt": user_prompt,
                "completion_A": completion_A,
                "completion_B": completion_B,
                "user_prompt_json": json.dumps(user_prompt, ensure_ascii=False),
                "completion_A_json": json.dumps(completion_A, ensure_ascii=False),
                "completion_B_json": json.dumps(completion_B, ensure_ascii=False),
            }
        )
    inputs = prompt_template.batch(prompt_inputs)

    logger.info("Start LLM judge annotation (%d annotations).", len(inputs))
    judge_results = do_inference(
        chat_model=judge_chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        return_top_logprobs=collect_top_logprobs,
        stage="judging",
    )
    if not collect_top_logprobs:
        judge_results = [InferenceResult(text=text) for text in judge_results]

    annotations = []
    for judge_input, judge_result, instruction, completion_A, completion_B in zip(
        inputs,
        judge_results,
        instructions,
        completions_A,
        completions_B,
        strict=True,
    ):
        annotations.append(
            JudgeAnnotation(
                judge_input=judge_input,
                judge_completion=judge_result.text,
                instruction=instruction,
                completion_A=completion_A,
                completion_B=completion_B,
                prompt_preset=resolved_prompt.preset_name,
                judge_top_logprobs=judge_result.first_token_top_logprobs,
            )
        )
    return annotations


def combine_swapped_prefs(prefs_ab: pd.Series, prefs_ba: pd.Series) -> pd.Series:
    """Combine swap_mode='both' prefs into one P(B wins) series: [pref_AB, 1 - pref_BA].

    ``prefs_ab`` are P(B wins) from the AB ordering; ``prefs_ba`` are P(B wins)
    from the swapped BA ordering, so ``1 - prefs_ba`` re-orients them to the AB
    frame before stacking.
    """
    return pd.concat(
        [prefs_ab.reset_index(drop=True), 1 - prefs_ba.reset_index(drop=True)]
    ).reset_index(drop=True)


def judge_and_parse_prefs(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    swap_mode: str = "fixed",
    strip_thinking_before_judging: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    truncate_input_chars: int = 8192,
    use_tqdm: bool = False,
    parse: JudgeParser | None = None,
) -> tuple[list[JudgeAnnotation], list[JudgeAnnotation] | None, pd.Series]:
    """Run judge annotation and parse preferences, handling swap_mode='both'.

    Returns:
        annotations: original-order JudgeAnnotations
        annotations_reversed: reversed-order JudgeAnnotations (None if swap_mode != "both")
        prefs: pd.Series of floats (0=A wins, 0.5=tie, 1=B wins, None=unparseable),
               already combined for swap_mode="both"
    """
    if parse is None:
        parse = PairScore()

    if swap_mode == "both":
        logger.info(
            "Correction for judge bias towards a certain model position is set."
        )
        logger.info(
            "Evaluating completions with models reversed with judge %s.",
            judge_chat_model,
        )

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        strip_thinking_before_judging=strip_thinking_before_judging,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        prompt_preset=prompt_preset,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=use_tqdm,
        collect_top_logprobs=parse.requires_top_logprobs,
    )

    annotations_reversed = None
    if swap_mode == "both":
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions,
            completions_A=completions_B,
            completions_B=completions_A,
            strip_thinking_before_judging=strip_thinking_before_judging,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            prompt_preset=prompt_preset,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
            collect_top_logprobs=parse.requires_top_logprobs,
        )

    def _none_to_nan(x):
        return float("nan") if x is None else x

    def _parse_and_warn(ann_list: list, label: str) -> pd.Series:
        if parse.requires_top_logprobs:
            n_no_logprobs = sum(1 for a in ann_list if a.judge_top_logprobs is None)
            if n_no_logprobs:
                logger.warning(
                    "%d/%d judge responses returned no logprobs (%s) — falling "
                    "back to discrete token parsing for those.",
                    n_no_logprobs,
                    len(ann_list),
                    label,
                )
        results = [
            parse(a.judge_completion, top_logprobs=a.judge_top_logprobs)
            for a in ann_list
        ]
        n_failed = sum(1 for r in results if r is None)
        if n_failed:
            logger.warning(
                "%d/%d judge outputs could not be parsed (%s) — those battles are dropped from stats.",
                n_failed,
                len(results),
                label,
            )
        return pd.Series(results)

    prefs = _parse_and_warn(annotations, "direct")

    if swap_mode == "both":
        prefs_reversed = _parse_and_warn(annotations_reversed, "reversed").apply(
            _none_to_nan
        )
        prefs = combine_swapped_prefs(prefs.apply(_none_to_nan), prefs_reversed)

    return annotations, annotations_reversed, prefs
