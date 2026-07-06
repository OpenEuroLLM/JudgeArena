import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from scipy.optimize import minimize_scalar

from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.log import get_logger
from judgearena.models import do_inference
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    resolve_judge_prompt,
)
from judgearena.prompts.registry import (
    resolve_run_judge_prompt as _resolve_run_judge_prompt,
)
from judgearena.repro import _to_jsonable, write_run_metadata
from judgearena.utils import (
    compute_pref_summary,
    data_root,
    download_hf,
    read_df,
    strip_thinking_tags,
    truncate,
)

logger = get_logger(__name__)


class PairScore:
    def __init__(self, *, temperature: float = 0.3, parser_mode: str = "score"):
        super(PairScore).__init__()
        self.temperature = temperature
        self.parser_mode = parser_mode

    def preference_from_scores(self, score_a: float, score_b: float) -> float:
        return 1 - np.exp(self.temperature * score_a) / (
            np.exp(self.temperature * np.array([score_a, score_b])).sum()
        )

    def parse_model_raw(self, judge_completion: str) -> float | None:
        if self.parser_mode != "score":
            raise ValueError(f"Unsupported parser_mode '{self.parser_mode}'.")
        score_a, score_b = self.parse_raw_scores(judge_completion)
        if score_a is None or score_b is None:
            return None
        return float(self.preference_from_scores(score_a, score_b))

    @staticmethod
    def parse_raw_scores(
        judge_completion: str,
    ) -> tuple[float | None, float | None]:
        """Extract the raw A and B scores from a judge completion (no temperature)."""
        # Strip thinking-model <think> blocks, then lower-case to avoid confusion
        # (e.g. when "a" is used instead of "A").
        text = strip_thinking_tags(judge_completion).lower()
        score_a = PairScore.get_regexp_match(text, r'score.*?a[": *\n]*(-?\d+)')
        score_b = PairScore.get_regexp_match(text, r'score.*?b[": *\n]*(-?\d+)')
        return score_a, score_b

    @staticmethod
    def get_regexp_match(s: str, regex: str, group_index: int = 1):
        m = re.search(re.compile(regex), s)
        if m is None:
            return None
        else:
            return float(m.group(group_index).strip(" "))


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
    provide_explanation: bool = True,
    multi_turn: bool = False,
) -> tuple[str, str]:
    resolved = resolve_judge_prompt(
        preset=DEFAULT_JUDGE_PROMPT_PRESET,
        provide_explanation=provide_explanation,
        multi_turn=multi_turn,
    )
    return resolved.system_prompt or "", resolved.user_prompt_template


def resolve_judge_prompts(
    *,
    provide_explanation: bool = False,
    multi_turn: bool = False,
    prompt_preset: str | None = None,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    task: str | None = None,
    system_file: str | None = None,
    user_file: str | None = None,
) -> ResolvedJudgePrompt:
    if system_prompt is not None and user_prompt_template is not None:
        return ResolvedJudgePrompt(
            preset_name=prompt_preset or "custom",
            parser_mode="score",
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
        provide_explanation=provide_explanation,
        multi_turn=multi_turn,
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

    # A bit ugly, only loads if local path exist as we do not have a local path of completion for cases such as
    # m-arena-hard.
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
            logger.info("Path %s exists, loading local model completions.", method)
            df = read_df(Path(method)).set_index("instruction_index").sort_index()
            logger.info("Loaded %d completions.", len(df))
            df.loc[:, "output"] = df.loc[:, "output"].fillna("")
            return df.loc[:, "output"]
        else:
            logger.info("Loading %s from %s dataset.", method, dataset)
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

    unique_string = dataset + "-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = data_root / "judge-evals" / unique_string
    logger.info("Saving results in %s", output_folder)
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
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        use_tqdm=use_tqdm,
        truncate_input_chars=truncate_input_chars,
        provide_explanation=provide_explanation,
        strip_thinking_before_judging=strip_thinking_before_judging,
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
        **compute_pref_summary(prefs).to_dict(),
        **resolved_prompt.metadata(),
    }
    pd.DataFrame(annotations).to_csv(output_folder / "annotations.csv", index=False)

    logger.info("%s against %s:\n%s", method_A, method_B, results)
    with open(output_folder / "results.json", "w") as f:
        json.dump(_to_jsonable(results), f, allow_nan=False)

    run_metadata = {
        "dataset": dataset,
        "method_A": method_A,
        "method_B": method_B,
        "num_annotations": num_annotations,
        "n_annotations": len(instructions),
        "use_tqdm": use_tqdm,
        "truncate_input_chars": truncate_input_chars,
        "provide_explanation": provide_explanation,
        **resolved_prompt.metadata(),
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
        )
    except OSError as e:
        logger.warning("Failed to write run metadata: %s", e)


@dataclass
class JudgeAnnotation:
    instruction: str  # instruction from the user
    completion_A: str  # completion of the first model
    completion_B: str  # completion of the second model
    judge_completion: str  # output of the judge
    judge_input: str | None = None  # input that was passed to the judge
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET


def annotate_battles(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    system_prompt: str | None = None,
    user_prompt_template: str = None,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
    provide_explanation: bool = False,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    strip_thinking_before_judging: bool = False,
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
    :return:
    """
    # alternatively pass list of tuples
    assert len(instructions) == len(completions_A) == len(completions_B)

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
    if strip_thinking_before_judging:
        completions_A = [strip_thinking_tags(c) for c in completions_A]
        completions_B = [strip_thinking_tags(c) for c in completions_B]

    inputs = prompt_template.batch(
        [
            {
                "user_prompt": user_prompt,
                "completion_A": truncate(completion_A, max_len=truncate_input_chars),
                "completion_B": truncate(completion_B, max_len=truncate_input_chars),
            }
            for user_prompt, completion_A, completion_B in zip(
                instructions, completions_A, completions_B, strict=True
            )
        ]
    )

    logger.info("Start LLM judge annotation (%d annotations).", len(inputs))
    judge_completions = do_inference(
        chat_model=judge_chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
    )

    annotations = []
    for judge_input, judge_completion, instruction, completion_A, completion_B in zip(
        inputs,
        judge_completions,
        instructions,
        completions_A,
        completions_B,
        strict=True,
    ):
        annotations.append(
            JudgeAnnotation(
                judge_input=judge_input,
                judge_completion=judge_completion,
                instruction=instruction,
                completion_A=completion_A,
                completion_B=completion_B,
                prompt_preset=resolved_prompt.preset_name,
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
    provide_explanation: bool = False,
    strip_thinking_before_judging: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    parser_mode: str = "score",
    truncate_input_chars: int = 8192,
    use_tqdm: bool = False,
    score_parser: "PairScore | None" = None,
) -> tuple[list[JudgeAnnotation], list[JudgeAnnotation] | None, pd.Series]:
    """Run judge annotation and parse preferences, handling swap_mode='both'.

    Returns:
        annotations: original-order JudgeAnnotations
        annotations_reversed: reversed-order JudgeAnnotations (None if swap_mode != "both")
        prefs: pd.Series of floats (0=A wins, 0.5=tie, 1=B wins, None=unparseable),
               already combined for swap_mode="both"
    """
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
        provide_explanation=provide_explanation,
        strip_thinking_before_judging=strip_thinking_before_judging,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        prompt_preset=prompt_preset,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=use_tqdm,
    )

    annotations_reversed = None
    if swap_mode == "both":
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions,
            completions_A=completions_B,
            completions_B=completions_A,
            provide_explanation=provide_explanation,
            strip_thinking_before_judging=strip_thinking_before_judging,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            prompt_preset=prompt_preset,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
        )

    def _none_to_nan(x):
        return float("nan") if x is None else x

    if score_parser is None:
        score_parser = PairScore(parser_mode=parser_mode)

    def _parse_and_warn(ann_list: list, label: str) -> pd.Series:
        results = [score_parser.parse_model_raw(a.judge_completion) for a in ann_list]
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
