import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate

from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.repro import _to_jsonable, write_run_metadata
from judgearena.utils import (
    compute_pref_summary,
    data_root,
    do_inference,
    download_hf,
    read_df,
    truncate,
)

from typing import Any


class PairScore:
    def __init__(self):
        super(PairScore).__init__()
        self.temperature = 0.3

    def preference_from_scores(self, score_a: float, score_b: float) -> float:
        return 1 - np.exp(self.temperature * score_a) / (
            np.exp(self.temperature * np.array([score_a, score_b])).sum()
        )

    def parse_model_raw(self, judge_completion: str) -> float | None:
        # lower case to avoid confusion, e.g. when "a" is used instead of "A"
        score_a = self.get_regexp_match(
            judge_completion.lower(), r'score.*?a[": *\n]*(-?\d+)'
        )
        score_b = self.get_regexp_match(
            judge_completion.lower(), r'score.*?b[": *\n]*(-?\d+)'
        )
        if score_a is None or score_b is None:
            return None
        else:
            return float(self.preference_from_scores(score_a, score_b))

    def get_regexp_match(self, s: str, regex: str, group_index: int = 1):
        m = re.search(re.compile(regex), s)
        if m is None:
            return None
        else:
            return float(m.group(group_index).strip(" "))


_COMPLETION_LABEL_SINGLE = "Answer"
_COMPLETION_LABEL_MULTI_TURN = "Conversation with User"
_EXPLANATION_SUFFIX = ", first starts with an explanation of your judgement"
_SCORE_FENCE = "\n```"


def load_judge_system_and_user_prompt(
    provide_explanation: bool = True,
    multi_turn: bool = False,
) -> tuple[str, str]:
    prompts_dir = Path(__file__).parent / "prompts"
    system_prompt = (prompts_dir / "system-prompt.txt").read_text()

    prompt_filename = (
        "prompt-with-explanation.txt" if provide_explanation else "prompt.txt"
    )
    user_prompt_template = (prompts_dir / prompt_filename).read_text()
    user_prompt_template = user_prompt_template.replace(
        "{completion_label}",
        _COMPLETION_LABEL_MULTI_TURN if multi_turn else _COMPLETION_LABEL_SINGLE,
    )
    user_prompt_template = user_prompt_template.replace(
        "{explanation_suffix}",
        _EXPLANATION_SUFFIX if provide_explanation else _SCORE_FENCE,
    )

    return system_prompt, user_prompt_template


def resolve_judge_prompts(
    *,
    provide_explanation: bool,
    multi_turn: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> tuple[str, str]:
    default_system_prompt, default_user_prompt_template = (
        load_judge_system_and_user_prompt(
            provide_explanation=provide_explanation, multi_turn=multi_turn
        )
    )
    return (
        system_prompt if system_prompt is not None else default_system_prompt,
        (
            user_prompt_template
            if user_prompt_template is not None
            else default_user_prompt_template
        ),
    )


# ---------------------------------------------------------------------------
# Reusable evaluation-result helpers
# ---------------------------------------------------------------------------


def build_annotation_dataframe(
    annotations: list["JudgeAnnotation"],
    annotations_reversed: list["JudgeAnnotation"] | None,
    instruction_indices: list[int],
    model_A: str,
    model_B: str,
    judge_model: str,
) -> pd.DataFrame:
    """Build a single DataFrame from forward (and optionally reversed) annotations.

    When *annotations_reversed* is provided (swap_mode="both"), both sets are
    concatenated with the model columns swapped for the reversed batch.
    """
    df = pd.DataFrame(annotations)
    df["instruction_index"] = instruction_indices
    df["model_A"] = model_A
    df["model_B"] = model_B
    df["judge"] = judge_model

    if annotations_reversed is not None:
        df_rev = pd.DataFrame(annotations_reversed)
        df_rev["instruction_index"] = instruction_indices
        df_rev["model_A"] = model_B
        df_rev["model_B"] = model_A
        df_rev["judge"] = judge_model
        df = pd.concat([df, df_rev])

    return df


@dataclass
class EvaluationResult:
    """Pure-data container for an evaluation run's outputs."""

    annotations_df: pd.DataFrame
    prefs: pd.Series
    summary: dict[str, Any]
    run_config: dict[str, Any]

    @property
    def results(self) -> dict[str, Any]:
        """Merged dict of *run_config* + *summary* + raw preferences list."""
        return {**self.run_config, **self.summary, "preferences": self.prefs.tolist()}


def build_evaluation_result(
    *,
    annotations_df: pd.DataFrame,
    prefs: pd.Series,
    run_config: dict[str, Any],
) -> EvaluationResult:
    """Compute the preference summary and package everything into an
    :class:`EvaluationResult`.  Pure logic — no disk access."""
    summary = compute_pref_summary(prefs)
    return EvaluationResult(
        annotations_df=annotations_df,
        prefs=prefs,
        summary=summary,
        run_config=run_config,
    )


def save_evaluation_result(
    evaluation_result: EvaluationResult,
    *,
    output_dir: Path,
    entrypoint: str,
    judge_system_prompt: str,
    judge_user_prompt_template: str,
    input_payloads: dict[str, Any],
    started_at_utc: datetime,
    annotations_filename: str = "annotations.csv",
    results_filename: str = "results.json",
) -> None:
    """Write an :class:`EvaluationResult` to disk (CSV + JSON + run metadata).

    Run-metadata writing is best-effort: an ``OSError`` is caught and logged.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluation_result.results

    # Save annotations
    evaluation_result.annotations_df.to_csv(
        output_dir / annotations_filename, index=False
    )

    # Save results JSON
    with open(output_dir / results_filename, "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)

    # Write reproducibility metadata (best-effort)
    try:
        write_run_metadata(
            output_dir=output_dir,
            entrypoint=entrypoint,
            run=evaluation_result.run_config,
            results=results,
            input_payloads=input_payloads,
            judge_system_prompt=judge_system_prompt,
            judge_user_prompt_template=judge_user_prompt_template,
            started_at_utc=started_at_utc,
        )
    except OSError as e:
        print(f"Warning: failed to write run metadata: {e}")


def evaluate_completions(
    dataset: str = "alpaca-eval",
    judge_chat_model: LLM = None,
    method_A: str = "gpt4_1106_preview",
    method_B: str = "llama-2-70b-chat-hf",
    num_annotations: int | None = 50,
    use_tqdm: bool = False,
    truncate_input_chars: int | None = 8192,
    provide_explanation: bool = False,
    swap_mode: str = "fixed",
):
    """Evaluate two completion methods head-to-head with an LLM judge.

    :param dataset: Name of an evaluation dataset (e.g. ``"alpaca-eval"``).
    :param judge_chat_model: A LangChain chat model used as judge.  When
        *None*, defaults to ``Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")``.
    :param method_A: Model/method name or local path to completions CSV.
    :param method_B: Another model/method to compare against *method_A*.
    :param num_annotations: Cap the number of instructions evaluated.
    :param use_tqdm: Show a progress bar (may not work with all providers).
    :param truncate_input_chars: Truncate completions to this many characters
        before sending to the judge.
    :param provide_explanation: Ask the judge to provide an explanation.
    :param swap_mode: ``"fixed"`` (default) or ``"both"`` to also evaluate
        with A/B swapped and average the preferences.
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

    unique_string = dataset + "-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = data_root / "judge-evals" / unique_string
    print(f"Saving results in {output_folder}")

    (
        judge_system_prompt,
        judge_user_prompt_template,
    ) = resolve_judge_prompts(provide_explanation=provide_explanation)

    annotations, annotations_reversed, prefs = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instructions.tolist(),
        completions_A=completions_A.loc[instructions.index].tolist(),
        completions_B=completions_B.loc[instructions.index].tolist(),
        swap_mode=swap_mode,
        provide_explanation=provide_explanation,
        system_prompt=judge_system_prompt,
        user_prompt_template=judge_user_prompt_template,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=use_tqdm,
    )

    annotations_df = build_annotation_dataframe(
        annotations=annotations,
        annotations_reversed=annotations_reversed,
        instruction_indices=instructions.index.tolist(),
        model_A=method_A,
        model_B=method_B,
        judge_model=str(judge_chat_model),
    )

    run_config = {
        "dataset": dataset,
        "method_A": method_A,
        "method_B": method_B,
        "num_annotations": num_annotations,
        "n_annotations": len(instructions),
        "use_tqdm": use_tqdm,
        "truncate_input_chars": truncate_input_chars,
        "provide_explanation": provide_explanation,
        "swap_mode": swap_mode,
    }

    eval_result = build_evaluation_result(
        annotations_df=annotations_df,
        prefs=prefs,
        run_config=run_config,
    )

    save_evaluation_result(
        eval_result,
        output_dir=output_folder,
        entrypoint="judgearena.evaluate.evaluate_completions",
        judge_system_prompt=judge_system_prompt,
        judge_user_prompt_template=judge_user_prompt_template,
        input_payloads={
            "instruction_index": instructions.index.tolist(),
            "instructions": instructions.tolist(),
            "completions_A": completions_A.loc[instructions.index].tolist(),
            "completions_B": completions_B.loc[instructions.index].tolist(),
        },
        started_at_utc=run_started_at,
    )

    print(f"{method_A} against {method_B}:\n{eval_result.results}")


@dataclass
class JudgeAnnotation:
    instruction: str  # instruction from the user
    completion_A: str  # completion of the first model
    completion_B: str  # completion of the second model
    judge_completion: str  # output of the judge
    judge_input: str | None = None  # input that was passed to the judge


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

    system_prompt, user_prompt_template = resolve_judge_prompts(
        provide_explanation=provide_explanation,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt_template)]
    )

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
    print(f"Start LLM judge annotation ({len(inputs)} annotations).")
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
            )
        )
    return annotations


def judge_and_parse_prefs(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    swap_mode: str = "fixed",
    provide_explanation: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    truncate_input_chars: int = 8192,
    use_tqdm: bool = False,
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
        provide_explanation=provide_explanation,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
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
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
        )

    def _none_to_nan(x):
        return float("nan") if x is None else x

    score_parser = PairScore()
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
