"""Judge sampled arena battles and produce canonical physical-battle rows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.evaluate import JudgeAnnotation, judge_and_parse_prefs

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.prompts.registry import ResolvedJudgePrompt


def serialize_judge_input(judge_input: object) -> str:
    if judge_input is None:
        return ""
    to_string = getattr(judge_input, "to_string", None)
    if callable(to_string):
        return to_string()
    return str(judge_input)


def preference_to_hard(preference: float | None) -> float:
    """Apply the shared pairwise hard-preference contract exactly."""
    if preference is None or pd.isna(preference):
        return float("nan")
    if preference < 0.5:
        return 0.0
    if preference > 0.5:
        return 1.0
    return 0.5


def validate_battle_conversations(df: pd.DataFrame) -> None:
    """Require matching user/assistant conversation pairs."""
    for _, battle in df.iterrows():
        battle_id = battle.get("battle_id", battle.get("question_id", "unknown"))
        prompts: list[str] = []
        for column in ("conversation_a", "conversation_b"):
            conversation = battle[column]
            if not isinstance(conversation, (list, tuple)) or len(conversation) < 2:
                raise ValueError(
                    f"Battle {battle_id!r} requires user and assistant turns in "
                    f"{column}."
                )
            user_turn, assistant_turn = conversation[:2]
            if (
                not isinstance(user_turn, dict)
                or not isinstance(assistant_turn, dict)
                or user_turn.get("role") != "user"
                or assistant_turn.get("role") != "assistant"
            ):
                raise ValueError(
                    f"Battle {battle_id!r} requires user and assistant turns in "
                    f"{column}."
                )
            try:
                prompts.append(extract_turn_text(user_turn))
                extract_turn_text(assistant_turn)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Battle {battle_id!r} has invalid turn content in {column}."
                ) from exc
        if prompts[0] != prompts[1]:
            raise ValueError(
                f"Battle {battle_id!r} has different user prompts across conversations."
            )


def _battle_texts(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    instructions = [extract_turn_text(conv[0]) for conv in df["conversation_a"]]
    completions_a = [extract_turn_text(conv[1]) for conv in df["conversation_a"]]
    completions_b = [extract_turn_text(conv[1]) for conv in df["conversation_b"]]
    return instructions, completions_a, completions_b


def _serialize_mapping(value: dict[str, object] | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _annotation_frame(
    original: pd.DataFrame,
    annotations: list[JudgeAnnotation],
    *,
    orientation: str,
    parser_name: str,
) -> pd.DataFrame:
    """Attach one judge pass to the arena's stored A/B identity."""
    reversed_pass = orientation == "reversed"
    rows = []
    for annotation, (_, battle) in zip(annotations, original.iterrows(), strict=True):
        parsed = annotation.parsed
        judge_input_preference = (
            float("nan") if parsed is None else float(parsed.preference)
        )
        preference = (
            1.0 - judge_input_preference
            if reversed_pass and parsed is not None
            else judge_input_preference
        )
        rows.append(
            {
                "battle_id": battle["battle_id"],
                "question_id": battle["question_id"],
                "model_a": battle["model_a"],
                "model_b": battle["model_b"],
                "winner": battle["winner"],
                "lang": battle["lang"],
                "instruction": annotation.instruction,
                "completion_a": (
                    annotation.completion_B
                    if reversed_pass
                    else annotation.completion_A
                ),
                "completion_b": (
                    annotation.completion_A
                    if reversed_pass
                    else annotation.completion_B
                ),
                "judge_input": serialize_judge_input(annotation.judge_input),
                "judge_completion": annotation.judge_completion,
                "judge_parser": parser_name,
                "judge_top_logprobs_json": _serialize_mapping(
                    annotation.judge_top_logprobs
                ),
                "parse_ok": parsed is not None,
                "pref_judge_input": judge_input_preference,
                "pref": preference,
                "orientation": orientation,
                "judge_input_model_a": (
                    battle["model_b"] if reversed_pass else battle["model_a"]
                ),
                "judge_input_model_b": (
                    battle["model_a"] if reversed_pass else battle["model_b"]
                ),
                "judge_input_completion_a": annotation.completion_A,
                "judge_input_completion_b": annotation.completion_B,
                "parsed_label": None if parsed is None else parsed.label,
                "parsed_scores_json": (
                    None if parsed is None else _serialize_mapping(parsed.scores)
                ),
                "parsed_details_json": (
                    None if parsed is None else _serialize_mapping(parsed.details)
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_battle_preferences(
    annotations: pd.DataFrame, *, swap_mode: str
) -> pd.DataFrame:
    """Combine canonical judge passes into one row per physical battle."""
    if swap_mode not in {"fixed", "both"}:
        raise ValueError(
            "Meta-evaluation aggregation supports fixed or both swap modes."
        )
    metadata_columns = (
        "battle_id",
        "question_id",
        "model_a",
        "model_b",
        "winner",
        "lang",
    )
    missing = sorted(set(metadata_columns) - set(annotations.columns))
    if missing:
        raise ValueError(f"Meta-evaluation annotations are missing columns: {missing}.")
    if annotations[list(metadata_columns)].isna().any().any():
        raise ValueError(
            "Meta-evaluation physical-battle metadata must not be missing."
        )

    expected_passes = 2 if swap_mode == "both" else 1
    expected_orientations = (
        {"direct", "reversed"} if swap_mode == "both" else {"single"}
    )
    rows = []
    for battle_id, passes in annotations.groupby("battle_id", sort=False):
        conflicting = [
            column
            for column in metadata_columns[1:]
            if passes[column].nunique(dropna=False) != 1
        ]
        if conflicting:
            raise ValueError(
                f"Battle {battle_id!r} has conflicting metadata: {conflicting}."
            )
        orientations = set(passes["orientation"])
        if len(passes) != expected_passes or orientations != expected_orientations:
            raise ValueError(
                f"Battle {battle_id!r} has {len(passes)} passes and orientations "
                f"{sorted(orientations)}; expected {sorted(expected_orientations)}."
            )
        valid = passes.loc[passes["parse_ok"] & passes["pref"].notna(), "pref"]
        parsed = len(valid)
        preference = float(valid.mean()) if parsed else float("nan")
        first = passes.iloc[0]
        rows.append(
            {
                "battle_id": battle_id,
                "question_id": first["question_id"],
                "model_a": first["model_a"],
                "model_b": first["model_b"],
                "winner": first["winner"],
                "lang": first["lang"],
                "parse_ok": parsed > 0,
                "pref": preference,
                "pref_hard": preference_to_hard(preference),
                "n_passes_expected": expected_passes,
                "n_passes_parsed": parsed,
                "parse_status": (
                    "complete"
                    if parsed == expected_passes
                    else "partial"
                    if parsed
                    else "missing"
                ),
            }
        )
    aggregated = pd.DataFrame(rows)
    if aggregated["battle_id"].duplicated().any():
        raise ValueError(
            "Aggregated meta-eval battles must have unique battle_id values."
        )
    return aggregated


def annotate_sample(
    df_sample: pd.DataFrame,
    cfg: RunConfig,
    *,
    judge_chat_model,
    resolved_prompt: ResolvedJudgePrompt,
) -> pd.DataFrame:
    if cfg.judge.swap_mode not in {"fixed", "both"}:
        raise ValueError(
            "Meta-evaluation annotation supports fixed or both swap modes."
        )
    parser = resolved_prompt.parser
    if parser is None:
        raise ValueError(
            f"Prompt preset {resolved_prompt.preset_name!r} has no judge parser."
        )

    df_sample = df_sample.copy()
    if "battle_id" not in df_sample:
        raise ValueError("annotate_sample requires stable battle_id values.")
    if df_sample["battle_id"].isna().any() or df_sample["battle_id"].duplicated().any():
        raise ValueError("annotate_sample requires unique, non-null battle_id values.")
    validate_battle_conversations(df_sample)

    instructions, completions_a, completions_b = _battle_texts(df_sample)
    annotations, reversed_annotations, _combined_preferences = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_a,
        completions_B=completions_b,
        swap_mode=cfg.judge.swap_mode,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        parse=parser,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
    )

    both = cfg.judge.swap_mode == "both"
    if len(annotations) != len(df_sample):
        raise ValueError(
            "Meta-evaluation judging returned an unexpected number of direct passes."
        )
    if both and (
        reversed_annotations is None or len(reversed_annotations) != len(df_sample)
    ):
        raise ValueError(
            "Meta-evaluation judging returned an unexpected number of reversed passes."
        )
    if not both and reversed_annotations is not None:
        raise ValueError("Fixed meta-evaluation judging returned reversed passes.")

    parts = [
        _annotation_frame(
            df_sample,
            annotations,
            orientation="direct" if both else "single",
            parser_name=parser.name,
        )
    ]
    if reversed_annotations is not None:
        parts.append(
            _annotation_frame(
                df_sample,
                reversed_annotations,
                orientation="reversed",
                parser_name=parser.name,
            )
        )
    return pd.concat(parts, ignore_index=True)
