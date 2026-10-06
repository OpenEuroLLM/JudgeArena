"""Create the fixed benchmark inputs used by a leaderboard version.

Fit reference ratings from human battles, select questions and opponent answers,
and save these with the evaluation config and judge prompts. These files stay
unchanged when candidates are evaluated; candidate results are saved separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.benchmarks.arena import resolve_task_languages
from judgearena.benchmarks.elo.calibration import calibrate_frozen_temperature
from judgearena.benchmarks.elo.leaderboard import (
    AnchorSet,
    comparable_config,
    protocol_identifier,
    rebuild_leaderboard,
)
from judgearena.benchmarks.elo.rating import fit_bradley_terry, winner_to_pref
from judgearena.config import JudgePromptSpec, RunConfig, dump_config, load_config
from judgearena.datasets import load_battles
from judgearena.evaluate import resolve_run_judge_prompt
from judgearena.prompts.registry import ResolvedJudgePrompt
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import EloProtocol, ResolvedTaskSpec


def _connected_models(battles: pd.DataFrame, baseline_model: str) -> set[str]:
    """Find models linked to the baseline through human comparisons."""
    adjacency: dict[str, set[str]] = {}
    for model_a, model_b in battles.loc[:, ["model_a", "model_b"]].itertuples(
        index=False, name=None
    ):
        model_a, model_b = str(model_a), str(model_b)
        adjacency.setdefault(model_a, set())
        adjacency.setdefault(model_b, set())
        if model_a == model_b:
            continue
        adjacency[model_a].add(model_b)
        adjacency[model_b].add(model_a)

    if baseline_model not in adjacency:
        return set()
    connected: set[str] = set()
    pending = [baseline_model]
    while pending:
        model = pending.pop()
        if model in connected:
            continue
        connected.add(model)
        pending.extend(adjacency.get(model, ()) - connected)
    return connected


def _prepare_freeze_config(
    config_path: RunConfig | str | Path,
    anchor_models: list[str],
    languages: list[str],
    battles_per_language: int,
) -> tuple[RunConfig, ResolvedTaskSpec, EloProtocol, str, list[str]]:
    """Resolve creation settings, validate references, and remove per-run sampling limits."""
    if battles_per_language <= 0:
        raise ValueError("battles_per_language must be positive.")
    if not languages or len(languages) != len(set(languages)):
        raise ValueError("languages must be a non-empty list without duplicates.")

    cfg = (
        config_path.model_copy(deep=True)
        if isinstance(config_path, RunConfig)
        else load_config(config_path)
    )
    task = getattr(cfg, "resolved_task", None) or get_packaged_task(cfg.task)
    if task is None or not isinstance(task.spec.protocol, EloProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define an ELO protocol.")
    protocol_spec = task.spec.protocol
    assert cfg.elo is not None
    cfg.elo = cfg.elo.resolve(protocol_spec.scoring)
    baseline_model = cfg.elo.baseline_model
    if not baseline_model:
        raise ValueError("The resolved ELO config must define baseline_model.")
    cfg.generation.n_instructions = None
    cfg.elo = cfg.elo.model_copy(
        update={
            "n_instructions_per_language": None,
            "elo_random_battles": None,
            "leaderboard_dir": None,
        }
    )

    selected_languages = resolve_task_languages(task, languages, setting="--languages")
    if selected_languages != languages:
        missing = [lang for lang in languages if lang not in selected_languages]
        raise ValueError(
            f"Languages {missing} are not part of task variant {task.task!r}."
        )
    cfg.elo = cfg.elo.model_copy(update={"languages": list(languages)})

    anchors = list(dict.fromkeys([baseline_model, *anchor_models]))
    if any(not model for model in anchors):
        raise ValueError("anchor models must be non-empty strings.")
    return cfg, task, protocol_spec, baseline_model, anchors


def _fit_language_anchors(
    battles: pd.DataFrame,
    language: str,
    anchors: list[str],
    baseline_model: str,
    min_anchor_battles: int,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, int], int]:
    """Fit reference ratings from one language's baseline-connected human battle graph.

    Exclude unlabelled and self-comparison rows; reject malformed model names.
    The fit includes connected non-reference models too. Return eligible language
    rows for panel selection, reference ratings and counts, and the fitted row count.
    """
    language_battles = battles.loc[battles["lang"] == language].copy()
    language_battles["pref"] = language_battles["winner"].map(winner_to_pref)
    language_battles = language_battles.dropna(subset=["pref"])
    for column in ("model_a", "model_b"):
        if any(
            not isinstance(model, str) or not model.strip()
            for model in language_battles[column]
        ):
            raise ValueError(
                f"Language {language!r}: {column} must contain non-empty model names."
            )
    language_battles = language_battles.loc[
        language_battles["model_a"].ne(language_battles["model_b"])
    ].reset_index(drop=True)

    connected = _connected_models(language_battles, baseline_model)
    disconnected = [model for model in anchors if model not in connected]
    if disconnected:
        raise ValueError(
            f"Language {language!r}: anchors are not connected to baseline "
            f"{baseline_model!r}: {disconnected}."
        )

    fit_battles = language_battles.loc[
        language_battles["model_a"].isin(connected)
        & language_battles["model_b"].isin(connected)
    ]
    counts = {
        model: int(
            (
                (fit_battles["model_a"] == model) | (fit_battles["model_b"] == model)
            ).sum()
        )
        for model in anchors
    }
    insufficient = {
        model: count for model, count in counts.items() if count < min_anchor_battles
    }
    if insufficient:
        raise ValueError(
            f"Language {language!r}: anchors need at least {min_anchor_battles} "
            f"usable human battles each; observed counts: {insufficient}."
        )
    fitted = fit_bradley_terry(
        fit_battles,
        baseline_model=baseline_model,
        baseline_rating=1000,
        scale=400,
        base=10,
    )
    ratings = {model: float(fitted[model]) for model in anchors}
    return language_battles, ratings, counts, len(fit_battles)


def _select_language_panel(
    language_battles: pd.DataFrame,
    language: str,
    anchors: list[str],
    battles_per_language: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Select a deterministic panel with roughly balanced anchor opponents."""
    candidates = []
    ordered = language_battles.assign(
        _question_key=language_battles["question_id"].astype(str)
    ).sort_values(["_question_key", "model_a", "model_b"], kind="stable")
    for row_index, row in ordered.iterrows():
        question_id = str(row["question_id"])
        if (
            not question_id.strip()
            or not extract_turn_text(row["conversation_a"][0]).strip()
        ):
            continue
        for side in ("a", "b"):
            model = row[f"model_{side}"]
            completion = extract_turn_text(row[f"conversation_{side}"][1])
            if model in anchors and completion.strip():
                candidates.append(
                    {
                        "question_id": question_id,
                        "row_index": row_index,
                        "source_position": side,
                        "opponent_model": model,
                    }
                )

    candidates = pd.DataFrame(candidates)
    if candidates.empty:
        raise ValueError(f"Language {language!r}: found no anchor questions.")
    unique_questions = candidates["question_id"].nunique()
    if unique_questions < battles_per_language:
        raise ValueError(
            f"Language {language!r}: found {unique_questions} distinct "
            f"anchor questions, need {battles_per_language}."
        )

    candidates = candidates.iloc[rng.permutation(len(candidates))].copy()
    candidates["_round"] = candidates.groupby("opponent_model", sort=False).cumcount()
    selected = (
        candidates.sort_values("_round", kind="stable")
        .drop_duplicates("question_id")
        .head(battles_per_language)
    )
    positions = rng.permutation(
        (["A", "B"] * ((battles_per_language + 1) // 2))[:battles_per_language]
    )
    panel_rows = []
    for offset, selected_row in enumerate(selected.itertuples(index=False)):
        row = language_battles.loc[selected_row.row_index]
        completion = row[f"conversation_{selected_row.source_position}"][1]
        panel_rows.append(
            {
                "panel_id": f"{language}-{offset:06d}",
                "question_id": selected_row.question_id,
                "lang": language,
                "instruction": extract_turn_text(row["conversation_a"][0]),
                "opponent_model": selected_row.opponent_model,
                "opponent_completion": extract_turn_text(completion),
                "candidate_position": str(positions[offset]),
            }
        )
    return panel_rows


def _write_artifacts(
    output: str | Path,
    cfg: RunConfig,
    panel: pd.DataFrame,
    anchors: AnchorSet,
    resolved_prompt: ResolvedJudgePrompt,
) -> Path:
    """Write fixed benchmark inputs and an initial reference-only leaderboard index."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=False)
    system_name = "judge-system-prompt.txt"
    user_name = "judge-user-prompt.txt"
    (output_path / system_name).write_text(
        resolved_prompt.system_prompt or "", encoding="utf-8"
    )
    (output_path / user_name).write_text(
        resolved_prompt.user_prompt_template, encoding="utf-8"
    )
    cfg.judge = cfg.judge.model_copy(
        update={
            "prompt_preset": None,
            "prompt": JudgePromptSpec(
                system_file=Path(system_name),
                user_file=Path(user_name),
                parser=resolved_prompt.parser.name,
            ),
        }
    )
    config_path = output_path / "config.yaml"
    dump_config(cfg, config_path)
    frozen_config = load_config(config_path)
    frozen = anchors.model_copy(
        update={
            "protocol_id": protocol_identifier(
                anchors, panel, comparable_config(frozen_config)
            )
        }
    )
    panel.to_parquet(output_path / "panel.parquet", index=False)
    frozen.save(output_path / "anchors.json")
    (output_path / "entries").mkdir()
    rebuild_leaderboard(output_path, frozen)
    return output_path


def freeze_leaderboard(
    config_path: RunConfig | str | Path,
    output: str | Path,
    anchor_models: list[str],
    languages: list[str],
    battles_per_language: int,
    *,
    name: str | None = None,
    version: str = "0.01",
    min_anchor_battles: int = 1,
) -> Path:
    """Create and save a benchmark version for subsequent candidate evaluations.

    Fit human reference ratings, select the opponent panel, and optionally
    calibrate the judge before saving the inputs. No candidate answers are
    generated here.
    """
    if Path(output).exists():
        raise FileExistsError(f"Frozen leaderboard output already exists: {output}")
    if min_anchor_battles < 1:
        raise ValueError("min_anchor_battles must be positive.")
    if not version.strip():
        raise ValueError("version must be non-empty.")
    cfg, task, protocol_spec, baseline_model, anchors = _prepare_freeze_config(
        config_path, anchor_models, languages, battles_per_language
    )
    battles = load_battles(task)

    rng = np.random.default_rng(cfg.run.seed)
    ratings_by_language: dict[str, dict[str, float]] = {}
    counts_by_language: dict[str, dict[str, int]] = {}
    human_battles_by_language: dict[str, int] = {}
    panel_rows: list[dict[str, object]] = []
    for language in languages:
        language_battles, ratings, counts, human_battles = _fit_language_anchors(
            battles, language, anchors, baseline_model, min_anchor_battles
        )
        ratings_by_language[language] = ratings
        counts_by_language[language] = counts
        human_battles_by_language[language] = human_battles
        panel_rows.extend(
            _select_language_panel(
                language_battles,
                language,
                anchors,
                battles_per_language,
                rng,
            )
        )

    panel = pd.DataFrame(panel_rows)
    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
    if resolved_prompt.parser is None:
        raise ValueError("Frozen Elo judging requires a registered prompt parser.")
    calibrate_frozen_temperature(
        cfg, battles, anchors, languages, resolved_prompt, arena=protocol_spec.arena
    )
    frozen = AnchorSet(
        name=name or Path(output).name,
        version=version,
        min_anchor_battles=min_anchor_battles,
        dataset_sources={
            source_name: source.model_dump(mode="json")
            for source_name, source in task.spec.dataset.sources.items()
        },
        task=task.task,
        arena=protocol_spec.arena,
        baseline_model=baseline_model,
        protocol_id="",  # Filled after the resolved config and prompts are saved.
        languages=tuple(languages),
        ratings_by_language=ratings_by_language,
        counts_by_language=counts_by_language,
        human_battles_by_language=human_battles_by_language,
        battles_per_language=battles_per_language,
        bootstrap_seed=cfg.run.seed,
    )
    return _write_artifacts(output, cfg, panel, frozen, resolved_prompt)
