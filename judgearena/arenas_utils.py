import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from fast_langdetect import detect_language
from huggingface_hub import snapshot_download

from judgearena.log import get_logger
from judgearena.tasks.schema import HuggingFaceDatasetSource

logger = get_logger(__name__)


def _download_arena_dataset(
    *,
    repo_id: str,
    default_allow_patterns: str | tuple[str, ...],
    dataset_sources: Mapping[str, HuggingFaceDatasetSource],
) -> str:
    """Download one arena source at the revision pinned by its task definition."""
    try:
        source = dataset_sources[repo_id]
    except KeyError as exc:
        raise ValueError(
            f"Arena task does not declare required dataset source {repo_id!r}."
        ) from exc
    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=source.allow_patterns or default_allow_patterns,
        force_download=False,
        revision=source.revision,
    )


def extract_turn_text(turn: dict) -> str:
    """Extract plain text from a conversation turn (user or assistant).

    Handles both the 100k schema (content is a plain string) and the 140k
    schema (content is an array of {type, text, ...} objects). Moderated or
    empty turns ship ``content: None`` and yield an empty string.
    """
    content = turn.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text") or ""
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


KNOWN_ARENAS = ["LMArena-100k", "LMArena-55k", "LMArena-140k", "ComparIA"]


def _load_arena_dataframe(
    arena: str,
    *,
    dataset_sources: Mapping[str, HuggingFaceDatasetSource],
) -> pd.DataFrame:
    assert arena in KNOWN_ARENAS
    if arena == "LMArena-55k":
        repo_id = "lmarena-ai/arena-human-preference-55k"
        path = _download_arena_dataset(
            repo_id=repo_id,
            default_allow_patterns="*.csv",
            dataset_sources=dataset_sources,
        )
        df = pd.read_csv(Path(path) / "train.csv")

        def _winner_55k(row) -> str | None:
            if row["winner_tie"]:
                return "tie"
            if row["winner_model_a"]:
                return "model_a"
            if row["winner_model_b"]:
                return "model_b"
            return None

        df["winner"] = df.apply(_winner_55k, axis=1)
        df = df[df["winner"].notna()].copy()

        df["conversation_a"] = df.apply(
            lambda r: [
                {"role": "user", "content": str(r["prompt"])},
                {"role": "assistant", "content": str(r["response_a"])},
            ],
            axis=1,
        )
        df["conversation_b"] = df.apply(
            lambda r: [
                {"role": "user", "content": str(r["prompt"])},
                {"role": "assistant", "content": str(r["response_b"])},
            ],
            axis=1,
        )
        df["question_id"] = df["id"]
        df["tstamp"] = 0
        df["benchmark"] = "LMArena-55k"

    elif "LMArena" in arena:
        size = arena.split("-")[1]  # "100k" or "140k"
        repo_id = f"lmarena-ai/arena-human-preference-{size}"
        path = _download_arena_dataset(
            repo_id=repo_id,
            default_allow_patterns="*parquet",
            dataset_sources=dataset_sources,
        )
        parquet_files = sorted((Path(path) / "data").glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        if "tstamp" in df.columns:
            # 100k: tstamp is a unix timestamp in seconds
            df["date"] = pd.to_datetime(df["tstamp"], unit="s")
        else:
            # 140k: timestamp is already a datetime
            df["tstamp"] = df["timestamp"].astype("int64") // 10**9
            df["date"] = df["timestamp"]

        if "question_id" not in df.columns:
            df["question_id"] = df["id"]

        df["benchmark"] = arena

    else:
        path = _download_arena_dataset(
            repo_id="ministere-culture/comparia-votes",
            default_allow_patterns="*",
            dataset_sources=dataset_sources,
        )

        df = pd.read_parquet(Path(path) / "votes.parquet")

        # unify schema
        df["tstamp"] = df["timestamp"]
        df["model_a"] = df["model_a_name"]
        df["model_b"] = df["model_b_name"]

        def get_winner(
            chosen_model_name: str,
            model_a: str,
            model_b: str,
            both_equal: bool,
            **kwargs,
        ):
            if both_equal:
                return "tie"
            else:
                if chosen_model_name is None or isinstance(chosen_model_name, float):
                    return None
                if chosen_model_name not in [model_a, model_b]:
                    warnings.warn(
                        f"Chosen model {chosen_model_name!r} not in model_a={model_a!r} or model_b={model_b!r}; skipping.",
                        stacklevel=2,
                    )
                    return None
                return "model_a" if chosen_model_name == model_a else "model_b"

        df["winner"] = df.apply(lambda row: get_winner(**row), axis=1)

        # filter battles without winner annotated
        df = df[~df.winner.isna()]
        df["benchmark"] = "ComparIA"
        df["question_id"] = df["id"]

    df["lang"] = df["conversation_a"].apply(
        lambda conv: detect_language(extract_turn_text(conv[0])).lower()
    )

    cols = [
        "question_id",
        "tstamp",
        "model_a",
        "model_b",
        "winner",
        "conversation_a",
        "conversation_b",
        "benchmark",
        "lang",
    ]
    df = df.loc[:, cols]

    # keep only one turn conversation for now as they are easier to evaluate
    df["turns"] = df.apply(lambda row: len(row["conversation_a"]) - 1, axis=1)
    n_before = len(df)
    df = df.loc[df.turns == 1]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.info(
            "[%s] Dropped %d/%d multi-turn battles (keeping single-turn only).",
            arena,
            n_dropped,
            n_before,
        )

    return df


def load_arena_dataframe(
    arena: str | None,
    *,
    dataset_sources: Mapping[str, HuggingFaceDatasetSource],
) -> pd.DataFrame:
    """Load battles from one or all arenas.

    :param arena: one of "LMArena-100k", "LMArena-140k", "ComparIA", "LMArena"
                  (concatenation of both LMArena variants), or None (all arenas).
    :param dataset_sources: pinned sources declared by the task, keyed by repo ID.
    :return: dataframe containing battles for the arena(s) selected.
    """
    if arena is None:
        arenas = KNOWN_ARENAS
    elif arena == "LMArena":
        arenas = ["LMArena-100k", "LMArena-55k", "LMArena-140k"]
    else:
        return _load_arena_dataframe(arena, dataset_sources=dataset_sources)
    return pd.concat(
        [_load_arena_dataframe(a, dataset_sources=dataset_sources) for a in arenas],
        ignore_index=True,
    )


def main():
    from judgearena.datasets import load_battles
    from judgearena.tasks.registry import load_tasks
    from judgearena.tasks.schema import EloProtocol

    for task_id, task in load_tasks().items():
        if not isinstance(task.spec.protocol, EloProtocol):
            continue
        logger.info("Loading %s", task_id)
        df = load_battles(task)
        arena = task.spec.protocol.arena
        n_battles = len(df)
        n_models = len(set(df["model_a"]) | set(df["model_b"]))
        n_languages = df["lang"].nunique()
        logger.info(
            "%s: %d battles, %d models, %d languages",
            arena,
            n_battles,
            n_models,
            n_languages,
        )


if __name__ == "__main__":
    main()
