import warnings
from pathlib import Path

import pandas as pd
from fast_langdetect import detect_language
from huggingface_hub import snapshot_download


def _extract_instruction_text(turn: dict) -> str:
    """Extract plain instruction text from a conversation first turn.

    Handles both the 100k schema (content is a plain string) and the 140k
    schema (content is an array of {type, text, ...} objects).
    """
    content = turn["content"]
    if isinstance(content, str):
        return content
    return " ".join(block["text"] for block in content if block.get("type") == "text")


KNOWN_ARENAS = ["LMArena-100k", "LMArena-140k", "ComparIA"]


def _load_arena_dataframe(
    arena: str, comparia_revision: str | None = None
) -> pd.DataFrame:
    assert arena in KNOWN_ARENAS
    if "LMArena" in arena:
        size = arena.split("-")[1]  # "100k" or "140k"
        path = snapshot_download(
            repo_id=f"lmarena-ai/arena-human-preference-{size}",
            repo_type="dataset",
            allow_patterns="*parquet",
            force_download=False,
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

        # 140k uses "both_bad" instead of "tie (bothbad)"
        df["winner"] = df["winner"].replace("both_bad", "tie (bothbad)")

        df["benchmark"] = arena

    else:
        path = snapshot_download(
            repo_id="ministere-culture/comparia-votes",
            repo_type="dataset",
            allow_patterns="*",
            revision=comparia_revision,
            force_download=False,
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
        lambda conv: detect_language(_extract_instruction_text(conv[0])).lower()
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
        print(
            f"[{arena}] Dropped {n_dropped}/{n_before} multi-turn battles (keeping single-turn only)."
        )

    return df


def load_arena_dataframe(
    arena: str | None,
    comparia_revision: str = "7a40bce496c1f2aa3be4001da85a49cb4743042b",
) -> pd.DataFrame:
    """Load battles from one or all arenas.

    :param arena: one of "LMArena-100k", "LMArena-140k", "ComparIA", "LMArena"
                  (concatenation of both LMArena variants), or None (all arenas).
    :param comparia_revision: pinned revision for the ComparIA dataset.
    :return: dataframe containing battles for the arena(s) selected.
    """
    if arena is None:
        arenas = KNOWN_ARENAS
    elif arena == "LMArena":
        arenas = ["LMArena-100k", "LMArena-140k"]
    else:
        return _load_arena_dataframe(arena, comparia_revision)
    return pd.concat(
        [_load_arena_dataframe(a, comparia_revision) for a in arenas],
        ignore_index=True,
    )


def main():
    for arena in KNOWN_ARENAS:
        print(f"Loading {arena}")
        df = _load_arena_dataframe(arena)
        n_battles = len(df)
        n_models = len(set(df["model_a"]) | set(df["model_b"]))
        n_languages = df["lang"].nunique()
        print(
            f"{arena}: {n_battles} battles, {n_models} models, {n_languages} languages"
        )


if __name__ == "__main__":
    main()
