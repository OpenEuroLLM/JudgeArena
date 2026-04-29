"""Version-aware m-ArenaHard loader.

Mirrors ``judgearena/instruction_dataset/arena_hard.py``: each supported
``m-arena-hard-v{X.Y}`` maps to its dataset-native baseline, and a parallel
private dict carries the upstream HF repo id. The dispatcher in
``judgearena/instruction_dataset/__init__.py`` uses
``split_m_arena_hard_dataset`` to parse ``m-arena-hard-v{X.Y}[-{lang}|-EU]``
and then calls ``load_m_arenahard``.
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

EU_LANGUAGES: tuple[str, ...] = (
    "cs",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "it",
    "nl",
    "pl",
    "pt",
    "ro",
    "uk",
)

NON_EU_LANGUAGES: tuple[str, ...] = (
    "ar",
    "fa",
    "he",
    "hi",
    "id",
    "ja",
    "ko",
    "ru",
    "tr",
    "vi",
    "zh",
)

ALL_LANGUAGES: tuple[str, ...] = (*EU_LANGUAGES, *NON_EU_LANGUAGES)

# Dataset name -> dataset-native baseline model. Shape mirrors
# `ARENA_HARD_BASELINES` in `arena_hard.py`. v0.1 uses Aya Expanse 8B (free
# completions from CohereLabs/deja-vu-pairwise-evals); v2.0 uses Gemini 2.5 Flash.
M_ARENA_HARD_BASELINES: dict[str, str] = {
    "m-arena-hard-v0.1": "CohereLabs/aya-expanse-8b",
    "m-arena-hard-v2.0": "google/gemini-2.5-flash",
}

# Dataset name -> upstream HF repo id. Kept private; the on-disk cache subdir
# is derived from the repo's short name.
_M_ARENA_HARD_HF_REPOS: dict[str, str] = {
    "m-arena-hard-v0.1": "CohereLabs/m-ArenaHard",
    "m-arena-hard-v2.0": "CohereLabs/m-ArenaHard-v2.0",
}


def is_m_arena_hard_dataset(dataset: str) -> bool:
    return split_m_arena_hard_dataset(dataset) is not None


def split_m_arena_hard_dataset(dataset: str) -> tuple[str, str | None] | None:
    """Parse ``m-arena-hard-v{X.Y}[-{lang}|-EU]`` into ``(version, suffix)``.

    Returns ``None`` for any name that doesn't match a known version or that
    carries an unknown suffix. ``suffix`` is ``None`` for the all-languages
    variant, ``"EU"`` for the EU subset, or a 2-letter code in
    :data:`ALL_LANGUAGES`. Versioned names only -- the unversioned
    ``m-arena-hard`` alias is deliberately not accepted.
    """
    for version in M_ARENA_HARD_BASELINES:
        if dataset == version:
            return version, None
        if dataset.startswith(f"{version}-"):
            suffix = dataset[len(version) + 1 :]
            if suffix == "EU" or suffix in ALL_LANGUAGES:
                return version, suffix
            return None
    return None


def m_arena_hard_native_baseline(dataset: str) -> str | None:
    """Baseline for a dataset name, or ``None`` if it isn't m-arena-hard."""
    parsed = split_m_arena_hard_dataset(dataset)
    if parsed is None:
        return None
    return M_ARENA_HARD_BASELINES[parsed[0]]


def load_m_arenahard(
    local_path: Path,
    version: str,
    language: str | None = None,
) -> pd.DataFrame:
    """Load m-ArenaHard prompts for the requested version and language subset.

    ``version`` must be a key in :data:`M_ARENA_HARD_BASELINES`. ``language``
    is ``None`` for the full 23-language union, ``"EU"`` for the EU subset,
    or a 2-letter language code for a single-language slice.

    The returned DataFrame carries the upstream columns plus a ``lang``
    column, with ``question_id`` rewritten to ``f"{question_id}-{lang}"`` so
    multi-language slices have unique identifiers.
    """
    if version not in _M_ARENA_HARD_HF_REPOS:
        raise ValueError(
            f"Unsupported m-ArenaHard version: {version!r}. "
            f"Known versions: {sorted(_M_ARENA_HARD_HF_REPOS)}."
        )
    repo_id = _M_ARENA_HARD_HF_REPOS[version]
    local_subdir = repo_id.split("/", 1)[1]
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="*",
        local_dir=local_path / local_subdir,
        force_download=False,
    )
    m_arena_root = local_path / local_subdir

    df_union: list[pd.DataFrame] = []
    for path in sorted(m_arena_root.rglob("*.parquet")):
        lg = path.parent.name
        if language == "EU" and lg in EU_LANGUAGES:
            df = pd.read_parquet(path)
            df["lang"] = lg
            df_union.append(df)
        elif language is None or language == lg:
            df = pd.read_parquet(path)
            df["lang"] = lg
            df_union.append(df)

    assert len(df_union) > 0, (
        f"No parquet matched under {m_arena_root} for language={language!r}."
    )
    df_res = pd.concat(df_union, ignore_index=True)
    df_res["question_id"] = df_res.apply(
        lambda row: f"{row['question_id']}-{row['lang']}", axis=1
    )
    return df_res


if __name__ == "__main__":
    from judgearena.utils import data_root

    load_m_arenahard(local_path=data_root, version="m-arena-hard-v0.1", language="EU")
