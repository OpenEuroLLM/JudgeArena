"""Pinned upstream revisions for every dataset/space JudgeArena downloads.

Pinning lets the run metadata answer "exactly which version of the data did
this run see?".  When upstream rewrites a dataset (e.g. ComparIA periodically
republishes), an unpinned ``snapshot_download`` will silently start returning
different bytes; pinned revisions force callers to opt into upgrades.

To bump a revision, paste the new commit SHA from the dataset's HuggingFace
revision page (or the GitHub commit page for the FastChat raw URL).
"""

from __future__ import annotations

# HuggingFace dataset / space revisions.  Keys are HuggingFace ``repo_id``
# strings; values are commit SHAs.  ``None`` is allowed for repos where we
# do not yet have a stable pin and is recorded as such in the metadata so
# the gap is visible.
HF_DATASET_REVISIONS: dict[str, str | None] = {
    # LMArena human-preference battles
    "lmarena-ai/arena-human-preference-100k": "72e85b3ddc9c81bf7b659d6b03d4126dfd8fb34a",
    "lmarena-ai/arena-human-preference-140k": "6322995ab34d7c2693e3f47dd13fa5caa0789a74",
    "lmarena-ai/arena-human-preference-55k": "18c298340948c0e7f7727399fd459cca6ce0ca6f",
    # ComparIA (already pinned via the legacy comparia_revision argument).
    "ministere-culture/comparia-votes": "7a40bce496c1f2aa3be4001da85a49cb4743042b",
    # m-ArenaHard (Cohere release)
    "CohereLabs/m-ArenaHard": "ab393a96cd0b134a1acfa96e080af31e5e73a393",
    # AlpacaEval instructions / model_outputs (geoalgo redistribution; the
    # repo now redirects to ``judge-arena/judge-arena-dataset`` upstream, but
    # ``snapshot_download`` follows the redirect transparently).
    "geoalgo/llmjudge": "004c4a992956eeefffd36b63ade470f32fd0a582",
    # MT-Bench questions (LMSYS Space).
    "lmsys/mt-bench": "a4b674ca573c24143824ac7f60d9173e7081e37d",
    # Multilingual fluency contexts.
    "geoalgo/multilingual-contexts-to-be-completed": "06e73c95ad18d71a04b5a1b6464ed89d38195039",
    # Arena-Hard official source (used via datasets.load_dataset).
    "lmarena-ai/arena-hard-auto": "15f3746e21432264ce9b453999bde4f3c946d2e6",
}


# Raw-URL pins (e.g. FastChat reference answers fetched as a raw GitHub URL).
# Mapping is "logical name" -> commit SHA on the upstream repo.  The downloader
# rewrites the URL to point at the pinned SHA.
RAW_URL_REVISIONS: dict[str, str | None] = {
    "lm-sys/FastChat": "587d5cfa1609a43d192cedb8441cac3c17db105d",
}


def hf_revision(repo_id: str) -> str | None:
    """Return the pinned revision for ``repo_id`` (or ``None`` if not pinned)."""
    return HF_DATASET_REVISIONS.get(repo_id)


def all_dataset_revisions() -> dict[str, str | None]:
    """Return a copy of every pin recorded in this module.

    Used by :func:`judgearena.repro.write_run_metadata` to record the
    pin table alongside each run so future readers know which version of
    the data was visible at the time of the run.
    """
    return {
        **HF_DATASET_REVISIONS,
        **{f"raw:{k}": v for k, v in RAW_URL_REVISIONS.items()},
    }
