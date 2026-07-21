"""Pinned revisions for external data not owned by packaged task YAML.

Pinning lets the run metadata answer "exactly which version of the data did
this run see?".  When upstream rewrites a dataset (e.g. ComparIA periodically
republishes), an unpinned ``snapshot_download`` will silently start returning
different bytes; pinned revisions force callers to opt into upgrades.

Sources used by declarative tasks keep their revision beside the task under
``judgearena/tasks/definitions`` instead of duplicating it here.

To bump a revision, paste the new commit SHA from the dataset's HuggingFace
revision page (or the GitHub commit page for the FastChat raw URL).
"""

from __future__ import annotations

# HuggingFace dataset / space revisions.  Keys are HuggingFace ``repo_id``
# strings; values are commit SHAs.  ``None`` is allowed for repos where we
# do not yet have a stable pin and is recorded as such in the metadata so
# the gap is visible.
HF_DATASET_REVISIONS: dict[str, str | None] = {
    # Sources used by packaged tasks are pinned in their task YAML definitions.
}


# Raw-URL pins (e.g. FastChat reference answers fetched as a raw GitHub URL).
# Mapping is "logical name" -> commit SHA on the upstream repo.  The downloader
# rewrites the URL to point at the pinned SHA.
RAW_URL_REVISIONS: dict[str, str | None] = {}


def hf_revision(repo_id: str) -> str | None:
    """Return the pinned revision for ``repo_id`` (or ``None`` if not pinned)."""
    return HF_DATASET_REVISIONS.get(repo_id)


def all_dataset_revisions() -> dict[str, str | None]:
    """Return a copy of every pin recorded in this module.

    Used by :func:`judgearena.artifacts.metadata.write_run_metadata` to record the
    pin table alongside each run so future readers know which version of
    the data was visible at the time of the run.
    """
    return {
        **HF_DATASET_REVISIONS,
        **{f"raw:{k}": v for k, v in RAW_URL_REVISIONS.items()},
    }
