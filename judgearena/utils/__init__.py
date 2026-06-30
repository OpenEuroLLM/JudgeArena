"""Generic, reusable utilities for JudgeArena.

Organized into focused submodules — :mod:`~judgearena.utils.text`,
:mod:`~judgearena.utils.io`, and :mod:`~judgearena.utils.eval` — and re-exported
here so ``from judgearena.utils import X`` keeps working. The model/inference
layer lives in :mod:`judgearena.models` (it is a core component, not a utility).
"""

from judgearena.utils.eval import (
    compute_pref_summary,
    print_results,
)
from judgearena.utils.io import (
    Timeblock,
    cache_function_dataframe,
    data_root,
    download_all,
    download_hf,
    generation_cache_token,
    read_df,
    safe_parse_int,
)
from judgearena.utils.text import (
    safe_text,
    strip_thinking_tags,
    strip_thinking_tags_with_metadata,
    truncate,
)

__all__ = [
    "Timeblock",
    "cache_function_dataframe",
    "compute_pref_summary",
    "data_root",
    "download_all",
    "download_hf",
    "generation_cache_token",
    "print_results",
    "read_df",
    "safe_parse_int",
    "safe_text",
    "strip_thinking_tags",
    "strip_thinking_tags_with_metadata",
    "truncate",
]
