"""Task identifiers and the ELO task -> arena mapping."""

from __future__ import annotations

ELO_TASK_PREFIX = "elo-"

# Lowercase CLI task name -> canonical arena identifier used inside
# ``judgearena.arenas_utils.KNOWN_ARENAS`` and the ``benchmark`` column of
# saved battle dataframes.  The CLI stays lowercase (matching ``alpaca-eval``
# conventions) while internal identifiers keep their original casing.
ELO_TASK_TO_ARENA: dict[str, str] = {
    "elo-lmarena-100k": "LMArena-100k",
    "elo-lmarena-140k": "LMArena-140k",
    "elo-lmarena": "LMArena",
    "elo-comparia": "ComparIA",
}
