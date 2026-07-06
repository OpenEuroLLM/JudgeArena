"""Project-wide constants: task identifiers and task→arena mapping."""

from __future__ import annotations

ELO_TASK_PREFIX = "elo-"
"""Prefix marking a task as an ELO-rating run (e.g. ``elo-lmarena-100k``)."""

# Lowercase CLI task name -> canonical arena identifier used inside
# the arena loaders and the ``benchmark`` column of saved battle dataframes.
# The CLI stays lowercase (matching ``alpaca-eval`` conventions) while internal
# identifiers keep their original casing.
ELO_TASK_TO_ARENA: dict[str, str] = {
    "elo-lmarena-100k": "LMArena-100k",
    "elo-lmarena-140k": "LMArena-140k",
    "elo-lmarena": "LMArena",
    "elo-comparia": "ComparIA",
}


# vLLM reasoning markers shared by the inference layer (judgearena.models) and
# the reasoning-tag stripping in judgearena.utils.text.
VLLM_REASONING_START_STR = "<think>"
VLLM_REASONING_END_STR = (
    "I have to give the solution based on the thinking directly now.</think>"
)
