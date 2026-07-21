"""Project-wide inference constants."""

from __future__ import annotations

# vLLM reasoning markers shared by the inference layer (judgearena.models) and
# the reasoning-tag stripping in judgearena.utils.text.
VLLM_REASONING_START_STR = "<think>"
VLLM_REASONING_END_STR = (
    "I have to give the solution based on the thinking directly now.</think>"
)
