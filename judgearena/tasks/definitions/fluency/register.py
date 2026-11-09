"""Register this task family's reusable components."""

from judgearena.datasets.registry import register_dataset
from judgearena.prompts.parsing import JUDGE_PARSERS
from judgearena.prompts.registry import (
    FLUENCY_JUDGE_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)
from judgearena.tasks.definitions.fluency.dataset import FluencyAdapter

FLUENCY_SYSTEM_PROMPT = (
    "You are a highly efficient assistant, who evaluates and selects the best "
    "large language model based on the quality of completion of a sentence. "
    "You will see a sentence to be completed and two completions from "
    "Assistant A and Assistant B and will have to decide which one is best. "
    "Make sure to not over-confidently prefer one assistant or the other and "
    "also make sure to not bias your preference based on the ordering or on "
    "the length of the answers."
)

SCORE_PARSER = JUDGE_PARSERS["score"]

register_prompt(
    JudgePromptPreset(
        name=FLUENCY_JUDGE_PROMPT_PRESET,
        parser=SCORE_PARSER,
        inline_system=FLUENCY_SYSTEM_PROMPT,
        user_file="prompt.txt",
        output_format_file="score-output.txt",
    )
)


register_dataset("fluency")(FluencyAdapter)
