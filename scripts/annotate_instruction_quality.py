"""
Example showing how to generate instruction quality with the prompt proposed in Arena-Hard.
"""

import re

from langchain_core.prompts import ChatPromptTemplate

from judgearena.utils import do_inference, make_model

max_len = 2000
system_prompt = None
user_prompt_template = f"""\
Your task is to evaluate how well the following input prompts can assess the capabilities of advanced AI assistants. For the input prompt, please analyze it based on the following 7 criteria. For each criteria, make sure to explain before determine whether the input satisfy it.

1. Specificity: Does the prompt ask for a specific, well-defined output without leaving any ambiguity? This allows the AI to demonstrate its ability to follow instructions and generate a precise, targeted response.
2. Domain Knowledge: Does the prompt test the AI’s knowledge and understanding in a specific domain or set of domains? The prompt must demand the AI to have a strong prior knowledge or mastery of domainspecific concepts, theories, or principles.
3. Complexity: Does the prompt have multiple components, variables, or levels of depth and nuance? This assesses the AI’s capability to handle complex, multi-faceted problems beyond simple queries.
4. Problem-Solving: Does the prompt require active problem-solving: analyzing and clearly defining the problem and systematically devising and implementing a solution? Note active problem-solving is not simply reciting facts or following a fixed set of instructions.
5. Creativity: Does the prompt require a creative approach or solution? This tests the AI’s ability to generate novel ideas tailored to the specific needs of the request or problem at hand.
6. Technical Accuracy: Does the prompt require an answer with a high degree of technical accuracy, correctness and precision? This assesses the reliability and truthfulness of the AI’s outputs.
7. Real-World Application: Does the prompt relate to real-world applications? This tests the AI’s ability to provide practical and actionable information that could be implemented in real-life scenarios.

After analyzing the input prompt based on these criteria, you must list the criteria numbers that the prompt satisfies in the format of a Python array like this "Criteria Satisfied: [1, 2, 4, 6, 7]".

# User-instruction (truncated to the first {max_len} characters)
{{instruction}}

# Output

**Just output the criteria satisfied as your output will be parsed.**
**Do not put any explanation.**
**Always return an output in the form of a list with the list of criteria satistied if any.**

Criteria Satisfied:
"""

judge_chat_model = make_model("OpenRouter/deepseek/deepseek-chat-v3.1", max_tokens=20)
instructions = [
    "Write Hello",
    "Yo",
    "I want you to write a python code to call LangChain to annotate instruction quality of a dataset. "
    "Include documentation and unit-tests.",
]


prompt_template = ChatPromptTemplate.from_messages(
    [
        # ("system", system_prompt),
        ("user", user_prompt_template)
    ]
)


def truncate(s: str, max_len: int | None = None):
    if not isinstance(s, str):
        return ""
    if max_len is not None:
        return s[:max_len]
    else:
        return s


inputs = prompt_template.batch(
    [
        {
            "instruction": truncate(instruction, max_len=max_len),
        }
        for instruction in instructions
    ]
)
print(f"Start LLM judge annotation ({len(inputs)} annotations).")
judge_completions = do_inference(
    chat_model=judge_chat_model,
    inputs=inputs,
    use_tqdm=False,
)

print(judge_completions)


def parse_completion(completion: str) -> list[int]:
    def safe_parse_int(x: str):
        try:
            return int(x)
        except ValueError:
            return None

    # Use a regular expression to match the criteria, e.g.
    # The string "Blablabla Criteria Satisfied: [1, 6]" is matched to [1, 6]
    regexp_match = re.search(r"\[(.*?)\]", completion)
    if regexp_match:
        # Extract the matched numbers and convert them to a list of integers
        numbers = regexp_match.group(1)
        numbers = re.sub(r"[^\d\.,]", "", numbers)  # Added 'r' prefix here
        res = [safe_parse_int(num.strip()) for num in numbers.split(",") if num]
        return [x for x in res if x is not None]
    return []


instruction_qualities = [
    parse_completion(judge_completion) for judge_completion in judge_completions
]

print(instruction_qualities)
