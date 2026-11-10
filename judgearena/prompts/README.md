# Prompt bundles

A YAML bundle keeps a template and its parser together:

```yaml
# grammar.yaml
judging_mode: single_answer
parser: json-score
user: |
  Rate grammatical correctness from 1 to 10.
  Answer: {completion}
  Return JSON: {{"score": number}}
```

Select it in your run config:

```yaml
judge:
  prompt:
    file: grammar.yaml
```

`judging_mode` describes whether the prompt receives one answer (`single_answer`)
or two (`pairwise`). A preset selection inherits its mode; it cannot override it.
For existing standalone configs that omit the mode, it is inferred from the parser.

Paths are relative to the containing YAML file. A bundle can also contain
`preset` and `options` to configure a registered bundle.
Saved run configs contain the expanded settings, so replay does not require
the original bundle file.

## Task-local bundles

Keep benchmark-specific bundles beside the task, for example
`tasks/definitions/my_benchmark/prompts/judge.yaml`. In the task definition:

```yaml
protocol:
  judge:
    prompt:
      file: prompts/judge.yaml
    category_prompts:
      coding:
        file: prompts/coding.yaml
```

Tasks and run configs accept the same prompt settings. Paths resolve beside the
file that declares them, including inherited task files. `judge.prompt` in a run
config overrides the task's prompt and category routing.

Existing preset names such as `alpaca-eval` and `arena-hard` remain available.
`alpaca-eval-json` and `alpaca-eval-pair-score` select the JSON ranking and
score-pair variants; either can be used with compatible pairwise workflows.
Place a `register.py` beside a task family to make its names available globally:

```python
# tasks/definitions/my_task/register.py
from pathlib import Path
from judgearena.prompts import JudgePromptPreset, register_prompt
from judgearena.prompts.parsing import register_parser, resolve_judge_parser

here = Path(__file__).resolve().parent
register_parser("my-verdict", resolve_judge_parser(f"{here}/parsers.py:PARSER"))
register_prompt(JudgePromptPreset.from_file("my-judge", here / "prompts/judge.yaml"))
```

Then use `prompt: {preset: my-judge}` in any task or run config. Names are freely
chosen; they need not match the folder or task name. Duplicate names raise errors.
Packaged `*/register.py` files load once. For external task folders, call
`load_tasks(definitions_root=...)` before loading the experiment config.
Registration files should only register components, without loading data or models.

A benchmark-specific parser can be referenced directly, without registration:

```yaml
# arena_hard/prompts/pairwise.yaml
judging_mode: pairwise
parser: ../parsers.py:ARENA_HARD_VERDICT
system_file: arena-hard-system-prompt.txt
user_file: arena-hard-prompt.txt
```

`parsers.py` exports `ARENA_HARD_VERDICT = ArenaHardVerdict()`. The object follows
`OutputParser`: `parse_result`, `result_type`, `requires_top_logprobs`, and a
metadata `name`. Shared parsers still use names such as `json-score`.

The Python path is relative to the YAML that declares it. Saved runs copy the
referenced source file for replay; its imported dependencies come from the Python
environment. Discovery imports only `register.py`; other Python files load when
explicitly referenced. Registered YAML bundles are expanded into saved configs.

# Decision judging

Use `OpenRouter/typesafe/jev-1.14` with the `decision` prompt preset. Export
`OPENROUTER_API_KEY`, then run an example from the repository root:

```bash
judgearena --config_path configs/decision_pairwise.yaml --task alpaca-eval
judgearena --config_path configs/decision_noul.yaml --task alpaca-eval
judgearena --config_path configs/decision_independent.yaml --task alpaca-eval
```

[Pairwise](../../configs/decision_pairwise.yaml) generates four Gemma answers
and compares them with saved AlpacaEval reference answers using Choice.
[Noul](../../configs/decision_noul.yaml) asks whether A is better than B and
converts the yes probability to a preference with `invert: true`.
[Independent](../../configs/decision_independent.yaml) scores four Gemma answers
using Score and reports their mean, without a baseline. These are custom
evaluations on AlpacaEval data, not its official judging protocol.

Select the dataset with `--task`. Pairwise examples inherit its baseline; the
independent example replaces the protocol to score answers without a baseline.
Change the candidate, judge, questions or metrics in the config.

## Inputs and results

| Judging mode | Default state | Result conversion |
|---|---|---|
| `pairwise` | `instruction`, `completion_a`, `completion_b` | `result.preference` |
| `single_answer` | `instruction`, `completion` | `result.score`, or raw answers when omitted |

Questions can mix Noul, Choice and Score. Omit `state` to use the defaults above.
To rename fields or include metadata, map the names Jev receives (left) to
prepared data fields (right):

```yaml
state:
  instruction: instruction
  completion: completion
  language: lang
```

- Choice preferences map each choice to a value: **0 favors A, 1 favors B**.
  `aggregation: selected` uses the chosen label; `expected` weights the values
  by the returned probabilities.
- Noul preferences select a question's yes probability. Use
  `{question: a_better, invert: true}` when yes means A wins.
- Scalar results select a Score or Noul question with
  `result: {score: {question: quality}}`. A pairwise runner can also use this
  single-answer mode to score each candidate and derive a preference.

Converted annotations retain all named answers in `parsed.details`, including
returned probabilities and confidence. Without a conversion, `parsed` contains
the named answers directly. Raw responses are retained for cache replay;
configured metrics consume the converted values or these details.

## Endpoints

The prompt renders decision requests; the model prefix selects the provider.
`prepare_model()` routes these inputs automatically. Direct `make_model()`
callers must pass `input_mode="decision"`.
OpenRouter supplies its decision endpoint and reads `OPENROUTER_API_KEY`.
For another compatible HTTP server, set the endpoint explicitly:

```yaml
judge:
  model: ChatOpenAI/openjev/openjev
  engine_kwargs:
    endpoint: http://localhost:8000/v1/systemone
  prompt:
    preset: decision
    options: # questions and result conversion as above
```
