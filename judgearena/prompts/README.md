# Prompt bundles

A YAML bundle keeps a template and its parser together:

```yaml
# grammar.yaml
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
