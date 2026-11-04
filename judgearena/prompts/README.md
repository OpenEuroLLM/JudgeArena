# Decision judging

Use `Decisions/<model>` with the `decision` prompt preset. Export
`OPENROUTER_API_KEY`, then run either example from the repository root:

```bash
judgearena --config_path configs/decision_pairwise.yaml
judgearena --config_path configs/decision_independent.yaml
```

[Pairwise](../../configs/decision_pairwise.yaml) generates four Gemma answers
and compares them with saved AlpacaEval reference answers using Choice.
[Independent](../../configs/decision_independent.yaml) scores four Gemma answers
using Score and reports their mean, without a baseline. These are custom
evaluations on AlpacaEval data, not its official judging protocol.

Both configs inherit the dataset from the task and supply a complete replacement
protocol. Change the candidate, judge, questions or metrics in the config.

## Inputs and results

| Judging mode | Default state | Result conversion |
|---|---|---|
| `pairwise` | `instruction`, `completion_a`, `completion_b` | `result.preference` |
| `single_answer` | `instruction`, `completion` | `result.score`, or raw answers when omitted |

Questions can mix Noul, Choice and Score. `state` optionally maps your question's
state names to prepared fields, including adapter metadata:

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

The endpoint must implement the Decisions request/response contract; an ordinary
chat-completions endpoint is insufficient. For a compatible local server, change
`judge.engine_kwargs.endpoint` and `judge.model`; omit `api_key_env` if no
authentication is needed. Text-generation sampling settings do not control
decision requests.

OpenRouter execution and cache replay have been verified. Local vLLM deployment
and in-process decision inference are not verified by this integration.
