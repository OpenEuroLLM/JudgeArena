# Decision judging

Use `OpenRouter/typesafe/jev-1.14` with the `decision` prompt preset. Export
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

## Noul example

To ask whether A is better than B, replace `judge.prompt` in the pairwise
example with:

```yaml
prompt:
  preset: decision
  options:
    judging_mode: pairwise
    questions:
      a_better:
        type: noul
        instructions: Does completion_a satisfy instruction better than completion_b?
    result:
      preference:
        question: a_better
        invert: true
```

Noul returns the probability of yes. With `invert: true`, a probability of
`0.8` becomes preference `0.2`, favoring A. No Choice mapping or aggregation
setting is needed.

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

The server must implement the Decisions request/response contract. Set
`api_key_env` if it needs authentication.
