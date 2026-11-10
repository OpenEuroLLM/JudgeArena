# EU-Guard

- `eu-guard`: generate responses, then judge whether each response refuses the request.
- `eu-guard-guards`: classify the dataset prompts directly; no generation model is needed.

Both use the independent pipeline and report F1, false compliance rate (FCR),
false refusal rate (FRR), and coverage against the dataset labels.

```bash
judgearena --config_path configs/eu_guard_jev.yaml
judgearena --config_path configs/eu_guard_prompt_jev.yaml
judgearena --config_path configs/eu_guard_prompt_gemma.yaml
```

Use `--run.filters '{"lang":["en","de","zh","pl","it"]}'` to select languages.
The task defaults use the shared `decision` preset with questions from `prompts/`.
The Gemma bundle uses the shared `json-score` parser. Registered bundle names are
`eu-guard-refusal-jev`, `eu-guard-guard-jev`, and `eu-guard-guard-gemma`.

`scripts/eu_guard.py` shows how an external guard can supply labels to the same metric.
