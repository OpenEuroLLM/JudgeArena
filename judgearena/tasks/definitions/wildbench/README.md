# WildBench

`wildbench-score` evaluates one response with the official WildBench V2 checklist prompt. `wildbench-reward` compares the evaluated model against the three official reference-output sets.

The shared task base pins the example dataset and common judge defaults. Each public YAML selects its mode, prompt, baseline policy, and versioned scorer. Dataset normalization, official prompt rendering, and metric implementations remain in the corresponding WildBench adapters.
