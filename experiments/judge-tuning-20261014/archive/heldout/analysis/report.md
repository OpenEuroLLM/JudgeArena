# Held-out evaluation: 3,000 shared battles

Completed 16 October 2026. Five configurations were frozen from validation before test inference. The test panel contains 3,000 battle IDs and 2,981 distinct normalized prompts. All five used the identical panel.

## Results

| Selection | Model / preset | Validation agreement | Test agreement (95% interval) | Coverage | Reference USD / 1,000 battles | Capped / unparsed |
|---|---|---:|---:|---:|---:|---:|
| Uniform MO-Hyperband (primary) | gemma-4-31B-it / mid-evidence-on | 47.1815% | 47.7667% (45.97–49.60) | 99.6000% | $0.493234 | 12 / 12 |
| PriMO without priors (primary) | gemma-4-31B-it / mid-criteria-off | 47.7452% | 47.0333% (45.21–48.85) | 99.8000% | $0.381694 | 6 / 6 |
| PriMO with priors (primary) | gemma-4-31B-it / default | 47.8579% | 47.8000% (46.01–49.60) | 99.6333% | $0.409911 | 11 / 11 |
| PriMO without priors (secondary) | Qwen3.5-9B / mid-verification-off | 43.4047% | 43.4333% (41.67–45.24) | 96.6667% | $0.316649 | 61 / 100 |
| PriMO with priors (secondary) | Qwen3.5-9B / mid-verdict-on | 45.2368% | 44.8000% (43.02–46.61) | 98.6667% | $0.343385 | 40 / 40 |

Intervals use 10,000 bootstrap resamples of the 2,981 prompt clusters, seed 0. These intervals describe test-sample uncertainty conditional on these selected configurations. Failed/unparsed judgments stay in the 3,000-battle denominator. Cap and unparsed counts overlap and must not be added.

## Paired comparisons among the three primary Gemma configurations

- PriMO with priors minus PriMO without priors: +0.7667 percentage points; paired 95% interval [-0.1666, +1.6955] (primary).
- PriMO with priors minus Uniform MO-Hyperband: +0.0333 percentage points; paired 95% interval [-0.8292, +0.8678] (secondary).
- PriMO without priors minus Uniform MO-Hyperband: -0.7333 percentage points; paired 95% interval [-1.7328, +0.2656] (secondary).

All three paired intervals include zero. This test does not establish an agreement advantage for priors. The prior configuration and uniform configuration have nearly identical point estimates; the prior configuration has lower reference token cost. These are selected configuration comparisons from one optimizer seed, not evidence of optimizer robustness across seeds. No formal multiplicity-adjusted significance claim is made.

## Validation to test

- `325aa74cc942`: agreement change +0.5852 percentage points; reference cost change -2.52%.
- `ea6e4bd6e590`: agreement change -0.7119 percentage points; reference cost change -2.53%.
- `f1d8421d913f`: agreement change -0.0579 percentage points; reference cost change -2.33%.
- `88685e7dc944`: agreement change +0.0286 percentage points; reference cost change -1.67%.
- `4d39e6174e26`: agreement change -0.4368 percentage points; reference cost change -0.70%.

## Cost basis and execution

Reference costs are native input/output token counts, including any cache hits, priced at the frozen USD-per-million-token rates: Gemma input 0.09/output 0.34; Qwen-9B input 0.10/output 0.15. They are normalized to 1,000 battles. They are not GPU rental costs. Actual hosted API spend was zero.

Allocated GPU accounting: Gemma 30008000 completed 11,595 seconds × 4 GPUs; failed Qwen startup 30008001 consumed 970 seconds × 4; corrected Qwen 30016680 completed 3,375 seconds × 4. Total **17.711111 L40S-hours**, below the separate approved 32-hour cap. CPU verification 30017084 completed in 1,071 seconds with zero GPUs.

The first Qwen worker failed before judgments because its entrypoint lacked the multiprocessing main guard. Only that operational issue was repaired; the failed entrypoint is preserved. Completed workers emitted shutdown/cleanup warnings; both successful parent jobs completed and all required result rows passed verification.

## Verification and provenance

- All five configurations: 3,000 aligned sampled battle IDs and 3,000 distinct annotation IDs; agreement, coverage and native-token cost independently recomputed by CPU analysis.
- Cache integrity: 10 databases copied to node-local temporary storage and opened read-only; all returned `ok`, all use `delete` rollback journaling, none WAL. Shared databases were unchanged.
- Frozen panel SHA256: `221ca1ac07ce7822458e3b1023c9971daf2cad709d85e0611ad4e513ea2cbea7`.
- Preparation audit excluded all previous prompt identities. Five old reserved rows collided only after strip normalization; excluded before outcomes, leaving 1,100 retained plus 1,900 deterministic additions from the same fresh pool. No label-based selection.
- Frozen-file audit verifies source prompt files, candidate configs, source configs, panel and provenance against their recorded hashes. Historical intermediate launch artifacts have separate exceptions: five missing launch/submission files and a changed `launch/worker_check.sh`. These are recorded in frozen-hashes.json; the audit must not be described as every historical file matching.
- Settings: fixed order, temperature 0, judge seed 42, thinking enabled, total output limit 8,192 tokens, context 65,536, completion truncation 8,192 characters, TP4, unchanged checkpoints/precision/parsers.

## Limitations

The 3,000-battle test size follows the referenced paper, but this is not an exact reproduction of its dataset/filtering. The panel has 2,981 distinct prompts, not 3,000 unique instructions. The single optimizer seed and adaptive validation selection limit algorithm-level conclusions. Test outcomes were not used to change candidates, prompts or settings.

## Artifacts and logs

- `test-comparison.json`: all estimates, bootstrap intervals and paired contrasts.
- `test-configs.parquet`: candidate-level metrics.
- `test-cost-agreement.png` / `.pdf`: all five test points; error bars are individual prompt-cluster bootstrap 95% intervals, not paired-difference intervals.
- `panel-audit.json`, `frozen-hashes.json`, `cache-integrity.json`, `gpu-ledger.json`: verification evidence.
- Remote per-battle results: `/work/dlclarge1/lushtake-hiwi/judge-tuning-heldout-3000-20261015/results/<config_id>/`.
- Logs: `/work/dlclarge1/lushtake-hiwi/slurmpilot/jobs/{ra1c19e,r509b5b,r57d0aa,rbcfc3f}/logs/`.
- Remote report directory: `/work/dlclarge1/lushtake-hiwi/judge-tuning-heldout-3000-20261015/analysis/`.
- Local report directory: `/private/tmp/judgearena-tuning-heldout-3000-20261015/analysis/`.
