# Intermediate judge tuning: validation-only results

One optimizer seed per arm; 60 configurations; fixed 400/1,200/3,548-battle panels. All inference was local. No held-out inference or paid API calls.

## Verification

- All 50 logical trial reports independently match recomputed agreement, coverage and expected cost from native token usage, including cached rows. Fixed settings and exact panel IDs verified.
- GPU allocation including qualification and retries: 108.8189 L40S-hours (120 approved).
- Uniform used 21 execution attempts for 20 logical evaluations after a timeout; both PriMO arms used 15 evaluations. The timed-out panel counts against the operational common horizon.
- Common horizon: 28,540 attempted config-battles; whole completed trials only. Native attempted totals: uniform 32,088 including retry; PriMO 30,436 each.
- Within this horizon uniform has four full-fidelity configurations and lacks Qwen-9B; each PriMO arm has six and covers all three models.
- Each PriMO arm made six BO proposals. GP acquisitions used supplied priors in five informative-arm calls and zero control calls; probabilistic prior omission is native NePS behavior.

## Best validation agreement within the common horizon

- Uniform MO-Hyperband: 47.182% agreement, $0.5060/1,000 battles; VLLM/google/gemma-4-31B-it, {"judge.model": "VLLM/google/gemma-4-31B-it", "judge.prompt_preset": "mid-evidence-on", "judge.swap_mode": "fixed"}.
- PriMO without priors: 47.745% agreement, $0.3916/1,000 battles; VLLM/google/gemma-4-31B-it, {"judge.model": "VLLM/google/gemma-4-31B-it", "judge.prompt_preset": "mid-criteria-off", "judge.swap_mode": "fixed"}.
- PriMO with informative priors: 47.858% agreement, $0.4197/1,000 battles; VLLM/google/gemma-4-31B-it, {"judge.model": "VLLM/google/gemma-4-31B-it", "judge.prompt_preset": "default", "judge.swap_mode": "fixed"}.

## Paired differences (percentage points)

- PriMO with informative priors minus PriMO without priors: 0.113 pp; paired 95% bootstrap interval [-0.789, 0.986] pp.
- PriMO with informative priors minus Uniform MO-Hyperband: 0.676 pp; paired 95% bootstrap interval [-0.141, 1.466] pp.
- PriMO without priors minus Uniform MO-Hyperband: 0.564 pp; paired 95% bootstrap interval [-0.338, 1.466] pp.

## Interpretation and limitations

No clear agreement benefit from informative priors is established: every paired interval includes zero. Both PriMO variants found validation configurations that were cheaper and slightly more accurate than uniform’s best, but one seed and adaptive validation selection cannot establish a reliable optimizer advantage. Cost/quality trade-offs remain: the no-prior Gemma frontier point is cheaper than informative-prior Gemma, with very similar agreement.

These intervals resample battles for already selected configurations; they do not measure optimizer-seed uncertainty or provide confirmatory held-out evidence. Uniform’s timeout and whole-trial horizon cause it to lack full-fidelity Qwen-9B inside the operational horizon. Sensitivity excluding timeout/retry duplication: uniform has five full-fidelity configurations and includes Qwen-9B within 28,540 logical config-battles. The best agreement/cost pair is unchanged for every arm, so this sensitivity does not reverse the reported best-agreement comparison.

Temperature zero did not yield identical independent generations across nodes. Shared cache reuse may reduce later arms’ GPU runtime, so runtime is operational accounting rather than a fair algorithm-speed comparison. Expected USD costs are frozen token-price reference estimates; actual API spend was zero.

Metadata erratum: configs/manifest.json retains an inherited allocated_l40s_hour_cap=16 field; its stage label, execution approval, provenance and scheduler limits use the approved 120-hour cap. Preserve the original manifest; this report records the stale field explicitly.

BO proposal diversity: six unique configurations without priors versus five unique configurations across six proposals with informative priors. A repeated proposal is native optimizer behavior, not a new configuration.

All 10 cache databases passed read-only SQLite integrity checks on node-local temporary copies in CPU job 30006935 (18 seconds, zero GPUs). Every database uses rollback journaling (`delete`), with no WAL databases. Copy SHA-256 hashes and per-database results are recorded in cache-integrity.json; shared source databases were unchanged. All 53 frozen source, prompt, configuration and panel file hashes match their recorded provenance (frozen-hashes.json).

Source artifacts: trials.parquet, verified-trials.parquet, comparison.json, paired-validation.json, metric-verification.json, optimizer-observations files and original annotations.

## Search trajectory figures

- [Merged Pareto plots](pareto-all-points.pdf): all three algorithms on one shared plot, with logarithmic cost axis. Colors identify algorithms, circle sizes identify 400/1,200/3,548-battle fidelities. Other observations are faded; hollow points are beyond the common horizon. A neutral line connects the empirical non-dominated set pooled across all three algorithms, using eligible observed evaluations across all fidelities. Agreement estimates use different panel sizes; this is a mixed-fidelity empirical front.
- [Agreement incumbents](search-trajectories.pdf): one best-so-far observed agreement curve per algorithm across all fidelities, against attempted battle budget only. Circle size identifies the fidelity of each incumbent-setting evaluation. No individual trial-score traces or point labels. These mixed-fidelity observed scores use different panel sizes and do not replace the full-fidelity comparison.
- [Merged Pareto-front evolution](pareto-evolution.pdf): shared algorithm axes at 10,000, 20,000 and 28,540 attempted config-battles, with consistent algorithm colors and fidelity sizes. Each snapshot connects the combined observed non-dominated set across algorithms and fidelities.

Reproduce with `launch/plot_trajectories.py`; chart source rows are `analysis/trajectory-points.csv`. These figures reuse existing verified data and require no inference. The incumbent data are also saved in `analysis/incumbent-points.csv`. Fidelity is encoded by marker area in three discrete sizes; sizes are not proportional to battle counts. No uncertainty bands are shown because there is one optimizer seed per arm.
