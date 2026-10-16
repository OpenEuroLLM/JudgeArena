# Plan: 3,000-battle held-out evaluation after the intermediate study

Status: user approved execution on L40S, up to8 concurrent, 15 October2026; proceeding under proposed32allocatedGPU-hour stage cap. Intermediate study remains complete at 108.8189 allocated L40S-hours; its approval excluded held-out inference. User approval: "proceed to run this on L40s (up to 8 at once)" following the32GPU-hour plan.

## Paper reference and scope

David Salinas, Omar Swelam and Frank Hutter, Tuning LLM Judge Design Decisions for 1/1000 of the Cost, arXiv:2501.17178v3, sections 5.6 and A.2: 3,548 validation instructions and 3,000 test instructions; model selection is validation-only. Section 5.6 reports 100 bootstrap replicates. Source: https://arxiv.org/html/2501.17178v3#S5.SS6 and https://arxiv.org/html/2501.17178v3#A1.SS2 .

Match the 3,000-observation held-out size, not claim a paper reproduction: our LMArena-140k English/top-20-model population, hash-based prompt split, judge checkpoints and filtering differ from the paper's filtered LMSys population. Count distinct battle IDs and also report distinct prompt counts; do not call 3,000 battle rows 3,000 unique instructions unless verified.

## 1. Freeze what will be tested before accessing test outcomes

Use the existing common operational search horizon of 28,540 attempted config-battles. Select candidates by their 3,548-battle validation results, not mixed-fidelity plotting estimates. Freeze exact configs/source hashes and selection rule in a new test-stage manifest. Do not replace validation files or the intermediate manifest.

Primary endpoints: human agreement for one validation-selected agreement winner per search arm:

- Uniform MO-Hyperband: 325aa74cc942, Gemma-4-31B-it / mid-evidence-on / fixed.
- PriMO without priors: ea6e4bd6e590, Gemma-4-31B-it / mid-criteria-off / fixed.
- PriMO with priors: f1d8421d913f, Gemma-4-31B-it / default / fixed.

Secondary cost-quality endpoints: add the unique cheaper full-fidelity frontier points from each PriMO arm:

- No priors: 88685e7dc944, Qwen3.5-9B / mid-verification-off / fixed.
- With priors: 4d39e6174e26, Qwen3.5-9B / mid-verdict-on / fixed.

Five unique configs total. This is the union of the three arm-specific full-fidelity validation fronts, including the uniform point even though it is dominated in the pooled validation front. No new search, judge families, prompt edits or optimizer seeds. Paper's size-bucket selection is a different selection rule; do not claim exact protocol replication.

## 2. Construct one frozen test panel of exactly 3,000 battles

CPU preparation on a compute node using pinned cached data, no model inference:

1. Recreate the same English/top-20-model candidate population and the existing prompt-hash validation/test partition (validation_fraction=0.5).
2. Exclude all prompts on the validation side, all qualification/calibration/search panels and any earlier test panels whose outcomes informed prior work. Include historical v1 test exclusions, not merely disjoint battle IDs. Check rendered/truncated prompt identities for duplicates across partitions too. Do not inspect test labels/scores to choose rows.
3. Audit the remaining test-pool size and prompt counts. The current 1,105 is a quota-100 sampled panel, not established to be the entire eligible pool. Confirm existing 1,105 rows remain eligible, then retain them and add exactly 1,895 rows from the same fresh pool, using a fixed hash priority and seed 0. Check existing sampling balance/connectivity of the top-20 comparison graph; document final per-model counts.
4. Require at least 3,000 genuinely eligible distinct battle IDs. If insufficient, stop and report counts; do not borrow validation rows, resplit after selection, duplicate battles, or silently broaden the population. A different population would require a revised design.
5. Freeze a new test panel file and checksum, source dataset revision, all exclusions/counts, prompt identity checks, deterministic sampling rule and panel seed before any judge runs. All five configs receive identical IDs and reference preferences.

No changing the panel after seeing outputs. Do not rerun the optimizer on test data. Exact panel availability is a preparation gate, not yet verified by this planning task.

## 3. Fixed inference settings and execution

Reuse the frozen intermediate source snapshot and exact selected config values: native checkpoint/precision, temperature 0, judge seed 42, thinking on, total output 8,192 tokens, context 65,536, completion truncation 8,192 characters, TP4, existing parsers and tie handling. All five candidates use fixed order: exactly 5 x 3,000 = 15,000 logical judge requests, before operational retries or verified cache hits. A reversed-order request is not added for this stage.

Use a separate test-stage root and manifest/results directory, e.g. judge-tuning-heldout-3000-20261015. Keep search NePS state untouched. Use Slurmpilot neutral names, offline cached checkpoints, blank API keys and read-only shared runtime. At most 8 own L40S and 10 across the user's account/chats, subject to stricter live account limits. Inspect all user GPU jobs before scheduling; never cancel others.

Group three Gemma configs sequentially in one TP4 worker and two Qwen-9B configs sequentially in another TP4 worker. Reuse loaded models only if existing runner supports the same fully resolved descriptor; otherwise account for each startup. Preserve per-config outputs and source hashes. Genuine capped/unparsed/context-error judgments remain failures under unchanged metrics. Operational recovery only; no scientific changes based on test failures. Exact cache hits may be reused if descriptor/input match; all tokens, including cache hits, count toward reference cost.

## 4. Proposed resource cap, not approval

Historical calibration measured roughly 1.52–1.65 seconds/new request for Gemma and 0.78–0.81 for Qwen-9B; these are rough warm estimates, not a forecast for new test prompts/presets. Formula for five candidates: (9,000 x Gemma_s + 6,000 x Qwen_s) x 4 / 3,600, about 20.4–21.9 L40S-hours before startup, CPU/report overhead inside GPU allocations, retries and shutdown. Recheck against completed intermediate timings before submission.

Propose a separate hard cap of 32 allocated L40S-hours: initial reservations Gemma 4 GPUs x 5 hours = 20, Qwen-9B 4 GPUs x 2 hours = 8, leaving 4 GPU-hours for cleanup/recovery. These reservations do not guarantee completion. If cost audit indicates they are inadequate, revise before launch. Count sacct parent-job ElapsedRaw x allocated GPUs for every stage job, including failures/retries, plus active/pending remaining reservations. No retry without cap headroom. Never treat unused intermediate hours as authorization for this stage.

Illustrative token-price reference cost from existing validation rates: 3 x (0.5059728974 + 0.3916198309 + 0.4196684273 + 0.3220275648 + 0.3458016065) = about USD 5.96 for all 15,000 judgments. This is equivalent token-price accounting, not a GPU monetary estimate or permission for paid APIs. Actual hosted API spend stays zero.

## 5. Predeclared analysis

- Primary: agreement on exactly 3,000 battles for each of the three frozen arm winners, with failures retained in denominator and coverage/cap/parse failures reported separately.
- Compare all three pairs on the same battles. Report paired agreement differences and 95% paired bootstrap intervals using 10,000 resamples, fixed seed 0, preserving row alignment. If prompts repeat, use prompt-cluster resampling and identify its uncertainty target. Report individual estimate standard errors too; paper used 100 bootstraps, so document our higher Monte Carlo precision as a deviation.
- Treat prior-vs-no-prior as the primary contrast; other contrasts secondary. If formal significance claims are made across all three contrasts, use a predeclared multiplicity adjustment (Holm). Do not claim optimization-seed robustness from a single search seed.
- Secondary: expected native-token USD per 1,000 battles using the same frozen prices, coverage, cap/parse-error rates, optional predeclared model-ranking diagnostics.
- Plot test cost/agreement for all five preselected configs, and validation-to-test changes. An empirical test Pareto envelope may summarize the tested set descriptively; do not use test outcomes to choose new candidates/prompts or call post-test selection independently validated.
- No testing only the appealing configurations after results arrive. Report all five, including reversals or failures. Primary agreement comparison remains the frozen three-candidate set.

## 6. Verification and deliverables

Before finalizing, prove 3,000 aligned IDs for each candidate, no prohibited prompt overlaps, immutable hashes/settings, exact agreement/coverage recomputation, native-token cost including cache hits, and complete allocation accounting. Read-only SQLite integrity checks use node-local copies on CPU compute nodes, rollback journaling preserved.

Deliver test manifest/panel hashes, config IDs, exclusion/count audit, per-battle outputs, per-config metrics, paired intervals, cost/quality plot, validation-to-test comparison, GPU ledger, full logs and concise report. Keep the intermediate validation report distinct and link the new held-out report. Pause the monitor at completion or a blocking integrity/budget issue.

## Gates

A. Approve the five-candidate scope and separate 32 allocated L40S-hour cap; no API spending.
B. Verify availability of 3,000 eligible fresh test battles and freeze panel/configs/analysis before inference.
C. Launch only once both A and B pass and measured resource estimates fit the cap. Otherwise report the concrete blocker/revised estimate.

## Preparation review before held-out inference

CPU30007737 found8,528freshrowsin9,679test-partitionrows;somepreviouslyreserved1105failstrip-normalizedpromptidentityexclusions. The sourcepanelusedexactprompttext;theheld-outauditchecksstrip-normalizedidentitiesaswell. No testinferenceoroutcomeselectionoccurred. CorrectedCPU30007985 recordsremovedreservedIDs/reasons,requiresexpectedprompt-collisionreason,retainsonlyeligibleexistingrowsandfillsexact3,000bythesamefixedhashpriorityfromthesamefreshpool. No borrowingvalidationorpopulationchange. This replaces the assumptionthatall1105couldberetained;thefreezeauditisstillrequired.
