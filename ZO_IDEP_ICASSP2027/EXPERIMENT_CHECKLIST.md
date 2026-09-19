# Experiment checklist for filling the paper

## E1. Reproduce the ICASSP 2026 sparse-regression setting
- Objective family: sparse linear regression used by Zhao et al. (ICASSP 2026).
- Start with `m=128, d=256`, `k ∈ {16,24,32}`.
- Match data generation, initialization, and noise as closely as possible to the baseline.
- Run 20-50 independent trials.

### Methods
1. PG-EP (first-order exact-penalty baseline; oracle reference)
2. SZOHT (NeurIPS 2022; black-box sparse baseline)
3. Full-ZO-EP (same exact penalty and CoordZO, never switch support)
4. ZO-IDEP (full method)
5. Optional: ZORO where its sparse-gradient assumptions are meaningful

### Metrics
- final objective
- `||x-x*||_2`
- exact support-recovery probability
- final support size
- total function queries
- wall-clock time
- first support-identification iteration `T_id`
- fraction of queries before/after `T_id`

## E2. Dimension/query scaling
- Increase `d ∈ {256, 1024, 4096}` (or larger if cheap).
- Keep `k/d` fixed in one sweep; also run fixed `k` in a second sweep if time allows.
- Plot objective/recovery against **cumulative function evaluations**, not just iterations.
- Verify whether the measured saving follows `(1-k/d)(1-T_id/T)`.

## E3. Smooth nonconvex sparse benchmark
- Recommended: sparse phase retrieval with a deterministic function-value interface.
- Compare SZOHT, Full-ZO-EP, ZO-IDEP.
- Report success probability across initializations plus query cost.
- If the exact theoretical assumptions require a bounded domain, make the restriction explicit in both code and paper.

## A1. Support-freeze ablation
At matched initialization and target objective:
- Full-coordinate ZO
- heuristic freeze (e.g. stable support for N iterations)
- certificate without strict margin
- ZO-IDEP certified freeze

## A2. Theory diagnostics
Log every iteration:
- `mu_t`
- support set and support size
- top-k ranking gap
- threshold margin
- finite-difference error bound
- certificate quantities `R, G, B`
- certificate pass/fail
- exact query count

## Falsification rules
The method's central mechanism is not supported if:
- certified support is wrong with non-negligible frequency;
- certification occurs only near termination, so query saving is negligible;
- restricted optimization needs repeated full-dimensional rechecks;
- Full-ZO-EP reaches the same result with nearly the same total query count.
