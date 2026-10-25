# ZO-IDEP reproducibility package

This directory implements the theory-aligned experiment plan of the paper.
Run commands from `ZO_IDEP_ICASSP2027/`:

```bash
python zo_idep/experiments/exp00_unit_tests.py
bash zo_idep/scripts/run_core_suite.sh
```

Each experiment writes newline-delimited records with the exact seed and
configuration to `results/raw/`, and an aggregate JSON file (including every
failed or slow seed) to `results/summary/`.  The plotting script consumes only
these frozen summaries and writes vector PDFs to `results/figures/`; the
three-panel diagnostic is additionally written to `figures/theory_diagnostics.pdf`
(git-ignored).

Development seeds and report seeds are disjoint.  Frozen protocol values are
recorded in `configs/` and redundantly embedded in every raw record.  Function
query totals exclude analytical-gradient diagnostics and shadow-full audits;
their overhead is logged separately.  The nonlinear rate experiment floors
the geometric finite-difference radius at `1e-6` to prevent floating-point
function-value cancellation, after a development run exposed a false numerical
certificate below machine-resolvable radii.

The sparse phase-retrieval experiment is explicitly a local-theory diagnostic.
Its `controlled_local_topk` warm start perturbs the planted synthetic signal and
is not a practical or query-free recovery initializer.  It tests preservation
of the Full-ZO-EP local endpoint and post-certificate query reduction in a
globally nonconvex objective; it must not be presented as end-to-end phase
retrieval performance.
