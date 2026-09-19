# ZO-IDEP Experiment Report

> **Status: superseded.** This report documents the early `scripts/` prototype,
> whose certificate is weaker than the paper's Theorem 3.3 and whose outputs
> differ from the submitted tables (see `scripts/DEPRECATED.md`). Paper numbers
> come from `zo_idep/` and are summarized in `zo_idep/results/EXECUTION_REPORT.md`.

All results below were generated inside this directory. Coordinate finite
differences are vectorized for speed, but each evaluated perturbation is counted
as a separate black-box query.

## Sparse linear regression

- Data: Gaussian design, noise standard deviation 0.1, planted amplitudes with
  random sign and magnitude uniform on `[0.25, 1]`.
- Optimizer: 300 iterations, `eta=0.9/L`, `mu=1e-3`, final `lambda=0.025`,
  geometric continuation during the first 100 iterations.
- Repetitions: 30 fixed seeds for each setting.
- Baselines: exact-gradient PG-EP, Full-ZO-EP, coordinate ZO-IHT, and the
  smooth-setting random-direction form of SZOHT.

| d | k | Full queries | ZO-IDEP queries (mean ± sd) | Mean switch | Saving |
|---:|---:|---:|---:|---:|---:|
| 256 | 16 | 153,600 | 58,992 ± 3,341 | 101.9 | 61.6% |
| 256 | 24 | 153,600 | 66,755 ± 7,186 | 111.8 | 56.5% |
| 256 | 32 | 153,600 | 80,964 ± 15,627 | 136.9 | 47.3% |
| 512 | 32 | 307,200 | 124,032 ± 21,205 | 108.2 | 59.6% |
| 1024 | 64 | 614,400 | 249,280 ± 28,674 | 108.8 | 59.4% |

In every run, the ZO-IDEP endpoint equals the Full-ZO-EP endpoint to numerical
precision. This is a stationarity-preservation result, not a planted-support
guarantee. At `(d,k)=(256,16)`, exact planted-support recovery is 73.3%; it falls
as the problem becomes more underdetermined.

Raw and aggregated data are in `results/regression_raw.csv`,
`results/regression_summary.csv`, and `results/traces.json`.

## Nonquadratic ridge-logistic stress test

- Data: 512 Gaussian covariates, dimension 128, planted sparsity 8, noiseless
  soft Bernoulli targets.
- Objective: logistic cross-entropy plus ridge coefficient 0.02.
- Optimizer: 500 iterations, `mu=0.01`, `lambda=0.02`, 100-step continuation.
- Repetitions: 30 fixed seeds.

ZO-IDEP certified at iteration 100 in all 30 runs. It used 32,240 queries versus
128,000 for Full-ZO-EP, a 74.8% reduction. Both methods had the same endpoint in
every run (reported endpoint gap exactly zero at saved precision), and both had
100% planted-support overlap. Mean objective was `0.629242 ± 0.007239` and mean
recovery error was `1.61108 ± 0.14073`.

This experiment tests nonlinear finite-difference dynamics but remains
deterministic and uses soft labels. It does not establish robustness to noisy
function evaluations. Raw data are in `results/nonlinear_logistic_raw.csv` and
the implementation is `scripts/run_nonlinear_experiment.py`.

## Reproduction

Canonical pipeline (paper tables and figures):

```bash
python3 zo_idep/experiments/exp01_sparse_regression.py --trials 30 --iterations 300
# exp02--exp07: see zo_idep/experiments/
python3 zo_idep/scripts/validate_results.py
make check
```

The legacy commands below reproduce this historical report only:

```bash
python scripts/run_experiments.py --trials 30 --iterations 300
python scripts/run_nonlinear_experiment.py --trials 30 --iterations 500
```
