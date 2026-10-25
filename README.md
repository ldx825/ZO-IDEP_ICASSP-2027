# Certified Zeroth-Order Exact Penalization for Black-Box Sparsity-Constrained Optimization (ZO-IDEP)

Code release for the paper **"Certified Zeroth-Order Exact Penalization for Black-Box Sparsity-Constrained Optimization"** (manuscript under review).

- Paper PDF: [`paper/main.pdf`](paper/main.pdf)
- Authors: Dongxu Liu, Dongyang Liu, Wenjia Shi, Keming Fan, Linhao Wang

This repository contains the reference implementation, the experiment package that produces the paper's tables and figures, and three standalone verifiers that re-derive the theory independently of the implementation.

---

## 1. Code overview

### 1.1 What the paper does

Exact-penalty methods turn the cardinality constraint `min f(x) s.t. ‖x‖₀ ≤ k` into a continuous objective through the trimmed-ℓ₁ penalty `λ h_k(·)`. Existing trimmed-ℓ₁ algorithms are first-order proximal methods; ZO-IDEP is the **function-value-only** realization, and it certifies when inactive-coordinate queries become permanently redundant.

* **Proximal map (Sec. 3, Eq. 2–3).** `P_η(z) = argmin_u {λ h_k(u) + ‖u−z‖²/2η}` has the closed form: keep the `k` largest-magnitude coordinates, soft-threshold the rest.
* **ZO gradient (Sec. 3, Eq. 4; Lemma 1).** Coordinate central differences `ĝ_i = (f(x+μe_i) − f(x−μe_i))/2μ` with `‖ĝ − ∇f‖_∞ ≤ Lμ/2`, i.e. `2d` queries per full step.
* **Certified support restriction (Sec. 4.2, Thm. 3).** From a valid localization radius `D_t` and error sequence `{ε_{t+s}}` the code forms `R_t = D_t + η√k·B_t`, `H_t = ‖ĝ_{S^c}‖_∞ + ε_t + 2LR_t`, `A_t = min_{i∈S}|x_i| − 2R_t − η(LR_t + ε̄_t)` and tests `H_t + ε̄_t < λ`, `A_t > η(H_t + ε̄_t)`. When it passes, every later iterate stays on `S` **and** the hypothetical full-coordinate proximal step returns exactly the same point, so `2d` queries per step collapse to `2k` (query ratio → `k/d`, Cor. 1).
* **Theory (Sec. 4.1).** Convergence to a sparse proximal fixed point, finite support identification, and a geometric post-switch rate under restricted strong convexity (Prop. 2).
* **Numerics (Sec. 5).** Exact post-switch step preservation, 58.3–62.5% finite-budget query savings, and the failure of heuristic "stable-support freeze" rules that the certificate replaces.

### 1.2 Where the method lives

| Module | Role |
|---|---|
| [`ZO_IDEP_ICASSP2027/zo_idep/src/prox.py`](ZO_IDEP_ICASSP2027/zo_idep/src/prox.py) | Deterministic global proximal map of the trimmed-ℓ₁ penalty (Eq. 2–3), with `stable_topk()` tie-breaking so "strict margins" are well defined. |
| [`ZO_IDEP_ICASSP2027/zo_idep/src/finite_difference.py`](ZO_IDEP_ICASSP2027/zo_idep/src/finite_difference.py) | Coordinate central differences with exact query counting and an optional query log (Eq. 4, Lemma 1). |
| [`ZO_IDEP_ICASSP2027/zo_idep/src/algorithms.py`](ZO_IDEP_ICASSP2027/zo_idep/src/algorithms.py) | Algorithm 1: full-coordinate phase, switch test, restricted phase, continuation/smoothing schedule, per-step query accounting. |
| [`ZO_IDEP_ICASSP2027/zo_idep/src/certificate.py`](ZO_IDEP_ICASSP2027/zo_idep/src/certificate.py) | Theorem 3 (invariant-support certificate) and the Prop. 2 strongly-convex specialization, returning every intermediate quantity (`R`, `H`, `A`, `q`, margins) for auditing. |
| [`ZO_IDEP_ICASSP2027/zo_idep/src/diagnostics.py`](ZO_IDEP_ICASSP2027/zo_idep/src/diagnostics.py) | Offline theorem diagnostics (stationarity witness, support margin). Analytical gradients are used **only** here, never to drive an update. |
| [`ZO_IDEP_ICASSP2027/zo_idep/src/objectives.py`](ZO_IDEP_ICASSP2027/zo_idep/src/objectives.py) | Synthetic smooth objectives (least squares, ridge-logistic, phase retrieval) with vectorized, auditable ZO oracles. |
| [`ZO_IDEP_ICASSP2027/zo_idep/src/utils.py`](ZO_IDEP_ICASSP2027/zo_idep/src/utils.py) | Seeding, result IO (raw records + aggregate JSON), config loading. |
| [`ZO_IDEP_ICASSP2027/zo_idep/experiments/common.py`](ZO_IDEP_ICASSP2027/zo_idep/experiments/common.py) | Shared runners used by exp01–exp07. |

### 1.3 Paper ↔ code correspondence

| Paper | Code |
|---|---|
| Sec. 3, Eq. (2)–(3) — trimmed-ℓ₁ proximal map | `zo_idep/src/prox.py` (`prox_trimmed_l1`, `stable_topk`) |
| Sec. 3, Eq. (4) / Lemma 1 — coordinate finite-difference estimator and its error | `zo_idep/src/finite_difference.py`; verified from scratch in [`verify_zo_idep_derivations.py`](verify_zo_idep_derivations.py) |
| Sec. 3, Algorithm 1 — full phase → switch → restricted phase | `zo_idep/src/algorithms.py` (with `zo_idep/src/certificate.py` as the switch test) |
| Sec. 4.1 — convergence to a sparse proximal fixed point, finite identification | `zo_idep/experiments/exp03_margin_sweep.py` (identification margin vs switch time), `exp02_mu_sweep.py` (estimator-error sweep) |
| Sec. 4.2, Thm. 3 — invariant-support certificate | `zo_idep/src/certificate.py`; independent check in [`verify_zo_idep_convex_certificate.py`](verify_zo_idep_convex_certificate.py) |
| Sec. 4.2, Prop. 2 / Cor. 1 — post-switch geometric rate and `Q_IDEP/Q_full → k/d` | `zo_idep/src/certificate.py` (strong-convex branch), `zo_idep/experiments/exp05_rate_and_accuracy.py` |
| Sec. 5 — main sparse-regression table with theorem diagnostics | `zo_idep/experiments/exp01_sparse_regression.py` |
| Sec. 5 — certified switch vs "stable-support freeze" heuristics | `zo_idep/experiments/exp04_certificate_vs_heuristic.py` |
| Sec. 5 — locally nonconvex (sparse phase retrieval) diagnostic | `zo_idep/experiments/exp06_sparse_phase_retrieval.py` |
| Sec. 5 — dimension scaling at fixed `k/d` | `zo_idep/experiments/exp07_scaling.py` |
| Nonlinear (ridge-logistic) instance, `ε = Lμ/2 > 0` | `zo_idep/experiments/exp02_mu_sweep.py`; [`verify_zo_idep_logistic.py`](verify_zo_idep_logistic.py) |
| Unit / oracle-accounting tests | `zo_idep/experiments/exp00_unit_tests.py` |

### 1.4 Repository layout

```
ZO_IDEP_ICASSP2027/
  zo_idep/
    src/           prox, finite differences, algorithm, certificate, objectives, diagnostics
    experiments/   exp00 unit tests + exp01-exp07 (paper tables and figures)
    scripts/       run_core_suite.sh, validate_results.py, build_paper_figures.py
    configs/       frozen protocol values (sparse regression, logistic, phase retrieval)
    results/       (not tracked) raw records, aggregate JSON, figures
verify_zo_idep_convex_certificate.py   standalone check of Thm. 3 vs the older test
verify_zo_idep_derivations.py          from-scratch re-derivation of Lemma 1 / Thm. 3
verify_zo_idep_logistic.py             same checks for the ridge-logistic case
paper/main.pdf                         compiled paper
```

---

## 2. Requirements

Pure NumPy (plus YAML for the config files); no GPU, no network and no pretrained model:

```bash
pip install numpy pyyaml
```

## 3. Running the experiments

Run everything from the package directory, because the scripts resolve `results/` relative to `ZO_IDEP_ICASSP2027/`:

```bash
cd ZO_IDEP_ICASSP2027

# unit tests + oracle accounting
python zo_idep/experiments/exp00_unit_tests.py

# full core suite: exp00 -> exp02 -> exp04 -> exp05 -> exp03 -> exp06 -> exp07 -> exp01
# then validates the frozen summaries and rebuilds the vector figures
bash zo_idep/scripts/run_core_suite.sh

# a single experiment, e.g. the main sparse-regression table
python zo_idep/experiments/exp01_sparse_regression.py --trials 30 --iterations 300
python zo_idep/experiments/exp01_sparse_regression.py --quick     # fast smoke run

# definition-of-done checks over the frozen summaries
python zo_idep/scripts/validate_results.py
```

Each experiment appends newline-delimited records (seed + configuration) to `zo_idep/results/raw/` and an aggregate JSON to `zo_idep/results/summary/`; `build_paper_figures.py` consumes only the frozen summaries. `results/` is not tracked in git, so run the suite once before calling the validator or the figure builder.

### Standalone theorem verification

These re-implement the paper's formulas from scratch (they do **not** import `zo_idep.src`) and can be run from the repository root:

```bash
python verify_zo_idep_derivations.py         # Lemma 1 sharpness, Thm. 3 equivalence on fresh seeds
python verify_zo_idep_convex_certificate.py  # Thm. 3 vs the older restricted-strong-convexity test
python verify_zo_idep_logistic.py            # ridge-logistic case, eps = L*mu/2 > 0
```

## 4. Notes and limitations

* Every method labelled ZO uses function values only. Analytical gradients are computed solely inside the offline diagnostics (`src/diagnostics.py`, verifiers) and never feed an update or a switch decision.
* Function-query totals exclude analytical-gradient diagnostics and shadow full-coordinate audits; that overhead is logged separately.
* The certificate consumes a valid localization radius `D_t` as data (compact level set, problem-specific bound, or the gradient-generated estimate in `certificate.py`).
* `exp06` is a *local*-theory diagnostic: its `controlled_local_topk` warm start perturbs the planted signal, so it tests preservation of the Full-ZO-EP endpoint and the post-certificate query reduction in a globally nonconvex objective. It is not end-to-end phase-retrieval performance.
* The nonlinear rate experiment floors the finite-difference radius at `1e-6` to avoid floating-point function-value cancellation.
* An earlier prototype pipeline is intentionally **not** part of this release: its switch test drops the `ε`, `2LR` and `2R` terms of Thm. 3 and is strictly weaker.

## 5. Citation

```bibtex
@misc{liu2026zoidep,
  title  = {Certified Zeroth-Order Exact Penalization for Black-Box Sparsity-Constrained Optimization},
  author = {Liu, Dongxu and Liu, Dongyang and Shi, Wenjia and Fan, Keming and Wang, Linhao},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 6. Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
