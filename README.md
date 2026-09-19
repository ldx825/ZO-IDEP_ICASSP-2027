# Certified Zeroth-Order Exact Penalization for Black-Box Sparsity-Constrained Optimization (ZO-IDEP)

Official code and paper materials for the **ICASSP 2027** submission.

> **Code:** https://github.com/ldx825/ZO-IDEP_ICASSP-2027 · **Paper PDF:** [`ZO_IDEP_ICASSP2027/main.pdf`](ZO_IDEP_ICASSP2027/main.pdf)

**Authors:** Dongxu Liu, Dongyang Liu, Wenjia Shi, Keming Fan, Linhao Wang (Jilin University / UESTC)

## Repository layout

| Path | Contents |
|---|---|
| `ZO_IDEP_ICASSP2027/` | Paper LaTeX source, figures, compiled `main.pdf`, and the canonical experiment package `zo_idep/` (exp00–exp07 + `common/`; results under `results/`) |
| `verify_zo_idep_convex_certificate.py` | Standalone verifier for the convex invariant-support certificate (Thm. 3): old/new decision equivalence, proof inequalities, brute-force trajectory equality |
| `verify_zo_idep_derivations.py` | Independent verification of the derivation chain |
| `verify_zo_idep_logistic.py` | Independent verification for the ridge-logistic case |

Note: the legacy prototype pipeline under `ZO_IDEP_ICASSP2027/scripts/` is deprecated (see its `DEPRECATED.md`) — use the canonical package `zo_idep/`.

## Quick start

```bash
cd ZO_IDEP_ICASSP2027
python3 zo_idep/experiments/exp00_unit_tests.py
bash zo_idep/scripts/run_core_suite.sh
python3 zo_idep/experiments/exp01_sparse_regression.py --trials 30 --iterations 300
python3 zo_idep/scripts/validate_results.py     # summary & checks -> results/

# Standalone theorem verification (run from this repository's root)
python3 verify_zo_idep_convex_certificate.py
python3 verify_zo_idep_derivations.py
python3 verify_zo_idep_logistic.py
```

## Compiling the paper

```bash
cd ZO_IDEP_ICASSP2027 && make    # requires pdflatex + bibtex; `make check` for the pre-submission check
```

ICASSP template files are included (`spconf.sty`, `IEEEbib.bst`).

## Citation

```bibtex
@inproceedings{liu2027zoidep,
  title     = {Certified Zeroth-Order Exact Penalization for Black-Box Sparsity-Constrained Optimization},
  author    = {Liu, Dongxu and Liu, Dongyang and Shi, Wenjia and Fan, Keming and Wang, Linhao},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

## Contact

Dongxu Liu — liudx9924@mails.jlu.edu.cn
