# Citation audit

This file records the verification source for every reference used in `main.tex`.

## Important note on Google Scholar

The direct Google Scholar BibTeX endpoint was attempted from the current environment on 2026-09-08, but Google Scholar returned HTTP 429 (rate limited). I therefore did **not** pretend to have copied Scholar BibTeX. Every entry below was instead cross-checked against the primary publisher/proceedings page, DOI metadata, official author PDF where available, and/or DBLP. The `.bib` file contains only references whose title/authors/venue/year were verified from those sources.

## Verified references

### `zhao2026exact`
- Jianing Zhao, Junbin Liu, Anthony Man-Cho So, Wing-Kin Ma.
- *An Exact Penalty Method for Sparsity-Constrained Optimization*.
- ICASSP 2026, Sparse Optimization I oral session.
- DOI: `10.1109/ICASSP55912.2026.11464104`.
- Verified from the authors' CUHK PDF, ICASSP 2026 accepted-paper/session pages, and DOI metadata.
- Primary PDF: https://www1.se.cuhk.edu.hk/~manchoso/papers/pen_sparse-icassp26.pdf

### `yagishita2026exact`
- Shotaro Yagishita, Jun-ya Gotoh.
- *Exact Penalization at d-Stationary Points of Cardinality- or Rank-Constrained Problem*.
- Optimization, vol. 75, no. 10, pp. 2701-2735.
- DOI: `10.1080/02331934.2025.2573666`.
- Verified from Taylor & Francis and arXiv `2209.02315`.

### `devazelhes2022zeroth`
- William de Vazelhes, Hualin Zhang, Huimin Wu, Xiaotong Yuan, Bin Gu.
- *Zeroth-Order Hard-Thresholding: Gradient Error vs. Expansivity*.
- NeurIPS 2022, Advances in Neural Information Processing Systems 35, pp. 22589-22601.
- DOI: `10.52202/068431-1641`.
- Verified from the official NeurIPS proceedings page and DBLP.

### `cai2022zoro`
- HanQin Cai, Daniel McKenzie, Wotao Yin, Zhenliang Zhang.
- *Zeroth-Order Regularized Optimization (ZORO): Approximately Sparse Gradients and Adaptive Sampling*.
- SIAM Journal on Optimization 32(2):687-714, 2022.
- DOI: `10.1137/21M1392966`.
- Verified from SIAM, authors' PDF/GitHub citation, and DBLP.

### `kazemi2024efficient`
- Ehsan Kazemi, Liqiang Wang.
- *Efficient Zeroth-Order Proximal Stochastic Method for Nonconvex Nonsmooth Black-Box Problems*.
- Machine Learning 113(1):97-120, 2024.
- DOI: `10.1007/s10994-023-06409-7`.
- Verified from Springer and DBLP.

### `bonettini2023abstract`
- Silvia Bonettini, Peter Ochs, Marco Prato, Simone Rebegoldi.
- *An Abstract Convergence Framework with Application to Inertial Inexact Forward--Backward Methods*.
- Computational Optimization and Applications 84(2):319-362, 2023.
- DOI: `10.1007/s10589-022-00441-4`.
- Verified from Springer-linked institutional records and arXiv `2302.07545`.

### `attouch2013convergence`
- Hedy Attouch, Jerome Bolte, Benar Fux Svaiter.
- *Convergence of Descent Methods for Semi-Algebraic and Tame Problems: Proximal Algorithms, Forward--Backward Splitting, and Regularized Gauss--Seidel Methods*.
- Mathematical Programming 137(1-2):91-129, 2013.
- DOI: `10.1007/s10107-011-0484-9`.
- Verified from Springer metadata/secondary indexing and DBLP.

### `liang2017activity`
- Jingwei Liang, Jalal Fadili, Gabriel Peyre.
- *Activity Identification and Local Linear Convergence of Forward--Backward-Type Methods*.
- SIAM Journal on Optimization 27(1):408-437, 2017.
- DOI: `10.1137/16M106340X`.
- Verified from SIAM and DBLP.

### `dai2022inexact`
- Yutong Dai, Daniel P. Robinson.
- *Inexact Proximal-Gradient Methods with Support Identification*.
- arXiv:2211.02214, 2022.
- Verified from arXiv, Optimization Online, and the authors' publication page.

### `nesterov2017random`
- Yurii Nesterov, Vladimir Spokoiny.
- *Random Gradient-Free Minimization of Convex Functions*.
- Foundations of Computational Mathematics 17(2):527-566, 2017.
- DOI: `10.1007/s10208-015-9296-2`.
- Verified from UCLouvain repository and DBLP.

## Claims intentionally not made

- No claim of being the first zeroth-order sparse optimizer.
- No claim of being the first method with support identification.
- No claim that generic criticality implies sparsity for trimmed penalties.
- No claim that the certificate recovers the planted support or a global optimum.
- No claim that the fixed-`mu` finite-horizon experiments prove the asymptotic
  summable-error theorem.

## 2026-09-14 verification update (revision round)

Crossref / arXiv / PMLR re-verification performed for the working `refs.bib`.
All entries confirmed real with correct venues. Fixes applied to `refs.bib`:

- `lapucci2021convergent`: first author corrected Marco -> **Matteo** Lapucci
  (Crossref, DOI 10.1007/s10957-020-01793-9).
- `giuliani2022derivativefree`: first author corrected Camila M. -> **Caio
  Merlini** Giuliani (Crossref, DOI 10.1007/s40314-021-01748-4, vol. 41, art. 56).
- `kungurtsev2021zeroth`: re-added (Comput. Optim. Appl. 80(3):731-753, 2021,
  DOI 10.1007/s10589-021-00313-3); cited again in the restored
  "Relation to Prior Work" section.

Newly web-verified this round (Crossref unless noted):

- zhao2026exact (IEEE ICASSP 2026, Barcelona, DOI 10.1109/ICASSP55912.2026.11464104)
- yagishita2026exact (Optimization 75(10):2701-2735, DOI 10.1080/02331934.2025.2573666)
- brilli2026complexity (JOTA 209(1), art. 3, DOI 10.1007/s10957-026-02938-y;
  unusual DOI string but confirmed by Crossref)
- berahas2022comparison (FoCM 22(2):507-560, DOI 10.1007/s10208-021-09513-z)
- cai2021zobcd (ICML 2021, PMLR 139:1193-1203)
- cai2022zoro (SIAM J. Optim. 32(2):687-714, DOI 10.1137/21M1392966)
- devazelhes2022zeroth (NeurIPS 35, 22589-22601, DOI 10.52202/068431-1641)
- wang2026auc (AISTATS 2026, PMLR 300:3268-3276)
- kimiaei2026rzcw (arXiv:2606.28124, submitted 26 Jun 2026)
- kazemi2024efficient (Mach. Learn. 113(1):97-120, DOI 10.1007/s10994-023-06409-7)
- lapucci2021convergent, giuliani2022derivativefree, giovannelli2022mixed
  (Comput. Optim. Appl. 82:293-327, DOI 10.1007/s10589-022-00363-1)
- bonettini2023abstract (COAP 84(2):319-362, DOI 10.1007/s10589-022-00441-4)
- liang2017activity (SIAM J. Optim. 27(1):408-437, DOI 10.1137/16M106340X)
- dai2022inexact (arXiv:2211.02214)
- lewis2009active (SIAM J. Optim. 20(3):1378-1405; Crossref: online 2009-10-16,
  print 2010-01; the bib keeps year 2009, which matches the online date)
- tibshirani/fan/zhang/bertsimas/gotoh/attouch/nesterov/ghadimi/duchi/blumensath/
  chen2017zoo corroborated via DOI strings in the Crossref reference lists of
  the papers above.

Note: `zo_idep/src/certificate.py` still implements the pre-revision
(RSC-based) certificate test; see the theory-verification notes. For
least-squares settings (eps=0) that test coincides exactly with the current
convex certificate, so Table 1/2 numbers are unaffected.

