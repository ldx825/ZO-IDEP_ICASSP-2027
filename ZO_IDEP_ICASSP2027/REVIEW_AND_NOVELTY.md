# ZO-IDEP review, novelty, and positioning

Date of audit: 2026-09-09.

## Executive assessment

The paper has a coherent optimization story, but its defensible claim is narrower
than the original draft suggests. The strongest claim is not that the certificate
recovers the data-generating support. It is that, under explicit local curvature
and margin conditions, a support selected by the trimmed-l1 proximal dynamics can
be frozen without losing full-space proximal stationarity, reducing subsequent
coordinate finite-difference cost from `2d` to `2k`.

This is potentially publishable as an ICASSP-scale theory-and-algorithm paper if
the experiments demonstrate all three links: early certification, agreement with
the full-coordinate trajectory/final objective, and material end-to-end query
savings. Support recovery must be reported separately because a stationarity-safe
support can differ from the planted support.

## Motivation and innovation

The motivating gap is that exact sparsity penalties and black-box sparse
optimization currently solve different halves of the problem. Exact-penalty
proximal methods provide an interpretable sparse objective and a closed-form
trimmed-l1 prox, but assume gradients. Existing sparse ZO methods either impose
hard thresholding directly or assume approximately sparse gradients. ZO-IDEP
asks whether the support exposed by exact-penalty dynamics can itself reduce the
cost of the black-box oracle.

The potentially new interface consists of:

1. a function-value-only inexact proximal iteration for the trimmed-l1 exact
   penalty;
2. a bridge from global-prox fixed points to exact k-sparsity when the penalty
   dominates the relevant gradient bound;
3. an explicit finite-difference-error term in the support-identification margin;
4. a computable, local stationarity certificate for switching from full to
   support-restricted coordinate differences;
5. an auditable saving formula tied to the observed switch time.

The individual ingredients are not new. The contribution is their theorem-level
connection and empirical validation. Avoid a priority claim until a broader
bibliographic search is completed.

## Difference from the previous EviZO-VP topic

The prior EviZO-VP project studies query allocation for visual prompting of a
frozen model. Its objects are tasks, prompt groups, transfer evidence, and
accuracy-versus-query curves. ZO-IDEP instead studies generic cardinality-
constrained optimization. Its objects are exact penalties, proximal fixed points,
support margins, restricted curvature, and coordinate finite-difference oracles.

The shared phrase "query reduction" is not a scientific overlap: EviZO-VP
allocates a budget across tasks/prompt groups, while ZO-IDEP changes the oracle
dimension after identifying a sparse solution manifold. Code, experiments,
claims, and paper assets should remain separate.

## Collision audit

No paper found in the targeted search combines the same trimmed-l1 exact penalty,
ZO finite differences, finite support identification, and a certificate-driven
`2d -> 2k` switch. The closest lines are:

- Zhao et al., *An Exact Penalty Method for Sparsity-Constrained Optimization*,
  ICASSP 2026: direct exact-penalty/proximal starting point, but first-order only.
  https://www1.se.cuhk.edu.hk/~manchoso/papers/pen_sparse-icassp26.pdf
- de Vazelhes et al., *Zeroth-Order Hard-Thresholding*, NeurIPS 2022: direct
  black-box cardinality baseline, but hard-thresholds throughout and does not use
  exact-penalty identification as a switch certificate.
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/8de5384f522efff26884559599c09312-Abstract-Conference.html
- Cai et al., ZORO, SIAM J. Optim. 2022: reduces queries by approximately sparse
  gradients, not by an identified sparse solution support.
  https://epubs.siam.org/doi/10.1137/21M1392966
- Cartis et al., derivative-free active-set identification for linearly
  constrained optimization: establishes that derivative-free identification is
  not new in general, but addresses binding constraints rather than trimmed-l1
  support and the proposed query switch.
  https://epubs.siam.org/doi/10.1137/08073545X
- Giuliani et al., derivative-free exact-penalty trust-region optimization:
  establishes that the broad phrase "derivative-free exact penalty" is not new.
  https://research.ibm.com/publications/a-derivative-free-exact-penalty-algorithm-basic-ideas-convergence-theory-and-computational-studies

Therefore the title and abstract should emphasize **support-adaptive
exact-penalty ZO**, not claim the first derivative-free exact-penalty method.

## Proof audit and required corrections

1. **Coordinate central-difference bound:** correct under an L-Lipschitz
   gradient. The coordinate error is at most `L mu / 2`; the vector 2-norm bound
   must include `sqrt(d)` before switching and `sqrt(k)` afterward.
2. **KL convergence:** plausible but the paper must explicitly assume the
   summable-error KL hypotheses used by the cited inexact framework. A descent
   inequality plus `sum mu_t < infinity` should not be presented as a complete
   standalone proof without the continuity/relative-error conditions.
3. **Closed graph of the global prox:** valid because the trimmed-l1 penalty is
   continuous and the proximal objective is coercive. The prox is set-valued at
   ties, so all statements must use membership rather than equality.
4. **P-stationary sparsity lemma:** correct. If a fixed point had more than k
   nonzeros, some nonzero tail coordinate would imply
   `|grad_i f(x*)| = lambda`, contradicting a strict gradient bound.
5. **Finite identification:** correct only for an exactly k-sparse limit with
   strict top-k and threshold margins. The under-saturated case requires a
   separate set-valued analysis.
6. **Candidate certificate:** the inequalities certify that the unique
   restricted minimizer is a full-space P-stationary point. They do not certify
   that it is the planted support, a global minimizer, or the same support that a
   different initialization would find.
7. **Permanent switching:** requires an additional statement that restricted
   iterations remain in the curvature neighborhood and use summable ZO errors.
   The certificate alone proves the target point's stationarity, not invariance
   of every future iterate.
8. **Experimental continuation:** an annealed penalty can be used as a warm
   start, but the fixed-penalty theorem applies only after the final lambda is
   reached. This separation must be explicit.

## Go/no-go criterion

Proceed with the current story only if certified ZO-IDEP reaches essentially the
same endpoint as Full-ZO-EP while saving a substantial fraction of total queries
and rarely violating the claimed stationarity conditions. If the certificate
passes only at termination or repeatedly freezes poor supports without endpoint
agreement, pivot from "certified identification" to a diagnostic study of the
trade-off between stationarity safety and statistical support recovery.

## Go/no-go outcome after experiments

**GO, with the narrowed claim above.** Across five sparse-regression settings and
30 trials each, every certified run matched the Full-ZO-EP endpoint and saved
47.3--61.6% of queries. A 30-trial nonquadratic ridge-logistic check also matched
every full-method endpoint while saving 74.8%. The regression study simultaneously
shows why the claim must remain narrow: certification can preserve a locally
stationary support that is not the planted one. See `EXPERIMENT_REPORT.md` for the
protocols and aggregate statistics.
