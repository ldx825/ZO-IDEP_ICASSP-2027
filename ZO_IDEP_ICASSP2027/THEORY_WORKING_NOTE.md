# ZO-IDEP Theory Checkpoint
## Zeroth-Order Exact-Penalty Optimization with Certified Support Identification
### ICASSP 2027 working theory note

> Status: theory interfaces A/B substantially closed under explicit assumptions.
> Baseline: Zhao et al., "An Exact Penalty Method for Sparsity-Constrained Optimization," ICASSP 2026.
> Current focus: vector cardinality constraint; group sparsity / low rank are future extensions.

---

# 1. Problem

We study

\[
\min_x f(x)\quad\text{s.t.}\quad \|x\|_0\le k,
\]

and use the trimmed-\(\ell_1\) exact penalty

\[
h_k(x)=\|x\|_1-\|x\|_{[k]},
\qquad
\Phi_\lambda(x)=f(x)+\lambda h_k(x).
\]

The ICASSP baseline uses a first-order nonconvex proximal-gradient method. We instead assume only function evaluations of \(f\).

---

# 2. Zeroth-order proximal iteration

Use coordinate central differences:

\[
\widehat g_{t,i}
=
\frac{f(x_t+\mu_t e_i)-f(x_t-\mu_t e_i)}{2\mu_t},
\]

then

\[
x_{t+1}
\in
\operatorname{Prox}_{\eta\lambda h_k}
\left(x_t-\eta\widehat g_t\right).
\]

The proximal subproblem is solved globally using the closed-form trimmed-\(\ell_1\) prox:
the \(k\) largest-magnitude entries of the prox input are left unchanged and the rest are soft-thresholded at level \(\eta\lambda\).

Define error

\[
e_t=\widehat g_t-\nabla f(x_t).
\]

For \(L\)-smooth \(f\),

\[
\|e_t\|_\infty\le \frac{L\mu_t}{2},
\qquad
\|e_t\|_2\le \frac{\sqrt d L\mu_t}{2}.
\]

---

# 3. Interface A: Global convergence of ZO exact-penalty proximal iterations

## Assumptions A

1. \(f\in C^1\) and \(\nabla f\) is \(L\)-Lipschitz.
2. \(\Phi_\lambda=f+\lambda h_k\) is bounded below.
3. The generated iterates remain bounded (e.g. coercivity / bounded level set).
4. \(\Phi_\lambda\) is a KL function.
5. \(0<\eta<1/L\).
6. \(\sum_t \mu_t<\infty\), hence \(\sum_t\|e_t\|<\infty\).
7. At each step the global trimmed-\(\ell_1\) proximal solution is used.

A clean deterministic schedule is, for example,

\[
\mu_t=\frac{\mu_0}{(t+1)^{1+\delta}},\qquad \delta>0.
\]

## Lemma A1: quasi-sufficient decrease

Let \(d_t=x_{t+1}-x_t\). Global optimality of the proximal subproblem gives

\[
\lambda h_k(x_{t+1})
+\langle \widehat g_t,d_t\rangle
+\frac{1}{2\eta}\|d_t\|^2
\le
\lambda h_k(x_t).
\]

Smoothness of \(f\) gives

\[
f(x_{t+1})
\le
f(x_t)
+\langle \nabla f(x_t),d_t\rangle
+\frac L2\|d_t\|^2.
\]

Hence

\[
\Phi_\lambda(x_{t+1})
\le
\Phi_\lambda(x_t)
-A\|d_t\|^2
-\langle e_t,d_t\rangle,
\]

where

\[
A=\frac{1}{2\eta}-\frac L2>0.
\]

Young's inequality yields

\[
\Phi_\lambda(x_{t+1})
\le
\Phi_\lambda(x_t)
-\frac A2\|d_t\|^2
+\frac{1}{2A}\|e_t\|^2.
\]

Thus the method satisfies an inexact sufficient-decrease condition with summable perturbation.

## Lemma A2: relative-error bound

Fermat's rule for the exact proximal subproblem gives a limiting subgradient
\(\xi_{t+1}\in\lambda\partial h_k(x_{t+1})\) satisfying

\[
\widehat g_t+\frac1\eta d_t+\xi_{t+1}=0.
\]

Then

\[
w_{t+1}
=
\nabla f(x_{t+1})+\xi_{t+1}
\in
\partial\Phi_\lambda(x_{t+1}),
\]

and

\[
\|w_{t+1}\|
\le
\left(L+\frac1\eta\right)\|d_t\|
+\|e_t\|.
\]

This is the perturbed relative-error condition used in abstract inexact KL convergence frameworks.

## Theorem A: full-sequence convergence

By applying an abstract KL convergence theorem for inexact descent methods with summable errors:

\[
\sum_t\|x_{t+1}-x_t\|<\infty,
\]

and the full sequence converges:

\[
x_t\to x^\star.
\]

Moreover,

\[
0\in\partial\Phi_\lambda(x^\star).
\]

The important strengthening comes next: because each update uses the **global proximal mapping**, the limit is more than an arbitrary critical point.

---

# 4. Proximal fixed-point limit

Let

\[
z_t=x_t-\eta\widehat g_t.
\]

Since \(x_t\to x^\star\), \(e_t\to0\), and \(\nabla f\) is continuous,

\[
z_t\to z^\star
=
x^\star-\eta\nabla f(x^\star).
\]

Each iterate satisfies

\[
x_{t+1}\in
\operatorname{Prox}_{\eta\lambda h_k}(z_t).
\]

Because \(h_k\) is continuous and the proximal objective is coercive, the graph of this global proximal map is closed. Passing to the limit gives

\[
\boxed{
x^\star
\in
\operatorname{Prox}_{\eta\lambda h_k}
\left(x^\star-\eta\nabla f(x^\star)\right).
}
\]

Call this **proximal stationarity / P-stationarity**.

---

# 5. Key new bridge: P-stationarity implies exact \(k\)-sparsity

Assume on the relevant bounded set

\[
\|\nabla f(x)\|_\infty\le M
\]

and choose

\[
\lambda>M.
\]

## Lemma A3: P-stationary exactness

If

\[
x^\star
\in
\operatorname{Prox}_{\eta\lambda h_k}
(x^\star-\eta\nabla f(x^\star)),
\]

then

\[
\boxed{\|x^\star\|_0\le k.}
\]

### Proof idea

Let

\[
z^\star=x^\star-\eta\nabla f(x^\star).
\]

The trimmed-\(\ell_1\) prox leaves the largest \(k\) entries of \(z^\star\) unchanged and soft-thresholds every tail entry by \(\eta\lambda\).

Suppose \(\|x^\star\|_0>k\). Then at least one nonzero output coordinate \(j\) must come from the soft-thresholded tail. Hence

\[
x_j^\star
=
z_j^\star-\eta\lambda\,\operatorname{sign}(z_j^\star)\ne0.
\]

But \(z_j^\star=x_j^\star-\eta\nabla_j f(x^\star)\), so

\[
\nabla_j f(x^\star)
=
-\lambda\,\operatorname{sign}(z_j^\star),
\]

which implies

\[
|\nabla_j f(x^\star)|=\lambda.
\]

This contradicts

\[
\|\nabla f(x^\star)\|_\infty\le M<\lambda.
\]

Therefore \(\|x^\star\|_0\le k\).

### Consequence

The convergence chain is now:

\[
\boxed{
\text{summably accurate ZO}
\Rightarrow
\text{KL convergence}
\Rightarrow
\text{P-stationary fixed point}
\Rightarrow
k\text{-sparse limit}.
}
\]

This avoids the known weakness that a generic critical point of a trimmed penalty can be dense.

---

# 6. Interface B1: local finite support identification

For clean support identification assume the nondegenerate case

\[
|S^\star|=k,
\qquad
S^\star=\operatorname{supp}(x^\star).
\]

Define

\[
z^\star
=
x^\star-\eta\nabla f(x^\star).
\]

Introduce two margins.

## Ranking margin

\[
\rho_\star
=
\min_{i\in S^\star}|z_i^\star|
-
\max_{j\notin S^\star}|z_j^\star|
>0.
\]

## Threshold margin

\[
\tau_\star
=
\eta\lambda
-
\max_{j\notin S^\star}|z_j^\star|
>0.
\]

Let

\[
m_\star
=
\min\{\rho_\star/2,\tau_\star\}.
\]

(The factor \(1/2\) is needed only on the ranking gap; the threshold gap
enters without it. Updated 2026-09-10 to match the paper's corrected
\(\delta_\star=\min\{\gamma_\star/2,\tau_\star\}\), where
\(\gamma_\star=\rho_\star\).)

## Lemma B1: proximal support stability

If a proximal input \(z\) satisfies

\[
\|z-z^\star\|_\infty<m_\star,
\]

then

\[
\operatorname{supp}
\left(
\operatorname{Prox}_{\eta\lambda h_k}(z)
\right)
=
S^\star.
\]

Reason:
1. the top-\(k\) ranking cannot flip because the perturbation is smaller than half the ranking gap;
2. all tail coordinates remain strictly below the soft-threshold \(\eta\lambda\).

---

# 7. ZO error enters the support-stability radius

The actual ZO proximal input is

\[
\widehat z_t=x_t-\eta\widehat g_t.
\]

Thus

\[
\widehat z_t-z^\star
=
(x_t-x^\star)
-\eta(\nabla f(x_t)-\nabla f(x^\star))
-\eta e_t.
\]

Therefore

\[
\|\widehat z_t-z^\star\|_\infty
\le
(1+\eta L)\|x_t-x^\star\|_2
+\eta\|e_t\|_\infty.
\]

For coordinate central differences,

\[
\|\widehat z_t-z^\star\|_\infty
\le
(1+\eta L)\|x_t-x^\star\|_2
+\frac{\eta L\mu_t}{2}.
\]

## Theorem B1: one-step ZO support identification

If

\[
(1+\eta L)\|x_t-x^\star\|_2
+\frac{\eta L\mu_t}{2}
<
m_\star,
\]

then

\[
\operatorname{supp}(x_{t+1})=S^\star.
\]

Since \(x_t\to x^\star\) and \(\mu_t\to0\), there exists a finite \(T_{\rm id}\) such that

\[
\boxed{
\operatorname{supp}(x_t)=S^\star,\qquad t\ge T_{\rm id},
}
\]

provided the subsequent iterates remain in the local invariant neighborhood. The next section closes that point.

---

# 8. Interface B2: permanent safe support restriction

A local identification result alone does not justify permanently throwing away all \(d-k\) coordinates. To obtain a **safe switch**, add a standard local curvature assumption.

## Assumptions B

Around the limiting solution \(x^\star\):

1. \(|S^\star|=k\).
2. \(f\) is \(C^2\).
3. The restricted Hessian satisfies

\[
mI
\preceq
\nabla^2_{S^\star S^\star}f(x)
\preceq
LI
\]

in a neighborhood, with \(m>0\).
4. Ranking and threshold margins \(\rho_\star,\tau_\star\) are positive.
5. The future ZO errors are sufficiently small (e.g. \(\mu_t\downarrow0\)).
6. Choose a restricted-gradient step \(\eta\le 1/L\).

Inside the identified manifold, the trimmed penalty is locally zero on the active coordinates and the tail remains zero. The full proximal update therefore reduces to an inexact gradient step on the active subspace:

\[
x_{t+1,S^\star}
=
x_{t,S^\star}
-\eta\widehat g_{t,S^\star},
\qquad
x_{t+1,S^{\star c}}=0.
\]

Restricted strong convexity makes this active-subspace map locally contractive. Hence there exists an invariant radius \(r>0\) and an admissible error level \(\bar\epsilon>0\) such that:

\[
\|x_t-x^\star\|\le r,\quad
\|e_t\|_\infty\le\bar\epsilon
\]

imply

\[
\|x_{t+1}-x^\star\|\le r
\]

and

\[
\operatorname{supp}(x_{t+1})=S^\star.
\]

## Theorem B2: safe permanent freeze

Once the iterates enter this invariant neighborhood, all later full-dimensional ZO-EP proximal steps keep the same support \(S^\star\). Therefore computing the \(d-k\) inactive coordinate finite differences becomes provably redundant.

At that point, replacing the full estimator by

\[
\widehat g_{t,S^\star}
\]

and setting all inactive coordinates to zero produces the same local active-manifold dynamics (up to the same active-coordinate ZO error).

This is the theoretical justification for permanently switching from \(d\)-dimensional to \(k\)-dimensional function queries.

---

# 9. Computable support certificate under restricted strong convexity

The theorem above uses the unknown \(x^\star\). We also want a conservative observable sufficient test.

Suppose the current point \(x\) is exactly \(k\)-sparse with support \(S\), and a full coordinate ZO estimate \(\widehat g\) has known error

\[
\|\widehat g-\nabla f(x)\|_\infty\le\epsilon.
\]

Assume \(f\) restricted to \(S\) is \(m\)-strongly convex.

Define

\[
R
=
\frac{\|\widehat g_S\|_2+\sqrt{k}\epsilon}{m}.
\]

Then the unique restricted minimizer \(x_S^\star\) satisfies

\[
\|x-x_S^\star\|_2\le R.
\]

Define an upper bound on inactive gradients at that minimizer:

\[
G
=
\|\widehat g_{S^c}\|_\infty
+\epsilon
+LR.
\]

Define a lower bound on its smallest active magnitude:

\[
A
=
\min_{i\in S}|x_i|-R.
\]

If

\[
G<\lambda
\]

and

\[
A>\eta G>0,
\]

then the restricted minimizer \(x_S^\star\) is also a full-space P-stationary point of the trimmed-\(\ell_1\) problem with support exactly \(S\):

- \(\nabla_S f(x_S^\star)=0\);
- inactive gradient magnitudes are below \(\lambda\), so tail coordinates are thresholded to zero;
- active magnitudes dominate inactive prox-input magnitudes, so Top-\(k\) selects \(S\).

For a **permanent** switch, use a stricter margin version of these inequalities so the ball of radius \(R\) lies inside the invariant support region.

This gives a conservative "certified freeze" mechanism using quantities available from the final full-dimensional ZO iteration plus known/estimated local constants \(m,L\).

---

# 10. Query complexity after certified identification

Before support identification, coordinate ZO uses

\[
2d
\]

function queries per iteration.

After certified support restriction, it uses

\[
2k.
\]

For \(T\) total iterations and identification time \(T_{\rm id}\),

\[
Q_{\rm adapt}
=
2dT_{\rm id}
+
2k(T-T_{\rm id}).
\]

Full-space coordinate ZO costs

\[
Q_{\rm full}=2dT.
\]

Therefore

\[
\frac{Q_{\rm adapt}}{Q_{\rm full}}
=
\frac{T_{\rm id}}{T}
+
\frac{k}{d}
\left(1-\frac{T_{\rm id}}T\right),
\]

and the exact query-saving fraction is

\[
\boxed{
\left(1-\frac{k}{d}\right)
\left(1-\frac{T_{\rm id}}T\right).
}
\]

Thus the gain increases with both sparsity \(k/d\ll1\) and earlier identification.

---

# 11. Current theorem chain

\[
\text{function values only}
\]

\[
\Downarrow
\]

\[
\text{coordinate ZO + exact nonconvex prox}
\]

\[
\Downarrow
\]

\[
\text{summable ZO error + KL}
\Rightarrow x_t\to x^\star
\]

\[
\Downarrow
\]

\[
x^\star\text{ is P-stationary}
\]

\[
\Downarrow\quad (\lambda>M)
\]

\[
\|x^\star\|_0\le k
\]

\[
\Downarrow\quad \text{nondegenerate }|S^\star|=k
\]

\[
\text{finite support identification}
\]

\[
\Downarrow\quad \text{local restricted curvature}
\]

\[
\text{safe permanent support freeze}
\]

\[
\Downarrow
\]

\[
2d\text{ queries/step}\to2k\text{ queries/step}.
\]

---

# 12. What is inherited vs what is potentially new

## Inherited machinery

- exact penalty / trimmed-\(\ell_1\) geometry: ICASSP 2026 and prior trimmed-lasso literature;
- d-stationary exactness: Yagishita & Gotoh;
- KL convergence framework: Attouch--Bolte--Svaiter and later inexact frameworks;
- finite active-manifold identification for first-order convex regularizers: Liang--Fadili--Peyré;
- inexact proximal support identification: Dai--Robinson;
- standard coordinate finite-difference error bound.

## Potentially new interface

1. convergence of function-value-only exact-penalty proximal iterations to a **P-stationary** sparse fixed point;
2. P-stationary exactness lemma for trimmed-\(\ell_1\) under \(\lambda>M\), used as the bridge from ZO convergence to exact sparsity;
3. explicit interaction between ZO estimation error and trimmed-\(\ell_1\) support margins;
4. a conservative ZO-query-based certificate for safe support restriction;
5. turning finite identification into an explicit black-box query reduction from \(d\) to \(k\).

Do not claim any of these as "first" before a second systematic literature audit.

---

# 13. Important caveats

1. The clean finite-identification theorem assumes the limiting solution has exactly \(k\) nonzeros. If it has fewer than \(k\), Top-\(k\) ties at zero require a separate treatment.
2. The permanent-freeze theorem uses local restricted strong convexity / positive curvature. Without it, finite identification may still hold, but a query-verifiable safe freeze is harder.
3. The current deterministic CoordZO theory assumes noiseless function evaluations. Stochastic/noisy black-box extensions need repeated queries or concentration arguments.
4. The summable smoothing schedule is chosen for a clean global convergence proof; practical fixed-\(\mu\) variants would converge only to a neighborhood unless separately analyzed.
5. The theory must distinguish "critical", "d-stationary", and "P-stationary"; they are not interchangeable.

---

# 14. Paper-template note

The supplied ICASSP template is the official two-column `spconf` style. It specifies an abstract of about 100--150 words and allows an additional final page containing references only. When drafting begins, the 4 technical pages should prioritize:

1. Introduction / gap;
2. problem + ZO-IDEP algorithm;
3. theorem chain (convergence, P-exactness, identification, query complexity);
4. compact experiments.

The support-identification/query-complexity chain should receive more space than broad applications.
