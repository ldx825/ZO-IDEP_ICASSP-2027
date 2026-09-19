#!/usr/bin/env python3
"""Independent verification, part 2: nonlinear (ridge-logistic) case with
eps = L mu / 2 > 0, mirroring the exp02 configuration but with fresh seeds.

Checks:
  1. Lemma 2.1 on the logistic: max |FD - grad| <= L mu / 2 (L = ||A||^2/4n + r).
  2. After a certified switch, brute-force FULL ZO proximal step equals the
     restricted step at every later iterate (no library code reused).
  3. Endpoint identity vs an independently implemented Full-ZO-EP.
"""
import numpy as np

RIDGE = 0.02
D, K = 128, 8


def make_problem(seed, m=512):
    r = np.random.default_rng(seed)
    A = r.normal(0.0, 1.0 / np.sqrt(K), size=(m, D))
    supp = np.sort(r.choice(D, size=K, replace=False))
    planted = np.zeros(D)
    planted[supp] = r.uniform(0.75, 1.5, K) * r.choice([-1.0, 1.0], K)
    scores = A @ planted
    y = 1.0 / (1.0 + np.exp(-scores))
    return A, y, planted, supp


def value(A, y, x):
    s = A @ x
    return float(np.mean(np.logaddexp(0.0, s) - y * s) + 0.5 * RIDGE * x @ x)


def grad(A, y, x):
    s = A @ x
    p = 1.0 / (1.0 + np.exp(-s))
    return A.T @ (p - y) / A.shape[0] + RIDGE * x


def fd_all(A, y, x, mu):
    s = A @ x
    fp_s = s[:, None] + mu * A
    fm_s = s[:, None] - mu * A
    n2 = x @ x
    fp = np.mean(np.logaddexp(0.0, fp_s) - y[:, None] * fp_s, axis=0) \
        + 0.5 * RIDGE * (n2 + 2 * mu * x + mu * mu)
    fm = np.mean(np.logaddexp(0.0, fm_s) - y[:, None] * fm_s, axis=0) \
        + 0.5 * RIDGE * (n2 - 2 * mu * x + mu * mu)
    return (fp - fm) / (2 * mu)


def fd_support(A, y, x, mu, S):
    s = A @ x
    cols = A[:, S]
    fp_s = s[:, None] + mu * cols
    fm_s = s[:, None] - mu * cols
    n2 = x @ x
    fp = np.mean(np.logaddexp(0.0, fp_s) - y[:, None] * fp_s, axis=0) \
        + 0.5 * RIDGE * (n2 + 2 * mu * x[S] + mu * mu)
    fm = np.mean(np.logaddexp(0.0, fm_s) - y[:, None] * fm_s, axis=0) \
        + 0.5 * RIDGE * (n2 - 2 * mu * x[S] + mu * mu)
    g = np.zeros_like(x)
    g[S] = (fp - fm) / (2 * mu)
    return g


def prox(z, eta, lam, k=K):
    order = np.lexsort((np.arange(z.size), -np.abs(z)))
    act = np.sort(order[:k])
    out = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0.0)
    out[act] = z[act]
    return out


print("== Lemma 2.1 on ridge-logistic (L = ||A||^2/(4n) + ridge) ==")
A, y, planted, supp = make_problem(55001)
L = float(np.linalg.norm(A, 2) ** 2 / (4 * A.shape[0]) + RIDGE)
print(f"  L = {L:.4f}, ridge m = {RIDGE}")
for mu in (1e-3, 1e-2, 3e-2):
    x = np.random.default_rng(7).normal(size=D)
    g = fd_all(A, y, x, mu)
    err = np.max(np.abs(g - grad(A, y, x)))
    print(f"  mu={mu:g}: max err={err:.3e} <= L mu/2={L*mu/2:.3e} "
          f"(ratio {err/(L*mu/2):.4f})")
print()

print("== fresh-seed certified logistic runs, brute-force equivalence ==")
eta = 0.1 / L
lam = 0.02
T = 800
tot_bad = 0
for seed in range(55001, 55006):
    A, y, planted, supp = make_problem(seed)
    # Full-ZO-EP (independent)
    xf = np.zeros(D)
    for t in range(T):
        ramp = min(1.0, (t + 1) / 100.0)
        lam_t = (lam / 40.0) * (40.0 ** ramp)
        xf = prox(xf - eta * fd_all(A, y, xf, 1e-3), eta, lam_t)
    # ZO-IDEP with brute-force audit
    x = np.zeros(D)
    S = None
    t_sw = None
    bad = 0
    for t in range(T):
        ramp = min(1.0, (t + 1) / 100.0)
        lam_t = (lam / 40.0) * (40.0 ** ramp)
        if S is None:
            g = fd_all(A, y, x, 1e-3)
            if t >= 99 and np.count_nonzero(np.abs(x) > 1e-10) == K:
                S_try = np.sort(np.flatnonzero(np.abs(x) > 1e-10))
                eps = L * 1e-3 / 2
                m = RIDGE
                q = np.sqrt(max(0.0, 1 - 2 * eta * m * L / (m + L)))
                R = (np.linalg.norm(g[S_try]) + np.sqrt(K) * eps) / m
                mask = np.ones(D, bool); mask[S_try] = False
                H = np.max(np.abs(g[mask])) + eps + 2 * L * R
                A_ = np.min(np.abs(x[S_try])) - 2 * R - eta * (L * R + eps)
                if q * R + eta * np.sqrt(K) * eps <= R and H + eps < lam \
                        and A_ > eta * (H + eps):
                    S = S_try; t_sw = t
            x_next = prox(x - eta * g, eta, lam_t)
        else:
            g = fd_support(A, y, x, 1e-3, S)
            x_next = np.zeros_like(x)
            x_next[S] = x[S] - eta * g[S]
            x_hyp = prox(x - eta * fd_all(A, y, x, 1e-3), eta, lam_t)
            if np.max(np.abs(x_hyp - x_next)) > 1e-10:
                bad += 1
        x = x_next
    gap = np.max(np.abs(x - xf))
    tot_bad += bad
    print(f"  seed={seed}: t_sw={t_sw} mismatches={bad} endpoint gap={gap:.2e}")
print(f"\nTOTAL mismatches: {tot_bad}")
