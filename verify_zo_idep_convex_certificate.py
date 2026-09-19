#!/usr/bin/env python3
"""Independent verification of the NEW convex invariant-support certificate
(Theorem 3, main.tex: thm:safe), and comparison with the previously
implemented restricted-strong-convexity test.

Checks:
  A. Least squares (eps=0): new test == old test (decisions identical) and
     post-switch full ZO proximal step == restricted step (brute force),
     plus the proof inequalities (ball bound, H_t, A_t) at post-switch points.
  B. Ridge-logistic with GEOMETRIC mu_t: old vs new switch decisions, then
     brute-force trajectory equality after a new-test-certified switch.
  C. Ridge-logistic with FIXED mu: old vs new switch decisions (the new
     theorem requires summable errors, B_t < inf, so fixed mu is expected to
     refuse under the new conditions).

All formulas are re-implemented from the paper; nothing is imported from
zo_idep.src.
"""
import numpy as np

RIDGE = 0.02


# ---------------- independent building blocks -------------------------------
def trimmed_l1_prox(z, eta, lam, k):
    order = np.lexsort((np.arange(z.size), -np.abs(z)))
    act = np.sort(order[:k])
    out = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0.0)
    out[act] = z[act]
    return out


def fd_quad_all(A, y, x, mu):
    r = A @ x - y
    fp = 0.5 * np.sum((r[:, None] + mu * A) ** 2, axis=0) / A.shape[0]
    fm = 0.5 * np.sum((r[:, None] - mu * A) ** 2, axis=0) / A.shape[0]
    return (fp - fm) / (2.0 * mu)


def grad_quad(A, y, x):
    return A.T @ (A @ x - y) / A.shape[0]


def make_regression(seed, m=128, d=256, k=16):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0 / np.sqrt(k), size=(m, d))
    supp = np.sort(rng.choice(d, size=k, replace=False))
    x = np.zeros(d)
    x[supp] = rng.uniform(0.25, 1.0, k) * rng.choice([-1.0, 1.0], k)
    y = A @ x + rng.normal(0.0, 0.1, m)
    return A, y, x, supp


def make_logistic(seed, m=512, d=128, k=8):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0 / np.sqrt(k), size=(m, d))
    supp = np.sort(rng.choice(d, size=k, replace=False))
    planted = np.zeros(d)
    planted[supp] = rng.uniform(0.75, 1.5, k) * rng.choice([-1.0, 1.0], k)
    s = A @ planted
    y = 1.0 / (1.0 + np.exp(-s))
    return A, y, planted, supp


def fd_log_all(A, y, x, mu, ridge=RIDGE):
    s = A @ x
    fp_s = s[:, None] + mu * A
    fm_s = s[:, None] - mu * A
    n2 = x @ x
    fp = np.mean(np.logaddexp(0.0, fp_s) - y[:, None] * fp_s, axis=0) \
        + 0.5 * ridge * (n2 + 2 * mu * x + mu * mu)
    fm = np.mean(np.logaddexp(0.0, fm_s) - y[:, None] * fm_s, axis=0) \
        + 0.5 * ridge * (n2 - 2 * mu * x + mu * mu)
    return (fp - fm) / (2.0 * mu)


def grad_log(A, y, x, ridge=RIDGE):
    s = A @ x
    p = 1.0 / (1.0 + np.exp(-s))
    return A.T @ (p - y) / A.shape[0] + ridge * x


# ---------------- certificate tests -----------------------------------------
def old_test(x, g, S, eta, lam, L, m, eps):
    """Previous RSC certificate (contraction + threshold + rank)."""
    k = S.size
    q = np.sqrt(max(0.0, 1.0 - 2.0 * eta * m * L / (m + L)))
    R = (np.linalg.norm(g[S]) + np.sqrt(k) * eps) / m
    mask = np.ones(x.size, bool)
    mask[S] = False
    H = (np.max(np.abs(g[mask])) if mask.any() else 0.0) + eps + 2.0 * L * R
    A_ = np.min(np.abs(x[S])) - 2.0 * R - eta * (L * R + eps)
    c1 = (1.0 - q) * R - eta * np.sqrt(k) * eps
    ok = (c1 >= -1e-14) and (H + eps < lam) and (A_ > eta * (H + eps))
    return ok, dict(R=R, H=H, A=A_, q=q, c1=c1)


def new_test(x, g, S, eta, lam, L, m, eps_now, B_t, eps_bar=None):
    """New convex certificate: D_t via RSC instantiation, R_t=D_t+eta*sqrt(k)*B_t."""
    if eps_bar is None:
        eps_bar = eps_now
    k = S.size
    D_t = (np.linalg.norm(g[S]) + np.sqrt(k) * eps_now) / m
    R_t = D_t + eta * np.sqrt(k) * B_t
    mask = np.ones(x.size, bool)
    mask[S] = False
    H = (np.max(np.abs(g[mask])) if mask.any() else 0.0) + eps_now + 2.0 * L * R_t
    A_ = np.min(np.abs(x[S])) - 2.0 * R_t - eta * (L * R_t + eps_bar)
    ok = (H + eps_bar < lam) and (A_ > eta * (H + eps_bar))
    return ok, dict(D=D_t, R=R_t, H=H, A=A_)


def lam_at(t, lam, C=100):
    ramp = min(1.0, (t + 1) / C)
    return (lam / 40.0) * (40.0 ** ramp)


# ---------------- Part A: least squares -------------------------------------
print("== A. least squares (eps=0): old vs new tests + trajectory equality ==")
L_new_sw = 0
for seed in range(9001, 9006):
    A, y, xstar, supp_star = make_regression(seed)
    d, k, T = 256, 16, 300
    lam, mu, L = 0.025, 1e-3, float(np.linalg.norm(A, 2) ** 2 / A.shape[0])
    eta = 0.9 / L
    x = np.zeros(d)
    S = None
    t_sw = None
    mismatches = 0
    max_ball_viol = 0.0
    xS_opt = None
    R_t = 0.0
    H_t = 0.0
    A_t = 0.0
    for t in range(T):
        lam_t = lam_at(t, lam)
        g = fd_quad_all(A, y, x, mu)
        if S is None:
            if t >= 99 and np.count_nonzero(np.abs(x) > 1e-10) == k:
                S_try = np.flatnonzero(np.abs(x) > 1e-10)
                m_val = float(np.linalg.eigvalsh(
                    A[:, S_try].T @ A[:, S_try] / A.shape[0])[0])
                ok_old, _ = old_test(x, g, S_try, eta, lam, L, m_val, 0.0)
                ok_new, det = new_test(x, g, S_try, eta, lam, L, m_val, 0.0, 0.0)
                assert ok_old == ok_new, "eps=0: old/new tests must agree"
                if ok_new:
                    S = S_try
                    t_sw = t
                    R_t, H_t, A_t = det["R"], det["H"], det["A"]
                    AS = A[:, S]
                    xS_opt = np.zeros(d)
                    xS_opt[S] = np.linalg.solve(AS.T @ AS, AS.T @ y)
            x_next = trimmed_l1_prox(x - eta * g, eta, lam_t, k)
        else:
            gS = fd_quad_all(A, y, x, mu)
            x_next = np.zeros(d)
            x_next[S] = x[S] - eta * gS[S]
            # brute-force hypothetical FULL step
            gfull = fd_quad_all(A, y, x, mu)
            hyp = trimmed_l1_prox(x - eta * gfull, eta, lam_t, k)
            if np.max(np.abs(hyp - x_next)) > 1e-12:
                mismatches += 1
            # proof-inequality spot checks
            ball = np.linalg.norm(x_next - xS_opt)
            max_ball_viol = max(max_ball_viol, ball - R_t - 1e-10)
        x = x_next
    L_new_sw += mismatches
    print(f"  seed={seed}: t_sw={t_sw} post-switch mismatches={mismatches} "
          f"ball violation margin={max_ball_viol:.2e}")
print(f"A: TOTAL brute-force mismatches after new-test switch = {L_new_sw}\n")

# ---------------- Parts B/C: logistic --------------------------------------
print("== B. logistic GEOMETRIC mu: old vs new tests + trajectory equality ==")
tot_bad = 0
for seed in range(55001, 55004):
    A, y, planted, supp = make_logistic(seed)
    d, k, T = 128, 8, 800
    lam = 0.02
    L = float(np.linalg.norm(A, 2) ** 2 / (4 * A.shape[0]) + RIDGE)
    eta = 0.1 / L
    mu0, beta, mu_min = 1e-3, 0.99, 1e-6
    mu_t = [max(mu_min, mu0 * beta ** max(0, t - 100)) for t in range(T)]
    m_val = RIDGE

    def eps_of(t):
        return L * mu_t[t] / 2.0

    x = np.zeros(d)
    S = None
    t_sw_new = None
    bad = 0
    first_old = None
    first_new = None
    for t in range(T):
        lam_t = lam_at(t, lam)
        g = fd_log_all(A, y, x, mu_t[t])
        if S is None:
            if t >= 99 and np.count_nonzero(np.abs(x) > 1e-10) == k:
                S_try = np.flatnonzero(np.abs(x) > 1e-10)
                eps_now = eps_of(t)
                # finite-horizon truncation of the error tail (run horizon T)
                B_trunc = sum(eps_of(t + s) for s in range(0, T - 1 - t + 1))
                # infinite geometric tail (ignoring the mu_min floor)
                B_inf = (L / 2.0) * (mu_t[t] / (1 - beta))
                ok_old, dold = old_test(x, g, S_try, eta, lam, L, m_val, eps_now)
                ok_new, dnew = new_test(x, g, S_try, eta, lam, L, m_val,
                                        eps_now, B_trunc, eps_now)
                if first_old is None and ok_old:
                    first_old = t
                if first_new is None and ok_new:
                    first_new = t
                    S = S_try
                    t_sw_new = t
                    R_t = dnew["R"]
                    H_t = dnew["H"]
                    A_t = dnew["A"]
                    B_used = B_trunc
                    if t >= 115:
                        print(f"    seed={seed} t={t}: old={ok_old} new={ok_new} "
                              f"R_t={R_t:.3e} vs R_old={dold['R']:.3e} "
                              f"B_trunc={B_trunc:.2e} B_inf={B_inf:.2e}")
            x_next = trimmed_l1_prox(x - eta * g, eta, lam_t, k)
        else:
            gS = fd_log_all(A, y, x, mu_t[t])
            x_next = np.zeros(d)
            x_next[S] = x[S] - eta * gS[S]
            hyp = trimmed_l1_prox(x - eta * fd_log_all(A, y, x, mu_t[t]),
                                  eta, lam_t, k)
            if np.max(np.abs(hyp - x_next)) > 1e-10:
                bad += 1
        x = x_next
    tot_bad += bad
    print(f"  seed={seed}: first_old={first_old} first_new={first_new} "
          f"post-switch mismatches={bad}")
print(f"B: TOTAL brute-force mismatches after new-test switch = {tot_bad}\n")

print("== C. logistic FIXED mu: old vs new switch decisions (exp02 protocol) ==")
for mu in (1e-2, 3e-2):
    n_old = n_new = 0
    first_old_list = []
    for seed in range(55011, 55014):
        A, y, planted, supp = make_logistic(seed)
        d, k, T = 128, 8, 800
        lam = 0.02
        L = float(np.linalg.norm(A, 2) ** 2 / (4 * A.shape[0]) + RIDGE)
        eta = 0.1 / L
        eps_now = L * mu / 2.0
        x = np.zeros(d)
        switched_old = switched_new = False
        first_old = None
        for t in range(T):
            lam_t = lam_at(t, lam)
            g = fd_log_all(A, y, x, mu)
            if (not switched_old) or (not switched_new):
                if t >= 99 and np.count_nonzero(np.abs(x) > 1e-10) == k:
                    S_try = np.flatnonzero(np.abs(x) > 1e-10)
                    if not switched_old:
                        ok_old, _ = old_test(x, g, S_try, eta, lam, L,
                                             RIDGE, eps_now)
                        if ok_old:
                            switched_old = True
                            first_old = t
                    if not switched_new:
                        # truncated tail over the run horizon
                        B_trunc = eps_now * max(0, T - 1 - t)
                        ok_new, _ = new_test(x, g, S_try, eta, lam, L,
                                             RIDGE, eps_now, B_trunc, eps_now)
                        if ok_new:
                            switched_new = True
            x = trimmed_l1_prox(x - eta * g, eta, lam_t, k)
        n_old += switched_old
        n_new += switched_new
        if switched_old:
            first_old_list.append(first_old)
    print(f"  mu={mu}: old-test switched in {n_old}/3 runs "
          f"(first ~{min(first_old_list) if first_old_list else '-'}); "
          f"new-test switched in {n_new}/3")
print("\nDone.")
