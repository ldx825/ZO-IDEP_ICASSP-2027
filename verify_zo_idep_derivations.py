#!/usr/bin/env python3
"""Independent verification of ZO-IDEP derivations.

Re-implements the paper formulas from scratch (no import of zo_idep.src
algorithm/certificate code) and checks:

1. Lemma 2.1: |FD error| <= L mu / 2, and the bound is attained (sharp).
2. Theorem 3.3 equivalence on fresh seeds: after a certified switch, the
   hypothetical FULL-coordinate ZO proximal step equals the restricted step
   at every later iterate (checked by brute force, no shadow-audit shortcut).
3. Endpoint identity vs an independently implemented Full-ZO-EP.
4. Query count identity: total = 2d*(t_sw+1) + 2k*(T - t_sw - 1).
"""
import numpy as np

rng = np.random.default_rng(12345)


# ---------------- independent building blocks (from the paper) ------------
def trimmed_l1_prox(z, eta, lam, k):
    order = np.lexsort((np.arange(z.size), -np.abs(z)))
    active = np.sort(order[:k])
    out = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0.0)
    out[active] = z[active]
    return out, active


def coord_fd_all(A, y, x, mu):
    r = A @ x - y
    fp = 0.5 * np.sum((r[:, None] + mu * A) ** 2, axis=0) / A.shape[0]
    fm = 0.5 * np.sum((r[:, None] - mu * A) ** 2, axis=0) / A.shape[0]
    return (fp - fm) / (2.0 * mu), 2 * x.size


def coord_fd_support(A, y, x, mu, S):
    r = A @ x - y
    cols = A[:, S]
    fp = 0.5 * np.sum((r[:, None] + mu * cols) ** 2, axis=0) / A.shape[0]
    fm = 0.5 * np.sum((r[:, None] - mu * cols) ** 2, axis=0) / A.shape[0]
    g = np.zeros_like(x)
    g[S] = (fp - fm) / (2.0 * mu)
    return g, 2 * S.size


def cert_conditions(x, ghat, S, eta, lam, L, mu):
    """Exact paper formulas (certdef)/(certconds), quadratic => eps = L mu/2? No:
    for least squares the central difference is exact, so eps = 0 (the paper's
    zero_quadratic mode)."""
    eps = 0.0
    m = np.linalg.eigvalsh(A[:, S].T @ A[:, S] / A.shape[0])[0]
    q = np.sqrt(max(0.0, 1.0 - 2 * eta * m * L / (m + L)))
    R = (np.linalg.norm(ghat[S]) + np.sqrt(S.size) * eps) / m
    mask = np.ones(x.size, bool); mask[S] = False
    H = np.max(np.abs(ghat[mask])) + eps + 2 * L * R
    A_ = np.min(np.abs(x[S])) - 2 * R - eta * (L * R + eps)
    ok = (q * R + eta * np.sqrt(S.size) * eps <= R) and (H + eps < lam) \
         and (A_ > eta * (H + eps))
    return ok, q, R, H, A_


def make_data(seed, m, d, k):
    r = np.random.default_rng(seed)
    A = r.normal(0.0, 1.0 / np.sqrt(k), size=(m, d))
    supp = np.sort(r.choice(d, size=k, replace=False))
    xstar = np.zeros(d)
    xstar[supp] = r.uniform(0.25, 1.0, k) * r.choice([-1.0, 1.0], k)
    y = A @ xstar + r.normal(0.0, 0.1, m)
    return A, y, xstar, supp


def obj(A, y, x):
    r = A @ x - y
    return 0.5 * float(r @ r) / A.shape[0]


# ------------------- test 1: Lemma 2.1 bound & sharpness ------------------
print("== Lemma 2.1: bound validity and sharpness ==")
for mu in (1e-2, 1e-3):
    Aq = np.sqrt(2.0) * np.eye(4)          # L = ||A||^2/n = 2
    L = 2.0
    xq = rng.normal(size=4)
    g, _ = coord_fd_all(Aq, np.zeros(4), xq, mu)
    exact = Aq.T @ (Aq @ xq) / Aq.shape[0]
    print(f"  mu={mu:g}: max|FD-grad|={np.max(np.abs(g - exact)):.2e} "
          f"<= L mu/2={L*mu/2:.2e} (slack {L*mu/2 - np.max(np.abs(g-exact)):.2e})")

# sharp example: f(x)= (L/2) x|x| + g x in 1-D, error exactly L mu/2
L, g0, mu = 2.0, 0.7, 1e-3
f = lambda t: 0.5 * L * t * abs(t) + g0 * t
err = abs((f(mu) - f(-mu)) / (2 * mu) - g0)
print(f"  sharpness example: |error|={err:.6e} == L mu/2={L*mu/2:.6e} "
      f"(ratio {err/(L*mu/2):.6f})")
print()

# ------------------- tests 2-4: fresh-seed certified runs -----------------
print("== Theorem 3.3 fresh-seed verification (independent implementation) ==")
count_mismatch = 0
for seed in range(31337, 31347):
    A, y, xstar, supp_star = make_data(seed, 128, 256, 16)
    d = 256; k = 16; T = 300; lam = 0.025; mu = 1e-3
    L = float(np.linalg.norm(A, 2) ** 2 / A.shape[0])
    eta = 0.9 / L

    # --- Full-ZO-EP (independent) ---
    xf = np.zeros(d); q_full = 0
    for t in range(T):
        ramp = min(1.0, (t + 1) / 100.0)
        lam_t = (lam / 40.0) * (40.0 ** ramp)
        g, c = coord_fd_all(A, y, xf, mu); q_full += c
        xf, _ = trimmed_l1_prox(xf - eta * g, eta, lam_t, k)
    # --- ZO-IDEP with brute-force equivalence check after switch ---
    x = np.zeros(d); q = 0; S = None; t_sw = None; bad = 0
    for t in range(T):
        ramp = min(1.0, (t + 1) / 100.0)
        lam_t = (lam / 40.0) * (40.0 ** ramp)
        if S is None:
            g, c = coord_fd_all(A, y, x, mu); q += c
            if t >= 99 and np.count_nonzero(np.abs(x) > 1e-10) == k:
                ok, qq, R, H, A_ = cert_conditions(x, g, np.flatnonzero(np.abs(x) > 1e-10), eta, lam, L, mu)
                if ok:
                    S = np.sort(np.flatnonzero(np.abs(x) > 1e-10)); t_sw = t
            x_next, _ = trimmed_l1_prox(x - eta * g, eta, lam_t, k)
        else:
            g, c = coord_fd_support(A, y, x, mu, S); q += c
            x_next = np.zeros_like(x)
            x_next[S] = x[S] - eta * g[S]
            # brute-force hypothetical FULL step at x (fresh full FD)
            gf, _ = coord_fd_all(A, y, x, mu)
            x_hyp, _ = trimmed_l1_prox(x - eta * gf, eta, lam_t, k)
            if not np.array_equal(np.sort(np.flatnonzero(np.abs(x_hyp) > 0)), S) or \
               np.max(np.abs(x_hyp - x_next)) > 1e-12:
                bad += 1
        x = x_next
    gap = np.max(np.abs(x - xf))
    expected_q = 2 * d * (t_sw + 1) + 2 * k * (T - t_sw - 1) if t_sw is not None else 2 * d * T
    count_mismatch += bad
    print(f"  seed={seed}: t_sw={t_sw} full-step mismatches after switch={bad} "
          f"endpoint gap vs Full-ZO-EP={gap:.2e} queries={q} "
          f"(formula {expected_q}, match={q == expected_q})")
print(f"\nTOTAL brute-force full-step mismatches after certified switch: {count_mismatch}")
