"""Invariant-support certificate from the paper."""
from __future__ import annotations

import numpy as np


def certificate(x: np.ndarray, ghat: np.ndarray, support: np.ndarray,
                eta: float, lam: float, L: float, m: float, eps: float,
                numerical_tol: float = 1e-12):
    x = np.asarray(x, dtype=float)
    ghat = np.asarray(ghat, dtype=float)
    support = np.sort(np.asarray(support, dtype=int))
    if support.size == 0 or m <= 0 or L < m or eta <= 0:
        return False, {
            "R": float("inf"), "H": float("inf"), "A": -float("inf"),
            "q": float("nan"), "chi": -float("inf"),
            "margin_invariance": -float("inf"),
            "margin_threshold": -float("inf"),
            "margin_rank": -float("inf"), "strict": False,
        }
    radicand = 1.0 - 2.0 * eta * m * L / (m + L)
    if radicand < -numerical_tol:
        raise ValueError("certificate contraction factor is not real")
    q = float(np.sqrt(max(radicand, 0.0)))
    k = support.size
    inactive = np.setdiff1d(np.arange(x.size), support, assume_unique=True)
    R = float((np.linalg.norm(ghat[support]) + np.sqrt(k) * eps) / m)
    inactive_grad = float(np.max(np.abs(ghat[inactive]))) if inactive.size else 0.0
    H = float(inactive_grad + eps + 2.0 * L * R)
    A = float(np.min(np.abs(x[support])) - 2.0 * R - eta * (L * R + eps))
    margins = {
        "margin_invariance": float((1.0 - q) * R - eta * np.sqrt(k) * eps),
        "margin_threshold": float(lam - H - eps),
        "margin_rank": float(A - eta * (H + eps)),
    }
    chi = float(min(margins.values()))
    details = {"R": R, "H": H, "A": A, "q": q, "chi": chi,
               **margins, "strict": bool(chi > 0.0)}
    # Numerical guard only: conditions (c2)-(c3) of Theorem 3.3 are strict for
    # the mathematics, while (c1) is non-strict.  Reported certificates in the
    # paper pass with chi >= 1e-4, far above this tolerance.
    return bool(chi >= -numerical_tol), details
