"""Deterministic global proximal map for the trimmed-l1 penalty."""
from __future__ import annotations

import itertools
import numpy as np


def stable_topk(z: np.ndarray, k: int) -> np.ndarray:
    """Return top-k magnitude indices, breaking ties by smaller index."""
    z = np.asarray(z)
    if not 0 <= k <= z.size:
        raise ValueError(f"k={k} must lie in [0,{z.size}]")
    if k == 0:
        return np.empty(0, dtype=int)
    order = np.lexsort((np.arange(z.size), -np.abs(z)))
    return np.sort(order[:k])


def prox_trimmed_l1(z: np.ndarray, eta: float, lam: float,
                    k: int) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic global trimmed-l1 proximal solution."""
    if eta <= 0 or lam < 0:
        raise ValueError("eta must be positive and lambda nonnegative")
    z = np.asarray(z, dtype=float)
    active = stable_topk(z, k)
    out = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0.0)
    out[active] = z[active]
    return out, active


def trimmed_l1(x: np.ndarray, k: int) -> float:
    return float(np.sum(np.abs(x)) - np.sum(np.abs(x[stable_topk(x, k)])))


def prox_objective(u: np.ndarray, z: np.ndarray, eta: float,
                   lam: float, k: int) -> float:
    return lam * trimmed_l1(u, k) + float(np.sum((u - z) ** 2)) / (2.0 * eta)


def brute_force_prox_candidates(z: np.ndarray, eta: float, lam: float,
                                k: int) -> list[np.ndarray]:
    """Enumerate protected-set candidates for small-dimensional tests."""
    z = np.asarray(z, dtype=float)
    candidates = []
    for protected in itertools.combinations(range(z.size), k):
        u = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0.0)
        u[list(protected)] = z[list(protected)]
        candidates.append(u)
    return candidates
