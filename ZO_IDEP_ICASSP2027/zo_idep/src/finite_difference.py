"""Auditable coordinate finite differences."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np


def coord_fd(f: Callable[[np.ndarray], float], x: np.ndarray, mu: float,
             coords: Sequence[int] | None = None,
             return_values: bool = False):
    """Central coordinate FD with exact query count and optional query log."""
    if mu <= 0:
        raise ValueError("mu must be positive")
    x = np.asarray(x, dtype=float)
    idx = np.arange(x.size, dtype=int) if coords is None else np.asarray(coords, dtype=int)
    estimate = np.zeros_like(x)
    audit = []
    for coordinate in idx:
        plus = x.copy()
        minus = x.copy()
        plus[coordinate] += mu
        minus[coordinate] -= mu
        f_plus = float(f(plus))
        f_minus = float(f(minus))
        estimate[coordinate] = (f_plus - f_minus) / (2.0 * mu)
        if return_values:
            audit.append({"coordinate": int(coordinate), "f_plus": f_plus,
                          "f_minus": f_minus})
    if return_values:
        return estimate, 2 * len(idx), audit
    return estimate, 2 * len(idx)
