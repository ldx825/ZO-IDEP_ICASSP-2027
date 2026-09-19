"""Offline theorem diagnostics; analytical gradients never drive ZO updates."""
from __future__ import annotations

import numpy as np


def stationarity_witness(objective, x: np.ndarray, x_next: np.ndarray,
                         ghat: np.ndarray, eta: float) -> float:
    step = x_next - x
    v_next = -ghat - step / eta
    witness = objective.gradient(x_next) + v_next
    return float(np.linalg.norm(witness))


def support_margin(objective, endpoint: np.ndarray, support: np.ndarray,
                   eta: float, lam: float):
    support = np.sort(np.asarray(support, dtype=int))
    inactive = np.setdiff1d(np.arange(endpoint.size), support, assume_unique=True)
    gradient = objective.gradient(endpoint)
    inactive_max = float(np.max(np.abs(gradient[inactive]))) if inactive.size else 0.0
    gamma = float(np.min(np.abs(endpoint[support])) - eta * inactive_max)
    tau = float(eta * (lam - inactive_max))
    # Theorem 3.2 margin: the ranking gap enters with a factor 1/2, the
    # soft-threshold gap does not; see the corrected (margins) definition.
    return {"gamma_star": gamma, "tau_star": tau,
            "delta_star": min(gamma / 2.0, tau),
            "inactive_gradient_max": inactive_max}


def prefix_stationarity(trace: list[dict], prefixes: list[int]):
    values = np.asarray([row["w_norm"] for row in trace], dtype=float)
    output = []
    for prefix in prefixes:
        used = values[:min(prefix, values.size)]
        output.append({"prefix": int(prefix), "min_w_norm": float(np.min(used)),
                       "avg_w_sq": float(np.mean(used ** 2)),
                       "residual_reference": float(prefix ** -0.5),
                       "squared_reference": float(prefix ** -1.0)})
    return output
