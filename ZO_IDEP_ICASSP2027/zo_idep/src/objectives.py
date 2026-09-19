"""Synthetic smooth objectives with vectorized, auditable ZO oracles."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclass
class LeastSquares:
    A: np.ndarray
    y: np.ndarray
    name: str = "sparse_regression"
    L: float = field(init=False)

    def __post_init__(self):
        self.L = float(np.linalg.norm(self.A, 2) ** 2 / self.A.shape[0])

    @property
    def dimension(self):
        return self.A.shape[1]

    def value(self, x):
        residual = self.A @ x - self.y
        return 0.5 * float(residual @ residual) / self.A.shape[0]

    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.y) / self.A.shape[0]

    def coord_fd(self, x, mu, coords=None, audit=False):
        idx = np.arange(x.size) if coords is None else np.asarray(coords, dtype=int)
        residual = self.A @ x - self.y
        columns = self.A[:, idx]
        plus = residual[:, None] + mu * columns
        minus = residual[:, None] - mu * columns
        f_plus = 0.5 * np.sum(plus * plus, axis=0) / self.A.shape[0]
        f_minus = 0.5 * np.sum(minus * minus, axis=0) / self.A.shape[0]
        estimate = np.zeros_like(x)
        estimate[idx] = (f_plus - f_minus) / (2.0 * mu)
        values = None
        if audit:
            values = [{"coordinate": int(i), "f_plus": float(fp), "f_minus": float(fm)}
                      for i, fp, fm in zip(idx, f_plus, f_minus)]
        return estimate, 2 * idx.size, values

    def restricted_m(self, support, x=None):
        columns = self.A[:, np.asarray(support, dtype=int)]
        return float(np.linalg.eigvalsh(columns.T @ columns / self.A.shape[0])[0])

    def random_direction_fd(self, x, mu, q, rng):
        d = x.size
        directions = rng.normal(size=(q, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        residual = self.A @ x - self.y
        shifted = residual[:, None] + mu * (self.A @ directions.T)
        f_plus = 0.5 * np.sum(shifted * shifted, axis=0) / self.A.shape[0]
        f_base = 0.5 * float(residual @ residual) / self.A.shape[0]
        estimate = (d / (q * mu)) * (directions.T @ (f_plus - f_base))
        return estimate, q + 1


@dataclass
class RidgeLogistic:
    A: np.ndarray
    y: np.ndarray
    ridge: float
    name: str = "ridge_logistic"
    L: float = field(init=False)

    def __post_init__(self):
        self.L = float(np.linalg.norm(self.A, 2) ** 2 / (4.0 * self.A.shape[0]) + self.ridge)

    @property
    def dimension(self):
        return self.A.shape[1]

    def value(self, x):
        scores = self.A @ x
        return float(np.mean(np.logaddexp(0.0, scores) - self.y * scores)
                     + 0.5 * self.ridge * (x @ x))

    def gradient(self, x):
        scores = self.A @ x
        return self.A.T @ (_sigmoid(scores) - self.y) / self.A.shape[0] + self.ridge * x

    def coord_fd(self, x, mu, coords=None, audit=False):
        idx = np.arange(x.size) if coords is None else np.asarray(coords, dtype=int)
        scores = self.A @ x
        shifts = self.A[:, idx] * mu
        plus_scores = scores[:, None] + shifts
        minus_scores = scores[:, None] - shifts
        norm_sq = float(x @ x)
        plus_norm = norm_sq + 2.0 * mu * x[idx] + mu * mu
        minus_norm = norm_sq - 2.0 * mu * x[idx] + mu * mu
        f_plus = (np.mean(np.logaddexp(0.0, plus_scores) - self.y[:, None] * plus_scores, axis=0)
                  + 0.5 * self.ridge * plus_norm)
        f_minus = (np.mean(np.logaddexp(0.0, minus_scores) - self.y[:, None] * minus_scores, axis=0)
                   + 0.5 * self.ridge * minus_norm)
        estimate = np.zeros_like(x)
        estimate[idx] = (f_plus - f_minus) / (2.0 * mu)
        values = None
        if audit:
            values = [{"coordinate": int(i), "f_plus": float(fp), "f_minus": float(fm)}
                      for i, fp, fm in zip(idx, f_plus, f_minus)]
        return estimate, 2 * idx.size, values

    def restricted_m(self, support, x=None):
        # The ridge term gives a known global restricted strong-convexity bound.
        return float(self.ridge)

    def random_direction_fd(self, x, mu, q, rng):
        d = x.size
        directions = rng.normal(size=(q, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        f_base = self.value(x)
        f_plus = np.asarray([self.value(x + mu * u) for u in directions])
        return (d / (q * mu)) * (directions.T @ (f_plus - f_base)), q + 1


@dataclass
class PhaseRetrieval:
    A: np.ndarray
    y: np.ndarray
    local_L: float
    name: str = "phase_retrieval"

    @property
    def dimension(self):
        return self.A.shape[1]

    @property
    def L(self):
        return self.local_L

    def value(self, x):
        scores = self.A @ x
        residual = scores * scores - self.y
        return 0.25 * float(residual @ residual) / self.A.shape[0]

    def gradient(self, x):
        scores = self.A @ x
        return self.A.T @ ((scores * scores - self.y) * scores) / self.A.shape[0]

    def hessian(self, x, support=None):
        idx = np.arange(x.size) if support is None else np.asarray(support, dtype=int)
        scores = self.A @ x
        columns = self.A[:, idx]
        weights = 3.0 * scores * scores - self.y
        return columns.T @ (weights[:, None] * columns) / self.A.shape[0]

    def coord_fd(self, x, mu, coords=None, audit=False):
        idx = np.arange(x.size) if coords is None else np.asarray(coords, dtype=int)
        scores = self.A @ x
        shifts = mu * self.A[:, idx]
        plus_residual = (scores[:, None] + shifts) ** 2 - self.y[:, None]
        minus_residual = (scores[:, None] - shifts) ** 2 - self.y[:, None]
        f_plus = 0.25 * np.mean(plus_residual ** 2, axis=0)
        f_minus = 0.25 * np.mean(minus_residual ** 2, axis=0)
        estimate = np.zeros_like(x)
        estimate[idx] = (f_plus - f_minus) / (2.0 * mu)
        values = None
        if audit:
            values = [{"coordinate": int(i), "f_plus": float(fp), "f_minus": float(fm)}
                      for i, fp, fm in zip(idx, f_plus, f_minus)]
        return estimate, 2 * idx.size, values

    def restricted_m(self, support, x=None):
        if x is None:
            return -1.0
        return float(np.linalg.eigvalsh(self.hessian(x, support))[0])

    def random_direction_fd(self, x, mu, q, rng):
        d = x.size
        directions = rng.normal(size=(q, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        f_base = self.value(x)
        f_plus = np.asarray([self.value(x + mu * u) for u in directions])
        return (d / (q * mu)) * (directions.T @ (f_plus - f_base)), q + 1


def make_sparse_regression(seed, m=128, d=256, k=16, a_min=0.25,
                           noise_std=0.1):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0 / np.sqrt(k), size=(m, d))
    support = np.sort(rng.choice(d, size=k, replace=False))
    planted = np.zeros(d)
    planted[support] = rng.uniform(a_min, 1.0, k) * rng.choice([-1.0, 1.0], k)
    y = A @ planted + rng.normal(0.0, noise_std, m)
    return LeastSquares(A, y), planted, support


def make_logistic(seed, m=512, d=128, k=8, a_min=0.75, ridge=0.02):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0 / np.sqrt(k), size=(m, d))
    support = np.sort(rng.choice(d, size=k, replace=False))
    planted = np.zeros(d)
    planted[support] = rng.uniform(a_min, 1.5, k) * rng.choice([-1.0, 1.0], k)
    y = _sigmoid(A @ planted)
    return RidgeLogistic(A, y, ridge), planted, support


def make_phase_retrieval(seed, m, d, k, a_min=0.5, noise_std=0.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0, size=(m, d))
    support = np.sort(rng.choice(d, size=k, replace=False))
    planted = np.zeros(d)
    planted[support] = rng.uniform(a_min, 1.0, k) * rng.choice([-1.0, 1.0], k)
    y = (A @ planted) ** 2 + rng.normal(0.0, noise_std, m)
    # Conservative initialization-scale value; phase experiments update the
    # diagnostic local constants and always enforce the shadow-full gate.
    local_L = float(np.linalg.norm(A, 2) ** 2 * max(np.max(np.abs(y)), 1.0) / m)
    return PhaseRetrieval(A, y, local_L), planted, support
