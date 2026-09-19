#!/usr/bin/env python3
"""Nonquadratic ZO-IDEP stress test on ridge-logistic sparse recovery.

Soft Bernoulli targets remove label-sampling noise so that any discrepancy
between full and support-restricted coordinate ZO is attributable to nonlinear
finite-difference dynamics and the certified switch. Every central-difference
point is counted as one function query.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from run_experiments import ROOT, topk_support, trimmed_l1_prox


@dataclass
class LogisticOracle:
    A: np.ndarray
    y: np.ndarray
    ridge: float
    queries: int = 0

    def value(self, x: np.ndarray, count: bool = True) -> float:
        if count:
            self.queries += 1
        scores = self.A @ x
        return float(np.mean(np.logaddexp(0.0, scores) - self.y * scores)
                     + 0.5 * self.ridge * (x @ x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        scores = self.A @ x
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        return self.A.T @ (probabilities - self.y) / self.A.shape[0] + self.ridge * x

    def coord_fd(self, x: np.ndarray, mu: float,
                 support: np.ndarray | None = None) -> np.ndarray:
        idx = np.arange(x.size) if support is None else np.asarray(support)
        scores = self.A @ x
        shifts = self.A[:, idx] * mu
        plus_scores = scores[:, None] + shifts
        minus_scores = scores[:, None] - shifts
        base_norm = x @ x
        plus_norm = base_norm + 2.0 * mu * x[idx] + mu * mu
        minus_norm = base_norm - 2.0 * mu * x[idx] + mu * mu
        fp = np.mean(np.logaddexp(0.0, plus_scores)
                     - self.y[:, None] * plus_scores, axis=0) + 0.5 * self.ridge * plus_norm
        fm = np.mean(np.logaddexp(0.0, minus_scores)
                     - self.y[:, None] * minus_scores, axis=0) + 0.5 * self.ridge * minus_norm
        gradient = np.zeros_like(x)
        gradient[idx] = (fp - fm) / (2.0 * mu)
        self.queries += 2 * idx.size
        return gradient


def make_problem(seed: int, samples: int, dimension: int, sparsity: int):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0 / np.sqrt(sparsity), size=(samples, dimension))
    support = np.sort(rng.choice(dimension, size=sparsity, replace=False))
    planted = np.zeros(dimension)
    planted[support] = (rng.uniform(0.75, 1.5, size=sparsity)
                        * rng.choice([-1.0, 1.0], size=sparsity))
    scores = A @ planted
    targets = 1.0 / (1.0 + np.exp(-scores))
    return A, targets, planted, support


def run(A: np.ndarray, y: np.ndarray, planted: np.ndarray,
        planted_support: np.ndarray, sparsity: int, method: str,
        iterations: int, lam: float, mu: float, ridge: float,
        continuation: int):
    oracle = LogisticOracle(A, y, ridge)
    dimension = A.shape[1]
    smoothness = float(np.linalg.norm(A, 2) ** 2 / (4.0 * A.shape[0]) + ridge)
    eta = 0.9 / smoothness
    epsilon = smoothness * mu / 2.0
    x = np.zeros(dimension)
    switched = False
    frozen_support = None
    switch_iteration = iterations

    for iteration in range(iterations):
        if method == "ZO-IDEP" and switched:
            gradient = oracle.coord_fd(x, mu, frozen_support)
        else:
            gradient = oracle.coord_fd(x, mu)

        if (method == "ZO-IDEP" and not switched and iteration >= continuation
                and np.count_nonzero(np.abs(x) > 1e-10) == sparsity):
            candidate = topk_support(x, sparsity)
            radius = (np.linalg.norm(gradient[candidate])
                      + np.sqrt(sparsity) * epsilon) / ridge
            inactive = np.setdiff1d(np.arange(dimension), candidate, assume_unique=True)
            inactive_bound = (np.max(np.abs(gradient[inactive])) + epsilon
                              + smoothness * radius)
            active_bound = np.min(np.abs(x[candidate])) - radius
            if inactive_bound < lam and active_bound > eta * inactive_bound > 0.0:
                switched = True
                frozen_support = candidate.copy()
                switch_iteration = iteration

        ramp = min(1.0, (iteration + 1) / continuation)
        lam_t = (lam / 40.0) * (40.0 ** ramp)
        x = trimmed_l1_prox(x - eta * gradient, eta * lam_t, sparsity)

    estimated_support = topk_support(x, sparsity)
    return {
        "method": method,
        "objective": oracle.value(x, count=False),
        "recovery_error": float(np.linalg.norm(x - planted)),
        "support_overlap": float(len(np.intersect1d(estimated_support, planted_support)) / sparsity),
        "queries": oracle.queries,
        "t_id": switch_iteration,
        "switched": int(switched),
        "endpoint": x,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--sparsity", type=int, default=8)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=0.02)
    parser.add_argument("--continuation", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2701)
    args = parser.parse_args()

    rows = []
    for trial in range(args.trials):
        problem = make_problem(args.seed + trial, args.samples, args.dimension, args.sparsity)
        full = run(*problem, args.sparsity, "Full-ZO-EP", args.iterations,
                   args.lam, args.mu, args.ridge, args.continuation)
        adaptive = run(*problem, args.sparsity, "ZO-IDEP", args.iterations,
                       args.lam, args.mu, args.ridge, args.continuation)
        endpoint_gap = float(np.linalg.norm(full.pop("endpoint") - adaptive.pop("endpoint")))
        for result in (full, adaptive):
            result.update({
                "trial": trial,
                "samples": args.samples,
                "dimension": args.dimension,
                "sparsity": args.sparsity,
                "lambda": args.lam,
                "mu": args.mu,
                "ridge": args.ridge,
                "endpoint_gap": endpoint_gap,
            })
            rows.append(result)
        print(f"finished nonlinear trial {trial + 1}/{args.trials}", flush=True)

    output = ROOT / "results" / "nonlinear_logistic_raw.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print("\nmethod, objective, error, overlap, queries, t_id, switch rate, endpoint gap")
    for method in ("Full-ZO-EP", "ZO-IDEP"):
        selected = [row for row in rows if row["method"] == method]
        means = [np.mean([row[key] for row in selected]) for key in
                 ("objective", "recovery_error", "support_overlap", "queries", "t_id", "switched", "endpoint_gap")]
        print(method + ", " + ", ".join(f"{value:.6g}" for value in means))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
