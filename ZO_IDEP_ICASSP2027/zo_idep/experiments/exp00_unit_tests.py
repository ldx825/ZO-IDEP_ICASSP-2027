#!/usr/bin/env python3
"""Unit and correctness tests required by the execution plan."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.certificate import certificate
from zo_idep.src.finite_difference import coord_fd
from zo_idep.src.objectives import LeastSquares
from zo_idep.src.prox import (brute_force_prox_candidates, prox_objective,
                              prox_trimmed_l1, stable_topk)


def test_prox_bruteforce():
    rng = np.random.default_rng(81)
    for d in range(2, 9):
        for k in range(d + 1):
            for _ in range(25):
                z = rng.normal(size=d)
                eta, lam = 0.37, 0.61
                answer, _ = prox_trimmed_l1(z, eta, lam, k)
                answer_value = prox_objective(answer, z, eta, lam, k)
                candidates = brute_force_prox_candidates(z, eta, lam, k)
                candidate_values = [prox_objective(u, z, eta, lam, k) for u in candidates]
                assert answer_value <= min(candidate_values) + 1e-11


def test_stable_ties():
    z = np.array([1.0, -1.0, 1.0, 0.2])
    assert np.array_equal(stable_topk(z, 2), np.array([0, 1]))
    answer, support = prox_trimmed_l1(z, 0.5, 0.4, 2)
    assert np.array_equal(support, np.array([0, 1]))
    assert np.allclose(answer, np.array([1.0, -1.0, 0.8, 0.0]))


def test_coord_fd_and_count():
    diagonal = np.arange(1.0, 7.0)
    x = np.linspace(-0.5, 0.5, 6)
    f = lambda u: 0.5 * float(np.sum(diagonal * u * u))
    estimate, count, audit = coord_fd(f, x, 1e-5, [1, 3, 5], True)
    expected = np.zeros_like(x)
    expected[[1, 3, 5]] = diagonal[[1, 3, 5]] * x[[1, 3, 5]]
    assert count == 6 and len(audit) == 3
    assert np.max(np.abs(estimate - expected)) < 1e-10


def test_certificate_margins():
    x = np.array([2.0, 1.0, 0.0, 0.0])
    ghat = np.array([0.05, -0.05, 0.01, -0.02])
    passed, details = certificate(x, ghat, np.array([0, 1]), 0.5, 0.5, 1.0, 1.0, 0.0)
    assert passed and details["strict"] and details["chi"] > 0
    assert set(details) >= {"R", "H", "A", "q", "chi", "margin_invariance",
                            "margin_threshold", "margin_rank"}


def test_shadow_equivalence_gate():
    d = 6
    A = np.sqrt(d) * np.eye(d)
    target = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    objective = LeastSquares(A, A @ target)
    config = RunConfig(iterations=3, eta=0.5, lam=0.5, mu0=1e-4,
                       continuation=0, cert_start=0, shadow_tol=1e-10)
    result = run_algorithm(objective, 2, "ZO_IDEP", config,
                           x0=target + np.array([0.01, -0.01, 0, 0, 0, 0]))
    assert result["switched"] == 1
    assert result["maximum_shadow_mismatch_inf"] <= 1e-10


def main():
    tests = [test_prox_bruteforce, test_stable_ties, test_coord_fd_and_count,
             test_certificate_margins, test_shadow_equivalence_gate]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} unit/correctness tests")


if __name__ == "__main__":
    main()
