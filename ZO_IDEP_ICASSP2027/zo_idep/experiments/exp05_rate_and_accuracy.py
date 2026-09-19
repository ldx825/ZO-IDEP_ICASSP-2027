#!/usr/bin/env python3
"""Exp05: post-certificate geometric rate and target-accuracy query ratios."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (aggregate_by, endpoint_gaps, reset_jsonl, save_run,
                    write_summary)
from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.objectives import make_logistic, make_sparse_regression
from zo_idep.src.prox import stable_topk

TARGETS = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3]


def restricted_optimum(objective, support, initial):
    support = np.sort(np.asarray(support, dtype=int))
    def value(active):
        x = np.zeros(objective.dimension)
        x[support] = active
        return objective.value(x)
    def gradient(active):
        x = np.zeros(objective.dimension)
        x[support] = active
        return objective.gradient(x)[support]
    result = minimize(value, initial[support], jac=gradient, method="L-BFGS-B",
                      options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10000})
    optimum = np.zeros(objective.dimension)
    optimum[support] = result.x
    return optimum, {"success": bool(result.success), "gradient_norm": float(np.linalg.norm(gradient(result.x))),
                     "iterations": int(result.nit), "message": str(result.message)}


def first_target_queries(trace, targets):
    output = {}
    for target in targets:
        hits = [row["queries"] for row in trace if row["reference_distance"] <= target]
        output[str(target)] = int(hits[0]) if hits else None
    return output


def fit_rate(trace, switch_time):
    rows = [row for row in trace if row["iteration"] > switch_time + 5
            and np.isfinite(row["reference_distance"]) and row["reference_distance"] > 1e-12]
    if len(rows) < 5:
        return float("nan"), float("nan"), 0
    steps = np.asarray([row["iteration"] - switch_time for row in rows], dtype=float)
    log_distance = np.log([row["reference_distance"] for row in rows])
    slope, intercept = np.polyfit(steps, log_distance, 1)
    return float(np.exp(slope)), float(slope), len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=52000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials = 3 if args.quick else args.trials
    betas = [0.95, 0.97, 0.99]
    raw_path = ROOT / "zo_idep/results/raw/exp05_rate_and_accuracy.jsonl"
    reset_jsonl(raw_path)
    records = []
    for objective_name in ["ridge_logistic", "sparse_regression"]:
        for beta in betas:
            for trial in range(trials):
                seed = args.seed_start + trial
                if objective_name == "ridge_logistic":
                    objective, planted, planted_support = make_logistic(seed)
                    eta, lam, mu0, shadow_tol, eps_mode = (0.1/objective.L, 0.02, 0.01, 1e-6, "smooth")
                    k = 8
                else:
                    objective, planted, planted_support = make_sparse_regression(seed, 128, 256, 16)
                    eta, lam, mu0, shadow_tol, eps_mode = (0.9/objective.L, 0.025, 1e-3, 1e-8, "zero_quadratic")
                    k = 16
                config = RunConfig(
                    iterations=args.iterations, eta=eta, lam=lam, mu0=mu0,
                    mu_min=1e-6, beta=beta, decay_start=100,
                    continuation=100, cert_start=99,
                    seed=seed, shadow_tol=shadow_tol, certificate_eps_mode=eps_mode,
                )
                preliminary = run_algorithm(objective, k, "ZO_IDEP", config,
                                            planted, planted_support)
                chosen_support = (np.asarray(preliminary["switch_support"], dtype=int)
                                  if preliminary["switched"] else stable_topk(preliminary["endpoint"], k))
                optimum, solver = restricted_optimum(objective, chosen_support,
                                                      preliminary["endpoint"])
                full = run_algorithm(objective, k, "FULL_ZO_EP", config, planted,
                                     planted_support, distance_reference=optimum)
                adaptive = run_algorithm(objective, k, "ZO_IDEP", config, planted,
                                         planted_support, distance_reference=optimum)
                gap_inf, gap_l2 = endpoint_gaps(adaptive["endpoint"], full["endpoint"])
                full_targets = first_target_queries(full["trace"], TARGETS)
                adaptive_targets = first_target_queries(adaptive["trace"], TARGETS)
                empirical_rho, slope, fit_points = fit_rate(adaptive["trace"], adaptive["switch_time"])
                theorem_rho = (max(adaptive["certificate_at_switch"]["q"], beta)
                               if adaptive["certificate_at_switch"] else float("nan"))
                for target in TARGETS:
                    q_full = full_targets[str(target)]
                    q_idep = adaptive_targets[str(target)]
                    record = {
                        "experiment": "exp05_rate_and_accuracy", "objective_name": objective_name,
                        "trial": trial, "seed": seed, "beta": beta, "target": target,
                        "switched": adaptive["switched"], "switch_time": adaptive["switch_time"],
                        "full_queries_target": q_full, "idep_queries_target": q_idep,
                        "target_reached_full": int(q_full is not None),
                        "target_reached_idep": int(q_idep is not None),
                        "query_ratio": (q_idep / q_full if q_full and q_idep else float("nan")),
                        "k_over_d": k / objective.dimension,
                        "empirical_rho": empirical_rho, "fit_slope": slope,
                        "fit_points": fit_points, "theorem_rho": theorem_rho,
                        "endpoint_gap_inf": gap_inf, "endpoint_gap_l2": gap_l2,
                        "solver_success": int(solver["success"]),
                        "solver_gradient_norm": solver["gradient_norm"],
                    }
                    records.append(record)
                    save_run(raw_path, record, {**config.__dict__, "objective": objective_name})
                print(f"{objective_name} beta={beta} trial={trial+1}/{trials} "
                      f"switch={adaptive['switched']} t={adaptive['switch_time']} rho={empirical_rho:.4f}",
                      flush=True)
    values = ["switched", "switch_time", "target_reached_full", "target_reached_idep",
              "query_ratio", "k_over_d", "empirical_rho", "theorem_rho",
              "endpoint_gap_inf", "endpoint_gap_l2", "solver_success", "solver_gradient_norm"]
    summary = aggregate_by(records, ["objective_name", "beta", "target"], values)
    write_summary("exp05_rate_and_accuracy", records, summary)


if __name__ == "__main__":
    main()
