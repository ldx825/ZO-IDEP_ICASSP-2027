#!/usr/bin/env python3
"""Exp02: nonlinear smoothing/error sweep on frozen report seeds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (aggregate_by, best_chi, compact_result, endpoint_gaps,
                    reset_jsonl, save_run, saving, write_summary)
from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.objectives import make_logistic
from zo_idep.src.prox import stable_topk


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
    solved = minimize(value, initial[support], jac=gradient, method="L-BFGS-B",
                      options={"gtol": 1e-12, "ftol": 1e-15, "maxiter": 10000})
    optimum = np.zeros(objective.dimension)
    optimum[support] = solved.x
    return optimum, solved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--seed-start", type=int, default=12000)
    parser.add_argument("--mu-index", type=int, choices=range(6))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials = 3 if args.quick else args.trials
    mus = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    if args.mu_index is not None:
        mus = [mus[args.mu_index]]
    suffix = ("" if args.mu_index is None else f"_part{args.mu_index}")
    raw_path = ROOT / f"zo_idep/results/raw/exp02_mu_sweep{suffix}.jsonl"
    reset_jsonl(raw_path)
    records = []
    for mu in mus:
        for trial in range(trials):
            seed = args.seed_start + trial
            objective, planted, support = make_logistic(seed)
            config = RunConfig(
                iterations=args.iterations, eta=0.1 / objective.L, lam=0.02,
                mu0=mu, continuation=100, cert_start=99, seed=seed,
                shadow_tol=1e-6, certificate_eps_mode="smooth",
            )
            full = run_algorithm(objective, 8, "FULL_ZO_EP", config, planted, support)
            adaptive = run_algorithm(objective, 8, "ZO_IDEP", config, planted, support)
            reference_support = stable_topk(full["endpoint"], 8)
            optimum, solver = restricted_optimum(objective, reference_support,
                                                  full["endpoint"])
            solver_gradient_norm = float(np.linalg.norm(
                objective.gradient(optimum)[reference_support]
            ))
            gap_inf, gap_l2 = endpoint_gaps(adaptive["endpoint"], full["endpoint"])
            record = {
                "experiment": "exp02_mu_sweep", "trial": trial, "seed": seed,
                "mu": mu, "eta_scale": 0.1, "theoretical_error_bound": objective.L * mu / 2,
                "actual_max_error": adaptive["maximum_zo_error_inf"],
                "actual_to_bound_max_ratio": adaptive["maximum_zo_error_ratio"],
                "error_bound_violations": adaptive["error_bound_violations"],
                "switched": adaptive["switched"], "switch_time": adaptive["switch_time"],
                "certificate_chi": (adaptive["certificate_at_switch"]["chi"]
                                    if adaptive["certificate_at_switch"] else float("nan")),
                "best_candidate_chi": best_chi(adaptive),
                "endpoint_gap_inf": gap_inf, "endpoint_gap_l2": gap_l2,
                "endpoint_mismatch": int(gap_inf > 1e-6),
                "full_queries": full["queries"], "idep_queries": adaptive["queries"],
                "query_saving": saving(adaptive["queries"], full["queries"]),
                "full_objective": full["objective"], "idep_objective": adaptive["objective"],
                "full_neighborhood_distance": float(np.linalg.norm(full["endpoint"] - optimum)),
                "idep_neighborhood_distance": float(np.linalg.norm(adaptive["endpoint"] - optimum)),
                "solver_success": int(solver.success or solver_gradient_norm <= 1e-8),
                "solver_gradient_norm": solver_gradient_norm,
            }
            records.append(record)
            save_run(raw_path, record, {**config.__dict__, "objective": "ridge_logistic"})
            print(f"mu={mu:g} trial={trial+1}/{trials} switch={record['switched']} "
                  f"t={record['switch_time']} gap={gap_inf:.2e}", flush=True)
    values = ["actual_max_error", "theoretical_error_bound", "actual_to_bound_max_ratio",
              "error_bound_violations", "switched", "switch_time", "certificate_chi",
              "best_candidate_chi", "endpoint_gap_inf", "endpoint_gap_l2",
              "endpoint_mismatch", "query_saving"]
    values += ["full_neighborhood_distance", "idep_neighborhood_distance",
               "solver_success", "solver_gradient_norm"]
    summary = aggregate_by(records, ["mu"], values)
    write_summary(f"exp02_mu_sweep{suffix}", records, summary)
    print(f"wrote {raw_path}")


if __name__ == "__main__":
    main()
