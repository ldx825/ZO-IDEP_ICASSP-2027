#!/usr/bin/env python3
"""Exp01: regression test of the paper table with theorem diagnostics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (aggregate_by, best_chi, compact_result, endpoint_gaps,
                    reset_jsonl, save_run, saving, write_summary)
from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.diagnostics import prefix_stationarity, support_margin
from zo_idep.src.objectives import make_sparse_regression
from zo_idep.src.prox import stable_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials = 3 if args.quick else args.trials
    settings = [(128, 256, 16), (128, 256, 24), (128, 256, 32),
                (128, 512, 32), (128, 1024, 64)]
    if args.quick:
        settings = settings[:2]
    methods = ["PG_EP", "SZOHT_RD", "COORD_ZO_IHT", "FULL_ZO_EP", "ZO_IDEP"]
    raw_path = ROOT / "zo_idep/results/raw/exp01_sparse_regression.jsonl"
    reset_jsonl(raw_path)
    records = []
    for m, d, k in settings:
        for trial in range(trials):
            seed = 2027 + 1000 * d + 37 * k + trial
            objective, planted, planted_support = make_sparse_regression(seed, m, d, k)
            config = RunConfig(
                iterations=args.iterations, eta=0.9 / objective.L, lam=0.025,
                mu0=1e-3, continuation=100, cert_start=99, seed=2027 + trial,
                shadow_tol=1e-8, certificate_eps_mode="zero_quadratic",
            )
            runs = {}
            for method in methods:
                runs[method] = run_algorithm(objective, k, method, config,
                                             planted, planted_support)
            full = runs["FULL_ZO_EP"]
            endpoint_support = stable_topk(full["endpoint"], k)
            margins = support_margin(objective, full["endpoint"], endpoint_support,
                                     config.eta, config.lam)
            for method, run in runs.items():
                gap_inf, gap_l2 = endpoint_gaps(run["endpoint"], full["endpoint"])
                record = {
                    "experiment": "exp01_sparse_regression", "trial": trial,
                    "seed": seed, "m": m, "d": d, "k": k, "method": method,
                    **compact_result(run), **margins,
                    "endpoint_gap_inf": gap_inf, "endpoint_gap_l2": gap_l2,
                    "query_saving_vs_full": saving(run["queries"], full["queries"]),
                    "best_candidate_chi": best_chi(run),
                    "prefix_stationarity": (prefix_stationarity(
                        run["trace"], [25, 50, 100, 200, args.iterations]
                    ) if method in {"PG_EP", "FULL_ZO_EP", "ZO_IDEP"} else []),
                }
                records.append(record)
                save_run(raw_path, record, {**config.__dict__, "objective": "sparse_regression"})
            print(f"d={d} k={k} trial={trial+1}/{trials} "
                  f"switch={runs['ZO_IDEP']['switched']} t={runs['ZO_IDEP']['switch_time']} "
                  f"gap={records[-1]['endpoint_gap_inf']:.2e}", flush=True)
    values = ["objective", "recovery_error", "support_overlap", "exact_support",
              "queries", "switch_time", "switched", "maximum_shadow_mismatch_inf",
              "maximum_zo_error_inf", "error_bound_violations", "endpoint_gap_inf",
              "endpoint_gap_l2", "query_saving_vs_full", "delta_star", "best_candidate_chi"]
    summary = aggregate_by(records, ["d", "k", "method"], values)
    write_summary("exp01_sparse_regression", records, summary)


if __name__ == "__main__":
    main()
