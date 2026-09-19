#!/usr/bin/env python3
"""Exp07: dimension scaling at fixed k/d."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (aggregate_by, endpoint_gaps, reset_jsonl, save_run, write_summary)
from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.objectives import make_sparse_regression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--include-2048", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials = 3 if args.quick else args.trials
    settings = [(128, 8), (256, 16), (512, 32), (1024, 64)]
    if args.include_2048:
        settings.append((2048, 128))
    if args.quick:
        settings = settings[:2]
    raw_path = ROOT / "zo_idep/results/raw/exp07_scaling.jsonl"
    reset_jsonl(raw_path)
    records = []
    for d, k in settings:
        m = max(128, 8 * k)
        for trial in range(trials):
            seed = 72000 + trial
            objective, planted, support = make_sparse_regression(seed, m, d, k)
            config = RunConfig(
                iterations=args.iterations, eta=0.9/objective.L, lam=0.025,
                mu0=1e-3, continuation=100, cert_start=99, seed=seed,
                shadow_tol=1e-8, certificate_eps_mode="zero_quadratic",
            )
            start = time.perf_counter()
            full = run_algorithm(objective, k, "FULL_ZO_EP", config, planted, support)
            full_time = time.perf_counter() - start
            start = time.perf_counter()
            adaptive = run_algorithm(objective, k, "ZO_IDEP", config, planted, support)
            adaptive_time = time.perf_counter() - start
            gap_inf, gap_l2 = endpoint_gaps(adaptive["endpoint"], full["endpoint"])
            record = {
                "experiment": "exp07_scaling", "trial": trial, "seed": seed,
                "m": m, "d": d, "k": k, "k_over_d": k/d,
                "switched": adaptive["switched"], "switch_time": adaptive["switch_time"],
                "full_queries": full["queries"], "idep_queries": adaptive["queries"],
                "query_ratio": adaptive["queries"] / full["queries"],
                "post_switch_queries_per_step": 2*k if adaptive["switched"] else 2*d,
                "endpoint_gap_inf": gap_inf, "endpoint_gap_l2": gap_l2,
                "full_time_sec": full_time, "idep_time_sec": adaptive_time,
                "maximum_shadow_mismatch_inf": adaptive["maximum_shadow_mismatch_inf"],
            }
            records.append(record)
            save_run(raw_path, record, {**config.__dict__, "objective": "sparse_regression"})
            print(f"d={d} k={k} trial={trial+1}/{trials} switch={adaptive['switched']} "
                  f"ratio={record['query_ratio']:.3f}", flush=True)
    values = ["switched", "switch_time", "full_queries", "idep_queries", "query_ratio",
              "post_switch_queries_per_step", "endpoint_gap_inf", "endpoint_gap_l2",
              "full_time_sec", "idep_time_sec", "maximum_shadow_mismatch_inf"]
    summary = aggregate_by(records, ["d", "k"], values)
    write_summary("exp07_scaling", records, summary)


if __name__ == "__main__":
    main()
