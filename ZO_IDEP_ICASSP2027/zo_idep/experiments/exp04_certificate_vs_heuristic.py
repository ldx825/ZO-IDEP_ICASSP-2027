#!/usr/bin/env python3
"""Exp04: certified switch versus stable-support freeze heuristics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (aggregate_by, endpoint_gaps, reset_jsonl, save_run, saving,
                    write_summary)
from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.objectives import make_logistic
from zo_idep.src.prox import stable_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--seed-start", type=int, default=42000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials = 3 if args.quick else args.trials
    raw_path = ROOT / "zo_idep/results/raw/exp04_certificate_vs_heuristic.jsonl"
    reset_jsonl(raw_path)
    records = []
    for trial in range(trials):
        seed = args.seed_start + trial
        # Disjoint development seeds selected this low-margin, slow-continuation
        # regime before report runs; no report seed was inspected during tuning.
        objective, planted, planted_support = make_logistic(seed, a_min=0.02)
        config = RunConfig(
            iterations=args.iterations, eta=0.1 / objective.L, lam=0.02,
            mu0=0.01, continuation=200, cert_start=199, seed=seed,
            shadow_tol=1e-6, certificate_eps_mode="smooth",
        )
        full = run_algorithm(objective, 8, "FULL_ZO_EP", config, planted, planted_support)
        final_support = stable_topk(full["endpoint"], 8)
        full_record = {
            "experiment": "exp04_certificate_vs_heuristic", "trial": trial,
            "seed": seed, "method": "FULL_ZO_EP", "switched": 0,
            "switch_time": args.iterations, "endpoint_gap_inf": 0.0,
            "endpoint_gap_l2": 0.0, "false_freeze": 0, "query_saving": 0.0,
            "objective": full["objective"], "full_objective": full["objective"],
            "recovery_error": full["recovery_error"],
            "maximum_shadow_mismatch_inf": 0.0,
        }
        records.append(full_record)
        save_run(raw_path, full_record, {**config.__dict__, "objective": "ridge_logistic",
                                         "a_min": 0.02})
        for method in ["ZO_IDEP", "STABLE_3", "STABLE_5", "STABLE_10",
                       "ORACLE_ENDSUPPORT"]:
            run = run_algorithm(objective, 8, method, config, planted, planted_support,
                                oracle_support=final_support)
            gap_inf, gap_l2 = endpoint_gaps(run["endpoint"], full["endpoint"])
            record = {
                "experiment": "exp04_certificate_vs_heuristic", "trial": trial,
                "seed": seed, "method": method, "switched": run["switched"],
                "switch_time": run["switch_time"], "endpoint_gap_inf": gap_inf,
                "endpoint_gap_l2": gap_l2, "false_freeze": int(gap_inf > 1e-6),
                "query_saving": saving(run["queries"], full["queries"]),
                "objective": run["objective"], "full_objective": full["objective"],
                "recovery_error": run["recovery_error"],
                "maximum_shadow_mismatch_inf": run["maximum_shadow_mismatch_inf"],
            }
            records.append(record)
            save_run(raw_path, record, {**config.__dict__, "objective": "ridge_logistic",
                                        "a_min": 0.02})
            print(f"trial={trial+1}/{trials} method={method} t={run['switch_time']} "
                  f"false={record['false_freeze']}", flush=True)
    values = ["switched", "switch_time", "endpoint_gap_inf", "endpoint_gap_l2",
              "false_freeze", "query_saving", "objective", "recovery_error",
              "maximum_shadow_mismatch_inf"]
    summary = aggregate_by(records, ["method"], values)
    write_summary("exp04_certificate_vs_heuristic", records, summary)


if __name__ == "__main__":
    main()
