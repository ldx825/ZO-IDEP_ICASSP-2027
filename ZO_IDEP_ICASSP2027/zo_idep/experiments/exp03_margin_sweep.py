#!/usr/bin/env python3
"""Exp03: measured identification margin versus certified switch time."""
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

from common import (aggregate_by, best_chi, endpoint_gaps, reset_jsonl, save_run,
                    saving, write_summary)
from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.diagnostics import support_margin
from zo_idep.src.objectives import make_sparse_regression
from zo_idep.src.prox import stable_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--seed-start", type=int, default=32000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials = 3 if args.quick else args.trials
    minima = [0.02, 0.05, 0.10, 0.25, 0.50]
    raw_path = ROOT / "zo_idep/results/raw/exp03_margin_sweep.jsonl"
    reset_jsonl(raw_path)
    records = []
    for a_min in minima:
        for trial in range(trials):
            seed = args.seed_start + trial
            objective, planted, planted_support = make_sparse_regression(
                seed, 128, 256, 16, a_min=a_min
            )
            config = RunConfig(
                iterations=args.iterations, eta=0.9 / objective.L, lam=0.025,
                mu0=1e-3, continuation=100, cert_start=99, seed=seed,
                shadow_tol=1e-8, certificate_eps_mode="zero_quadratic",
            )
            full = run_algorithm(objective, 16, "FULL_ZO_EP", config, planted, planted_support)
            adaptive = run_algorithm(objective, 16, "ZO_IDEP", config, planted, planted_support)
            reference_support = stable_topk(full["endpoint"], 16)
            margins = support_margin(objective, full["endpoint"], reference_support,
                                     config.eta, config.lam)
            gap_inf, gap_l2 = endpoint_gaps(adaptive["endpoint"], full["endpoint"])
            record = {
                "experiment": "exp03_margin_sweep", "trial": trial, "seed": seed,
                "a_min": a_min, **margins,
                "support_saturated": int(full["support_size"] == 16),
                "switched": adaptive["switched"], "switch_time": adaptive["switch_time"],
                "certificate_chi": (adaptive["certificate_at_switch"]["chi"]
                                    if adaptive["certificate_at_switch"] else float("nan")),
                "best_candidate_chi": best_chi(adaptive),
                "endpoint_gap_inf": gap_inf, "endpoint_gap_l2": gap_l2,
                "endpoint_mismatch": int(gap_inf > 1e-8),
                "query_saving": saving(adaptive["queries"], full["queries"]),
                "support_overlap": adaptive["support_overlap"],
            }
            records.append(record)
            save_run(raw_path, record, {**config.__dict__, "objective": "sparse_regression"})
            print(f"a_min={a_min:g} trial={trial+1}/{trials} delta={margins['delta_star']:.2e} "
                  f"switch={record['switched']} t={record['switch_time']}", flush=True)
    values = ["delta_star", "gamma_star", "tau_star", "support_saturated", "switched",
              "switch_time", "certificate_chi", "best_candidate_chi", "endpoint_gap_inf",
              "endpoint_gap_l2", "endpoint_mismatch", "query_saving", "support_overlap"]
    summary = aggregate_by(records, ["a_min"], values)
    write_summary("exp03_margin_sweep", records, summary)


if __name__ == "__main__":
    main()
