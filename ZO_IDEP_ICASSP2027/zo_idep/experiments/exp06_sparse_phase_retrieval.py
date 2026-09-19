#!/usr/bin/env python3
"""Exp06: sparse phase retrieval in a locally nonconvex setting."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (aggregate_by, endpoint_gaps, reset_jsonl, save_run, saving,
                    write_summary)
from zo_idep.src.algorithms import RunConfig, run_algorithm
from zo_idep.src.objectives import make_phase_retrieval
from zo_idep.src.prox import stable_topk


def controlled_local_initialization(planted, k, seed):
    """Controlled local-basin start for testing the local certificate.

    This deliberately uses the planted solution and is therefore a diagnostic
    warm start, not a deployable phase-retrieval initialization method.
    """
    rng = np.random.default_rng(seed + 991)
    dense = planted + rng.normal(0.0, 0.02, size=planted.size)
    support = stable_topk(dense, k)
    initial = np.zeros(planted.size)
    initial[support] = dense[support]
    return initial


def local_constants(objective, x0, k):
    """Frozen local diagnostic constants around the common initializer."""
    support = stable_topk(x0, k)
    hessian = objective.hessian(x0)
    spectral_radius = float(np.linalg.norm(hessian, 2))
    restricted_eigenvalues = np.linalg.eigvalsh(objective.hessian(x0, support))
    # L is deliberately padded; m remains a diagnostic lower bound and the
    # certificate cannot pass when local curvature is nonpositive.
    objective.local_L = max(1e-8, 2.0 * spectral_radius)
    return float(restricted_eigenvalues[0]), objective.local_L


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--seed-start", type=int, default=62000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    trials = 3 if args.quick else args.trials
    settings = [(128, 8), (128, 16), (256, 8), (256, 16)]
    if args.quick:
        settings = settings[:2]
    methods = ["FULL_ZO_EP", "ZO_IDEP", "COORD_ZO_IHT", "SZOHT_RD"]
    raw_path = ROOT / "zo_idep/results/raw/exp06_sparse_phase_retrieval.jsonl"
    reset_jsonl(raw_path)
    records = []
    for d, k in settings:
        m = int(math.ceil(4 * k * math.log(d)))
        for trial in range(trials):
            seed = args.seed_start + 1000 * d + 37 * k + trial
            objective, planted, planted_support = make_phase_retrieval(seed, m, d, k)
            x0 = controlled_local_initialization(planted, k, seed)
            initial_m, local_L = local_constants(objective, x0, k)
            config = RunConfig(
                iterations=args.iterations, eta=0.8 / local_L, lam=0.10,
                mu0=1e-4, beta=0.95, decay_start=100, continuation=100,
                cert_start=99, seed=seed, shadow_tol=1e-6,
                certificate_eps_mode="smooth", szoht_q=2 * (3 * k + 2),
                szoht_step_scale=0.02,
            )
            full = None
            for method in methods:
                run = run_algorithm(objective, k, method, config, planted,
                                    planted_support, x0=x0)
                if method == "FULL_ZO_EP":
                    full = run
                gap_inf, gap_l2 = ((0.0, 0.0) if full is run else
                                   endpoint_gaps(run["endpoint"], full["endpoint"]))
                record = {
                    "experiment": "exp06_sparse_phase_retrieval", "trial": trial,
                    "seed": seed, "m": m, "d": d, "k": k, "method": method,
                    "initial_recovery_error": float(min(np.linalg.norm(x0 - planted),
                                                        np.linalg.norm(x0 + planted))),
                    "initial_restricted_m": initial_m, "local_L": local_L,
                    "objective": run["objective"], "recovery_error": run["recovery_error"],
                    "support_overlap": run["support_overlap"],
                    "switched": run["switched"], "switch_time": run["switch_time"],
                    "queries": run["queries"],
                    "query_saving_vs_full": saving(run["queries"], full["queries"]),
                    "endpoint_gap_inf": gap_inf, "endpoint_gap_l2": gap_l2,
                    "endpoint_mismatch": int(gap_inf > 1e-6),
                    "maximum_shadow_mismatch_inf": run["maximum_shadow_mismatch_inf"],
                    "error_bound_violations": run["error_bound_violations"],
                }
                records.append(record)
                save_run(raw_path, record, {**config.__dict__, "objective": "phase_retrieval",
                                             "m": m, "d": d, "k": k,
                                             "initializer": "controlled_local_topk",
                                             "initializer_noise_std": 0.02})
            print(f"d={d} k={k} trial={trial+1}/{trials} "
                  f"cert={records[-3]['switched']} t={records[-3]['switch_time']}",
                  flush=True)
    values = ["objective", "recovery_error", "support_overlap", "switched",
              "switch_time", "queries", "query_saving_vs_full", "endpoint_gap_inf",
              "endpoint_gap_l2", "endpoint_mismatch", "maximum_shadow_mismatch_inf",
              "error_bound_violations"]
    summary = aggregate_by(records, ["d", "k", "method"], values)
    write_summary("exp06_sparse_phase_retrieval", records, summary)


if __name__ == "__main__":
    main()
