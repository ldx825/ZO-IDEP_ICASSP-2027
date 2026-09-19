#!/usr/bin/env python3
"""Fail-fast Definition-of-Done checks for frozen report results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "zo_idep" / "results" / "summary"
RAW = ROOT / "zo_idep" / "results" / "raw"

EXPECTED = {
    "exp01_sparse_regression": 5 * 30 * 5,
    "exp02_mu_sweep": 6 * 30,
    "exp03_margin_sweep": 5 * 30,
    "exp04_certificate_vs_heuristic": 6 * 30,
    "exp05_rate_and_accuracy": 2 * 3 * 30 * 7,
    "exp06_sparse_phase_retrieval": 4 * 30 * 4,
    "exp07_scaling": 4 * 30,
}


def load(name):
    return json.loads((SUMMARY / f"{name}.json").read_text())["records"]


def main():
    report = {"experiments": {}, "critical_gate": "PASS", "errors": []}
    all_records = {}
    for name, expected in EXPECTED.items():
        records = load(name)
        all_records[name] = records
        raw_lines = sum(1 for line in (RAW / f"{name}.jsonl").read_text().splitlines()
                        if line.strip())
        ok = len(records) == expected and raw_lines == expected
        report["experiments"][name] = {
            "summary_records": len(records), "raw_records": raw_lines,
            "expected_records": expected, "pass": ok,
        }
        if not ok:
            report["errors"].append(f"{name}: incomplete record count")

    paired_names = ["exp01_sparse_regression", "exp02_mu_sweep",
                    "exp03_margin_sweep", "exp05_rate_and_accuracy",
                    "exp06_sparse_phase_retrieval", "exp07_scaling"]
    maximum_gap = 0.0
    maximum_shadow = 0.0
    for name in paired_names:
        for row in all_records[name]:
            if "method" in row and row["method"] != "ZO_IDEP":
                continue
            gap = row.get("endpoint_gap_inf")
            if gap is not None:
                maximum_gap = max(maximum_gap, float(gap))
            shadow = row.get("maximum_shadow_mismatch_inf")
            if shadow is not None:
                maximum_shadow = max(maximum_shadow, float(shadow))
    report["maximum_endpoint_gap_inf"] = maximum_gap
    report["maximum_shadow_mismatch_inf"] = maximum_shadow
    if maximum_shadow > 1e-6:
        report["critical_gate"] = "FAIL"
        report["errors"].append("certified shadow-full mismatch exceeds 1e-6")

    exp02 = all_records["exp02_mu_sweep"]
    report["exp02_error_bound_violations"] = int(sum(r["error_bound_violations"]
                                                       for r in exp02))
    if report["exp02_error_bound_violations"]:
        report["errors"].append("nonlinear finite-difference bound violation")

    report["pass"] = not report["errors"]
    path = SUMMARY / "validation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
