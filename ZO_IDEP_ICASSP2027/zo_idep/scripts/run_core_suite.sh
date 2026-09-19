#!/usr/bin/env bash
set -euo pipefail

paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$paper_dir"

python zo_idep/experiments/exp00_unit_tests.py
python zo_idep/experiments/exp02_mu_sweep.py "$@"
python zo_idep/experiments/exp04_certificate_vs_heuristic.py "$@"
python zo_idep/experiments/exp05_rate_and_accuracy.py "$@"
python zo_idep/experiments/exp03_margin_sweep.py "$@"
python zo_idep/experiments/exp06_sparse_phase_retrieval.py "$@"
python zo_idep/experiments/exp07_scaling.py "$@"
python zo_idep/experiments/exp01_sparse_regression.py "$@"
python zo_idep/scripts/validate_results.py
python zo_idep/scripts/build_paper_figures.py
