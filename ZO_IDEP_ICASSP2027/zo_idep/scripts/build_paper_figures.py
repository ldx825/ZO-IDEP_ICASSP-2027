#!/usr/bin/env python3
"""Build the frozen theory-diagnostic and target-accuracy vector figures."""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "zo_idep" / "results" / "summary"
RESULT_FIGURES = ROOT / "zo_idep" / "results" / "figures"
PAPER_FIGURES = ROOT / "figures"

mpl.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def records(name):
    path = SUMMARY / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen summary: {path}")
    return json.loads(path.read_text())["records"]


def finite(values):
    return np.asarray([value for value in values if value is not None and math.isfinite(value)],
                      dtype=float)


def mean_se(values):
    values = finite(values)
    if not values.size:
        return np.nan, 0.0
    return float(np.mean(values)), (float(np.std(values, ddof=1) / np.sqrt(values.size))
                                    if values.size > 1 else 0.0)


def grouped(rows, key):
    output = {}
    for row in rows:
        output.setdefault(row[key], []).append(row)
    return output


def build_theory_diagnostics():
    mu_rows = records("exp02_mu_sweep")
    margin_rows = records("exp03_margin_sweep")
    rate_rows = records("exp05_rate_and_accuracy")
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.05))

    groups = grouped(mu_rows, "mu")
    mus = np.asarray(sorted(groups), dtype=float)
    switches, switch_se, mismatch = [], [], []
    for mu in mus:
        mean, se = mean_se([r["switch_time"] for r in groups[mu]])
        switches.append(mean)
        switch_se.append(se)
        mismatch.append(100.0 * np.mean([r["endpoint_mismatch"] for r in groups[mu]]))
    axes[0].errorbar(mus, switches, yerr=switch_se, marker="o", lw=1.1,
                     capsize=2, color="#1f77b4")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"FD radius $\mu$")
    axes[0].set_ylabel(r"switch time $T_{\rm sw}$")
    axes[0].set_title("(a) Smoothing / certificate")
    for x, y, rate in zip(mus, switches, mismatch):
        axes[0].annotate(f"{rate:.0f}%", (x, y), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=5.5)
    axes[0].text(0.02, 0.03, "labels: endpoint mismatch", transform=axes[0].transAxes,
                 fontsize=5.5)

    successful = [r for r in margin_rows if r["delta_star"] is not None]
    colors = {0.02: "#440154", 0.05: "#3b528b", 0.1: "#21918c",
              0.25: "#5ec962", 0.5: "#fde725"}
    for a_min, rows in grouped(successful, "a_min").items():
        axes[1].scatter([r["delta_star"] for r in rows],
                        [r["switch_time"] for r in rows], s=10, alpha=.72,
                        color=colors.get(float(a_min)), label=f"{float(a_min):g}")
    axes[1].set_xlabel(r"measured margin $\delta_\star$")
    axes[1].set_ylabel(r"switch time $T_{\rm sw}$")
    axes[1].set_title("(b) Identification margin")
    axes[1].legend(title=r"$a_{\min}$", ncol=2, frameon=False, handletextpad=.2,
                   columnspacing=.6, title_fontsize=6)

    selected = [r for r in rate_rows
                if r["objective_name"] == "ridge_logistic" and r["beta"] == 0.99]
    groups = grouped(selected, "target")
    targets = np.asarray(sorted(groups, reverse=True), dtype=float)
    ratios, ratio_se = [], []
    for target in targets:
        mean, se = mean_se([r["query_ratio"] for r in groups[target]])
        ratios.append(mean)
        ratio_se.append(se)
    axes[2].errorbar(targets, ratios, yerr=ratio_se, marker="o", lw=1.1,
                     capsize=2, color="#d62728", label="ZO-IDEP / Full")
    kd = float(selected[0]["k_over_d"])
    axes[2].axhline(kd, color="black", ls="--", lw=.8, label=r"$k/d$")
    axes[2].set_xscale("log")
    axes[2].invert_xaxis()
    axes[2].set_xlabel(r"target distance $\varepsilon$")
    axes[2].set_ylabel("function-query ratio")
    axes[2].set_title(r"(c) Local oracle ratio ($\beta=.99$)")
    axes[2].legend(frameon=False, loc="best")

    for axis in axes:
        axis.grid(True, alpha=.2, lw=.5)
    fig.tight_layout(w_pad=1.2)
    RESULT_FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULT_FIGURES / "theory_diagnostics.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIGURES / "theory_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def build_rate_accuracy():
    rows = records("exp05_rate_and_accuracy")
    selected = [r for r in rows
                if r["objective_name"] == "ridge_logistic" and r["beta"] == 0.99]
    groups = grouped(selected, "target")
    targets = np.asarray(sorted(groups, reverse=True), dtype=float)
    full_means, full_se, idep_means, idep_se, ratios, ratio_se = [], [], [], [], [], []
    for target in targets:
        sample = groups[target]
        for destination, errors, key in [
                (full_means, full_se, "full_queries_target"),
                (idep_means, idep_se, "idep_queries_target"),
                (ratios, ratio_se, "query_ratio")]:
            mean, se = mean_se([r[key] for r in sample])
            destination.append(mean)
            errors.append(se)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.25))
    axes[0].errorbar(targets, full_means, yerr=full_se, marker="o", capsize=2,
                     label="Full-ZO-EP")
    axes[0].errorbar(targets, idep_means, yerr=idep_se, marker="s", capsize=2,
                     label="ZO-IDEP")
    axes[0].set_ylabel("function queries")
    axes[0].legend(frameon=False)
    axes[1].errorbar(targets, ratios, yerr=ratio_se, marker="o", capsize=2)
    axes[1].axhline(selected[0]["k_over_d"], color="black", ls="--", lw=.8,
                    label=r"$k/d$")
    axes[1].set_ylabel("ZO-IDEP / Full")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel(r"target distance $\varepsilon$")
        axis.grid(True, alpha=.2, lw=.5)
    fig.tight_layout()
    fig.savefig(RESULT_FIGURES / "rate_accuracy.pdf", bbox_inches="tight")
    plt.close(fig)



def _paper_axis():
    fig, axis = plt.subplots(figsize=(2.25, 1.65))
    axis.grid(True, alpha=.2, lw=.5)
    axis.tick_params(labelsize=8)
    axis.xaxis.label.set_size(9)
    axis.yaxis.label.set_size(9)
    return fig, axis


def build_paper_panels():
    """Build the three Fig. 1 panels at their final two-column print size."""
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    mu_rows = records("exp02_mu_sweep")
    groups = grouped(mu_rows, "mu")
    mus = np.asarray(sorted(groups), dtype=float)
    rates = np.asarray([100.0 * np.mean([r["switched"] for r in groups[mu]])
                        for mu in mus])
    fig, axis = _paper_axis()
    axis.plot(mus, rates, marker="o", lw=1.4, ms=4)
    axis.set_xscale("log")
    axis.set_ylim(-5, 105)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xlabel(r"Finite-difference radius $\mu$")
    axis.set_ylabel("Certification rate (%)")
    fig.tight_layout(pad=.05)
    fig.savefig(PAPER_FIGURES / "mu_certification.pdf", bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)

    margin_rows = records("exp03_margin_sweep")
    groups = grouped(margin_rows, "a_min")
    deltas, switches, switch_se = [], [], []
    for a_min in sorted(groups):
        mean, _ = mean_se([r["delta_star"] for r in groups[a_min]])
        deltas.append(mean)
        mean, se = mean_se([r["switch_time"] for r in groups[a_min]])
        switches.append(mean)
        switch_se.append(se)
    all_deltas = finite([r["delta_star"] for r in margin_rows])
    all_switches = finite([r["switch_time"] for r in margin_rows])
    correlation = float(np.corrcoef(all_deltas, all_switches)[0, 1])
    fig, axis = _paper_axis()
    axis.errorbar(deltas, switches, yerr=switch_se, marker="o", lw=1.4,
                  ms=4, capsize=2)
    axis.set_xlabel(r"Measured identification margin $\delta_\star$")
    axis.set_ylabel(r"Mean switch iteration $T_{\rm sw}$")
    axis.text(.98, .96, rf"$r={correlation:.3f}$", transform=axis.transAxes,
              ha="right", va="top", fontsize=8)
    fig.tight_layout(pad=.05)
    fig.savefig(PAPER_FIGURES / "margin_switch.pdf", bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)

    rate_rows = [r for r in records("exp05_rate_and_accuracy")
                 if r["objective_name"] == "ridge_logistic" and r["beta"] == 0.99]
    groups = grouped(rate_rows, "target")
    targets = np.asarray(sorted(groups, reverse=True), dtype=float)
    ratios, ratio_se = [], []
    for target in targets:
        mean, se = mean_se([r["query_ratio"] for r in groups[target]])
        ratios.append(mean)
        ratio_se.append(se)
    fig, axis = _paper_axis()
    axis.errorbar(targets, ratios, yerr=ratio_se, marker="o", lw=1.4,
                  ms=4, capsize=2)
    kd = float(rate_rows[0]["k_over_d"])
    axis.axhline(kd, color="black", ls="--", lw=.8)
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_ylim(0, 1.05)
    axis.set_xlabel(r"Target accuracy $\varepsilon$")
    axis.set_ylabel(r"$Q_{\rm IDEP}/Q_{\rm full}$")
    axis.text(.98, .08, rf"$k/d={kd:.4f}$", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=8)
    fig.tight_layout(pad=.05)
    fig.savefig(PAPER_FIGURES / "accuracy_query_ratio.pdf", bbox_inches="tight",
                pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    build_theory_diagnostics()
    build_rate_accuracy()
    build_paper_panels()
