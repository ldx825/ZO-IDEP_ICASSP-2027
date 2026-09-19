#!/usr/bin/env python3
"""Reproducible synthetic experiments for ZO-IDEP.

The implementation uses only function values for every method labelled ZO.
Coordinate finite differences are evaluated in a vectorized batch, but every
point f(x +/- mu e_i) is counted as one oracle query.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def topk_support(x: np.ndarray, k: int) -> np.ndarray:
    return np.sort(np.argpartition(np.abs(x), -k)[-k:])


def trimmed_l1_prox(z: np.ndarray, threshold: float, k: int) -> np.ndarray:
    support = topk_support(z, k)
    out = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
    out[support] = z[support]
    return out


@dataclass
class LeastSquaresOracle:
    A: np.ndarray
    y: np.ndarray
    queries: int = 0

    def value(self, x: np.ndarray, count: bool = True) -> float:
        if count:
            self.queries += 1
        residual = self.A @ x - self.y
        return 0.5 * float(residual @ residual) / self.A.shape[0]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.A.T @ (self.A @ x - self.y) / self.A.shape[0]

    def coord_fd(self, x: np.ndarray, mu: float,
                 support: np.ndarray | None = None) -> np.ndarray:
        idx = np.arange(x.size) if support is None else np.asarray(support)
        residual = self.A @ x - self.y
        cols = self.A[:, idx]
        plus = residual[:, None] + mu * cols
        minus = residual[:, None] - mu * cols
        fp = 0.5 * np.sum(plus * plus, axis=0) / self.A.shape[0]
        fm = 0.5 * np.sum(minus * minus, axis=0) / self.A.shape[0]
        grad = np.zeros_like(x)
        grad[idx] = (fp - fm) / (2.0 * mu)
        self.queries += 2 * idx.size
        return grad

    def random_direction_fd(self, x: np.ndarray, mu: float, q: int,
                            rng: np.random.Generator) -> np.ndarray:
        d = x.size
        directions = rng.normal(size=(q, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        residual = self.A @ x - self.y
        shifted = residual[:, None] + mu * (self.A @ directions.T)
        f_plus = 0.5 * np.sum(shifted * shifted, axis=0) / self.A.shape[0]
        f_base = 0.5 * float(residual @ residual) / self.A.shape[0]
        self.queries += q + 1
        return (d / (q * mu)) * (directions.T @ (f_plus - f_base))


def make_regression(seed: int, m: int, d: int, k: int,
                    noise_std: float = 0.1):
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 1.0 / np.sqrt(k), size=(m, d))
    support = np.sort(rng.choice(d, size=k, replace=False))
    x_star = np.zeros(d)
    # A small amplitude floor makes exact-support recovery a meaningful test.
    amp = rng.uniform(0.25, 1.0, size=k)
    x_star[support] = amp * rng.choice([-1.0, 1.0], size=k)
    y = A @ x_star + rng.normal(0.0, noise_std, size=m)
    return A, y, x_star, support


def metrics(x: np.ndarray, x_star: np.ndarray, support_star: np.ndarray,
            k: int):
    estimated = topk_support(x, k)
    overlap = len(np.intersect1d(estimated, support_star)) / k
    exact = float(np.array_equal(estimated, support_star))
    return np.linalg.norm(x - x_star), overlap, exact, int(np.count_nonzero(np.abs(x) > 1e-10))


def certificate(oracle: LeastSquaresOracle, x: np.ndarray, ghat: np.ndarray,
                support: np.ndarray, eta: float, lam: float):
    As = oracle.A[:, support]
    hess_s = As.T @ As / oracle.A.shape[0]
    m_restricted = float(np.linalg.eigvalsh(hess_s)[0])
    if m_restricted <= 1e-12:
        return False, np.inf, np.inf, -np.inf
    L = float(np.linalg.norm(oracle.A, 2) ** 2 / oracle.A.shape[0])
    R = float(np.linalg.norm(ghat[support]) / m_restricted)
    inactive = np.setdiff1d(np.arange(x.size), support, assume_unique=True)
    G = float(np.max(np.abs(ghat[inactive])) + L * R) if inactive.size else L * R
    B = float(np.min(np.abs(x[support])) - R)
    return bool(G < lam and B > eta * G and B > 0.0), R, G, B


def run_method(A: np.ndarray, y: np.ndarray, x_star: np.ndarray,
               support_star: np.ndarray, k: int, method: str,
               iterations: int, lam: float, mu: float,
               continuation: int = 100, rng_seed: int = 0):
    oracle = LeastSquaresOracle(A, y)
    d = A.shape[1]
    L = float(np.linalg.norm(A, 2) ** 2 / A.shape[0])
    eta = 0.9 / L
    x = np.zeros(d)
    rng = np.random.default_rng(rng_seed)
    switched = False
    switch_support: np.ndarray | None = None
    t_id = iterations
    trace = []
    t0 = time.perf_counter()

    for t in range(iterations):
        ramp = min(1.0, (t + 1) / max(continuation, 1))
        lam_t = (lam / 40.0) * (40.0 ** ramp)
        if method == "PG-EP":
            g = oracle.gradient(x)
        elif method == "SZOHT":
            q = 2 * (3 * k + 2)
            g = oracle.random_direction_fd(x, mu, q, rng)
        elif method == "ZO-IDEP" and switched:
            g = oracle.coord_fd(x, mu, switch_support)
        else:
            g = oracle.coord_fd(x, mu)

        cert_pass = False
        R = G = np.nan
        B = np.nan
        # Reuse the full estimate already needed by the dense update.  Thus
        # certification itself incurs no additional oracle calls.
        if (method == "ZO-IDEP" and not switched and t >= continuation
                and np.count_nonzero(np.abs(x) > 1e-10) == k):
            current_support = topk_support(x, k)
            cert_pass, R, G, B = certificate(
                oracle, x, g, current_support, eta, lam
            )
            if cert_pass:
                switched = True
                switch_support = current_support.copy()
                t_id = t

        if method in ("Coord-ZO-IHT", "SZOHT"):
            x_next = np.zeros_like(x)
            step = 0.2 / L if method == "SZOHT" else eta
            z = x - step * g
            active = topk_support(z, k)
            x_next[active] = z[active]
        else:
            x_next = trimmed_l1_prox(x - eta * g, eta * lam_t, k)

        current_support = topk_support(x_next, k)

        err, overlap, exact, support_size = metrics(x_next, x_star, support_star, k)
        trace.append({
            "iteration": t + 1,
            "queries": oracle.queries,
            "objective": oracle.value(x_next, count=False),
            "recovery_error": err,
            "support_overlap": overlap,
            "exact_support": exact,
            "support_size": support_size,
            "certified": int(switched),
            "certificate_pass": int(cert_pass),
            "lambda": lam_t,
            "R": R,
            "G": G,
            "B": B,
        })
        x = x_next

    elapsed = time.perf_counter() - t0
    err, overlap, exact, support_size = metrics(x, x_star, support_star, k)
    return {
        "method": method,
        "objective": oracle.value(x, count=False),
        "recovery_error": err,
        "support_overlap": overlap,
        "exact_support": exact,
        "support_size": support_size,
        "queries": oracle.queries,
        "time_sec": elapsed,
        "t_id": t_id,
        "switched": int(switched),
        "wrong_freeze": int(switched and not np.array_equal(switch_support, support_star)),
        "trace": trace,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def aggregate(rows: list[dict]):
    groups = {}
    for row in rows:
        key = (row["d"], row["k"], row["method"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (d, k, method), vals in sorted(groups.items()):
        item = {"d": d, "k": k, "method": method, "trials": len(vals)}
        for field in ("objective", "recovery_error", "support_overlap", "exact_support",
                      "support_size", "queries", "time_sec", "t_id", "switched", "wrong_freeze"):
            data = np.asarray([v[field] for v in vals], dtype=float)
            item[field + "_mean"] = float(data.mean())
            item[field + "_std"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        summary.append(item)
    return summary


def plot_pipeline():
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 1.65))
    ax.axis("off")
    labels = ["Full CoordZO\n$2d$ queries", "Trimmed-$\\ell_1$\nexact prox",
              "Margin\ncertificate", "Support-only ZO\n$2k$ queries"]
    xs = [0.12, 0.38, 0.64, 0.88]
    colors = ["#d9e8fb", "#e6ddf5", "#fce2bd", "#d9f0df"]
    for x, label, color in zip(xs, labels, colors):
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="#333333"))
    for left, right in zip(xs, xs[1:]):
        ax.annotate("", xy=(right - 0.09, 0.5), xytext=(left + 0.09, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.text(0.76, 0.13, "certified switch", ha="center", fontsize=8, color="#8c510a")
    fig.tight_layout()
    fig.savefig(FIGURES / "pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_traces(all_rows: list[dict]):
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected = [r for r in all_rows if r["d"] == 256 and r["k"] == 16 and r["trial"] < 10]
    methods = ["SZOHT", "Full-ZO-EP", "ZO-IDEP"]
    colors = {"SZOHT": "#7570b3", "Full-ZO-EP": "#d95f02", "ZO-IDEP": "#1b9e77"}
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for method in methods:
        trials = [r for r in selected if r["method"] == method]
        if not trials:
            continue
        grid = sorted(set(q for r in trials for q in [p["queries"] for p in r["trace"]]))
        xmax = min(max(grid), 70000)
        xgrid = np.linspace(0, xmax, 180)
        curves = []
        for row in trials:
            q = np.asarray([p["queries"] for p in row["trace"]])
            e = np.asarray([p["recovery_error"] for p in row["trace"]])
            curves.append(np.interp(xgrid, q, e, left=e[0], right=e[-1]))
        curves = np.asarray(curves)
        mean = curves.mean(axis=0)
        se = curves.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else np.zeros_like(mean)
        ax.plot(xgrid, mean, label=method, color=colors[method], lw=1.5)
        ax.fill_between(xgrid, mean-se, mean+se, color=colors[method], alpha=0.16)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(r"Recovery error $\|x-x^\star\|_2$")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "query_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)

    trials = [r for r in selected if r["method"] == "ZO-IDEP"]
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    for row in trials:
        it = [p["iteration"] for p in row["trace"]]
        ov = [p["support_overlap"] for p in row["trace"]]
        ax.plot(it, ov, color="#1b9e77", alpha=0.25, lw=0.8)
        if row["switched"]:
            ax.axvline(row["t_id"], color="#d95f02", alpha=0.18, lw=0.8)
    if trials:
        n = len(trials[0]["trace"])
        mean = np.mean([[p["support_overlap"] for p in r["trace"]] for r in trials], axis=0)
        ax.plot(np.arange(1, n+1), mean, color="#006d5b", lw=2, label="mean overlap")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Support overlap")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "support_identification.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--methods", default="PG-EP,SZOHT,Coord-ZO-IHT,Full-ZO-EP,ZO-IDEP")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    trials = 5 if args.quick else args.trials
    iterations = 180 if args.quick else args.iterations
    settings = [(128, 256, 16), (128, 256, 24), (128, 256, 32),
                (256, 512, 32), (512, 1024, 64)]
    if args.quick:
        settings = settings[:2]
    methods = [name.strip() for name in args.methods.split(",") if name.strip()]
    rows = []
    trace_path = RESULTS / "traces.json"
    if args.append and trace_path.exists():
        with trace_path.open() as handle:
            rows = json.load(handle)
    for m, d, k in settings:
        for trial in range(trials):
            A, y, x_star, support = make_regression(args.seed + 1000*d + 37*k + trial, m, d, k)
            L = float(np.linalg.norm(A, 2) ** 2 / m)
            # Fixed before trials: large enough to suppress noise coordinates,
            # small enough to permit support changes during the dense phase.
            lam = 0.025
            mu = 1e-3
            for method in methods:
                result = run_method(A, y, x_star, support, k, method, iterations,
                                    lam, mu, rng_seed=args.seed + trial)
                result.update({"trial": trial, "m": m, "d": d, "k": k,
                               "lambda": lam, "mu": mu, "eta": 0.9/L})
                rows.append(result)
            print(f"finished d={d} k={k} trial={trial+1}/{trials}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    raw_fields = ["trial", "m", "d", "k", "method", "lambda", "mu", "eta",
                  "objective", "recovery_error", "support_overlap", "exact_support",
                  "support_size", "queries", "time_sec", "t_id", "switched", "wrong_freeze"]
    write_csv(RESULTS / "regression_raw.csv", rows, raw_fields)
    summary = aggregate(rows)
    summary_fields = list(summary[0])
    write_csv(RESULTS / "regression_summary.csv", summary, summary_fields)
    with trace_path.open("w") as handle:
        json.dump([{k: v for k, v in row.items() if k != "trace"} | {"trace": row["trace"]}
                   for row in rows], handle)
    plot_pipeline()
    plot_traces(rows)
    print(f"wrote {RESULTS / 'regression_summary.csv'}")


if __name__ == "__main__":
    main()
