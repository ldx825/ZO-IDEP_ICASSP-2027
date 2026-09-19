"""Reference algorithms with exact query accounting and theorem diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .certificate import certificate
from .diagnostics import stationarity_witness
from .prox import prox_trimmed_l1, stable_topk


@dataclass
class RunConfig:
    iterations: int = 300
    eta: float | None = None
    lam: float = 0.025
    mu0: float = 1e-3
    mu_min: float = 0.0
    beta: float = 1.0
    decay_start: int = 0
    continuation: int = 100
    cert_start: int | None = None
    support_tol: float = 1e-10
    numerical_tol: float = 1e-12
    shadow_tol: float = 1e-8
    szoht_q: int | None = None
    szoht_step_scale: float = 0.2
    seed: int = 0
    shadow_audit: bool = True
    certificate_eps_mode: str = "smooth"


def _lambda_at(t: int, config: RunConfig) -> float:
    if config.continuation <= 0:
        return config.lam
    ramp = min(1.0, (t + 1) / config.continuation)
    return (config.lam / 40.0) * (40.0 ** ramp)


def _mu_at(t: int, config: RunConfig) -> float:
    exponent = max(0, t - config.decay_start)
    # A positive floor avoids catastrophic cancellation when a long geometric
    # schedule reaches radii below floating-point function-value resolution.
    return max(config.mu_min, config.mu0 * (config.beta ** exponent))


def _support(x: np.ndarray, tolerance: float) -> np.ndarray:
    return np.flatnonzero(np.abs(x) > tolerance)


def run_algorithm(objective, k: int, method: str, config: RunConfig,
                  planted: np.ndarray | None = None,
                  planted_support: np.ndarray | None = None,
                  x0: np.ndarray | None = None,
                  oracle_support: np.ndarray | None = None,
                  distance_reference: np.ndarray | None = None):
    """Run one method; exact gradients are used only for diagnostics/PG-EP."""
    method = method.upper().replace("-", "_")
    allowed = {"PG_EP", "FULL_ZO_EP", "COORD_ZO_IHT", "SZOHT_RD",
               "ZO_IDEP", "STABLE_3", "STABLE_5", "STABLE_10",
               "ORACLE_ENDSUPPORT"}
    if method not in allowed:
        raise ValueError(f"unknown method {method}")
    d = objective.dimension
    eta = 0.9 / objective.L if config.eta is None else config.eta
    x = np.zeros(d) if x0 is None else np.asarray(x0, dtype=float).copy()
    rng = np.random.default_rng(config.seed)
    queries = 0
    diagnostic_queries = 0
    switched = False
    switch_time = config.iterations
    switch_support = None
    switch_certificate = None
    trace = []
    support_history: list[tuple[int, ...]] = []
    maximum_shadow_mismatch = 0.0
    maximum_zo_error = 0.0
    maximum_error_ratio = 0.0
    error_bound_violations = 0

    stable_required = int(method.split("_")[1]) if method.startswith("STABLE_") else None
    cert_start = config.continuation if config.cert_start is None else config.cert_start

    for t in range(config.iterations):
        mu = _mu_at(t, config)
        lam_t = _lambda_at(t, config)
        support_now = _support(x, config.support_tol)
        # Heuristic freezes monitor the selected top-k set even while the
        # exact-penalty iterate still has a nonzero tail.  The certified method
        # separately requires the mathematical support to have size exactly k.
        support_tuple = tuple(int(i) for i in stable_topk(x, k))
        support_history.append(support_tuple)

        if method == "PG_EP":
            ghat = objective.gradient(x)
            query_increment = 0
        elif method == "SZOHT_RD":
            q_directions = config.szoht_q or 2 * (3 * k + 2)
            ghat, query_increment = objective.random_direction_fd(x, mu, q_directions, rng)
        elif switched:
            ghat, query_increment, _ = objective.coord_fd(x, mu, switch_support)
        else:
            ghat, query_increment, _ = objective.coord_fd(x, mu)
        queries += query_increment

        cert_details = None
        if (method == "ZO_IDEP" and not switched and t >= cert_start
                and support_now.size == k and lam_t >= config.lam * (1.0 - 1e-12)):
            m_value = objective.restricted_m(support_now, x)
            L_value = max(float(objective.L), float(m_value))
            if config.certificate_eps_mode == "zero_quadratic":
                if objective.name != "sparse_regression":
                    raise ValueError("zero_quadratic certificate bound requires a quadratic objective")
                eps = 0.0
            elif config.certificate_eps_mode == "smooth":
                eps = L_value * mu / 2.0
            else:
                raise ValueError(f"unknown certificate_eps_mode={config.certificate_eps_mode}")
            passed, cert_details = certificate(
                x, ghat, support_now, eta, config.lam, L_value, m_value, eps,
                config.numerical_tol,
            )
            if passed:
                switched = True
                switch_time = t
                switch_support = support_now.copy()
                switch_certificate = cert_details.copy()

        if not switched and stable_required is not None:
            if (len(support_history) >= stable_required
                    and len(set(support_history[-stable_required:])) == 1):
                switched = True
                switch_time = t
                switch_support = np.asarray(support_history[-1], dtype=int)

        if (method == "ORACLE_ENDSUPPORT" and not switched and oracle_support is not None
                and support_now.size == k
                and np.array_equal(np.sort(support_now), np.sort(oracle_support))):
            switched = True
            switch_time = t
            switch_support = support_now.copy()

        shadow_ghat = None
        shadow_next = None
        shadow_mismatch = math.nan
        uses_restricted_step = switched and method in {
            "ZO_IDEP", "STABLE_3", "STABLE_5", "STABLE_10", "ORACLE_ENDSUPPORT"
        }
        if uses_restricted_step:
            x_next = np.zeros_like(x)
            x_next[switch_support] = x[switch_support] - eta * ghat[switch_support]
            if config.shadow_audit:
                if query_increment == 2 * d:
                    shadow_ghat = ghat.copy()
                else:
                    shadow_ghat, shadow_count, _ = objective.coord_fd(x, mu)
                    diagnostic_queries += shadow_count
                shadow_next, _ = prox_trimmed_l1(
                    x - eta * shadow_ghat, eta, config.lam, k
                )
                shadow_mismatch = float(np.max(np.abs(x_next - shadow_next)))
                maximum_shadow_mismatch = max(maximum_shadow_mismatch, shadow_mismatch)
                if method == "ZO_IDEP" and shadow_mismatch > config.shadow_tol:
                    raise RuntimeError(
                        f"critical certificate failure at t={t}: shadow mismatch "
                        f"{shadow_mismatch:.3e} > {config.shadow_tol:.3e}"
                    )
        elif method in {"COORD_ZO_IHT", "SZOHT_RD"}:
            step = (config.szoht_step_scale / objective.L
                    if method == "SZOHT_RD" else eta)
            z = x - step * ghat
            active = stable_topk(z, k)
            x_next = np.zeros_like(x)
            x_next[active] = z[active]
        else:
            x_next, _ = prox_trimmed_l1(x - eta * ghat, eta, lam_t, k)

        exact_gradient = objective.gradient(x)
        diagnostic_ghat = shadow_ghat if shadow_ghat is not None else ghat
        zo_error = float(np.max(np.abs(diagnostic_ghat - exact_gradient)))
        error_bound = float(objective.L * mu / 2.0)
        maximum_zo_error = max(maximum_zo_error, zo_error)
        if error_bound > 0:
            maximum_error_ratio = max(maximum_error_ratio, zo_error / error_bound)
        if objective.name != "phase_retrieval" and zo_error > error_bound + 1e-10:
            error_bound_violations += 1

        if method in {"COORD_ZO_IHT", "SZOHT_RD"}:
            w_norm = math.nan
        else:
            witness_ghat = shadow_ghat if shadow_ghat is not None else ghat
            w_norm = stationarity_witness(objective, x, x_next, witness_ghat, eta)

        next_support = stable_topk(x_next, k)
        overlap = math.nan
        exact_support = math.nan
        if planted_support is not None:
            overlap = len(np.intersect1d(next_support, planted_support)) / k
            exact_support = float(np.array_equal(next_support, np.sort(planted_support)))
        recovery_error = math.nan
        if planted is not None:
            if objective.name == "phase_retrieval":
                recovery_error = float(min(np.linalg.norm(x_next - planted),
                                           np.linalg.norm(x_next + planted)))
            else:
                recovery_error = float(np.linalg.norm(x_next - planted))
        reference_distance = (float(np.linalg.norm(x_next - distance_reference))
                              if distance_reference is not None else math.nan)

        trace.append({
            "iteration": t + 1, "queries": queries, "diagnostic_queries": diagnostic_queries,
            "mu": mu, "lambda": lam_t, "objective": objective.value(x_next),
            "recovery_error": recovery_error, "support_size": int(_support(x_next, config.support_tol).size),
            "support": [int(i) for i in next_support],
            "reference_distance": reference_distance,
            "support_overlap": overlap, "exact_support": exact_support,
            "zo_error_inf": zo_error, "zo_error_bound": error_bound,
            "zo_error_ratio": zo_error / error_bound if error_bound > 0 else 0.0,
            "w_norm": w_norm, "switched": int(switched),
            "shadow_mismatch_inf": shadow_mismatch,
            "chi": cert_details["chi"] if cert_details is not None else math.nan,
            "margin_invariance": (cert_details["margin_invariance"]
                                  if cert_details is not None else math.nan),
            "margin_threshold": (cert_details["margin_threshold"]
                                 if cert_details is not None else math.nan),
            "margin_rank": cert_details["margin_rank"] if cert_details is not None else math.nan,
        })
        x = x_next

    final_support = stable_topk(x, k)
    result = {
        "method": method, "objective": objective.value(x), "queries": int(queries),
        "diagnostic_queries": int(diagnostic_queries), "switched": int(switched),
        "switch_time": int(switch_time), "switch_support": (switch_support.tolist()
                                                               if switch_support is not None else []),
        "support_size": int(_support(x, config.support_tol).size),
        "maximum_shadow_mismatch_inf": float(maximum_shadow_mismatch),
        "maximum_zo_error_inf": float(maximum_zo_error),
        "maximum_zo_error_ratio": float(maximum_error_ratio),
        "error_bound_violations": int(error_bound_violations),
        "certificate_at_switch": switch_certificate,
        "endpoint": x,
        "trace": trace,
    }
    if planted_support is not None:
        result["support_overlap"] = float(len(np.intersect1d(final_support, planted_support)) / k)
        result["exact_support"] = float(np.array_equal(final_support, np.sort(planted_support)))
    if planted is not None:
        if objective.name == "phase_retrieval":
            result["recovery_error"] = float(min(np.linalg.norm(x - planted), np.linalg.norm(x + planted)))
        else:
            result["recovery_error"] = float(np.linalg.norm(x - planted))
    return result
