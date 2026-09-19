# ZO-IDEP execution and Definition-of-Done report

Date: 2026-09-09 (UTC)

## Outcome

PASS.  The complete experiment suite ran on frozen, disjoint report seeds.  All
3,120 expected summary records and all 3,120 raw records are present.  The
strict correctness gate passed: the maximum post-certificate shadow-full
one-step mismatch and the maximum paired ZO-IDEP/Full endpoint gap are both
exactly `0.0` in the stored double-precision results.

## Implementation and tests

- Implemented deterministic trimmed-l1 prox with index-stable ties, coordinate
  central finite differences with exact query counts, the full certificate and
  its three margins/slack, six comparison/safety algorithms, analytical-only
  diagnostics, and all Exp00--Exp07 drivers.
- Exp00 passes all five required checks: brute-force prox enumeration through
  dimension 8, random/tie cases, FD values/counts, certificate margins, and the
  shadow-equivalence gate.
- Nonlinear deterministic FD checks have 0 bound violations.  Analytical
  gradients and shadow queries never enter reported function-query totals.
- A development run exposed catastrophic function-value cancellation after a
  geometric radius fell to about `1e-15`.  The frozen rate protocol therefore
  records `mu_min=1e-6`; after this correction every strict certificate passes
  the shadow gate.  This is a numerical implementation safeguard, not a change
  to the scientific claim.

## Frozen experiment results

| Experiment | Report records | Main result |
|---|---:|---|
| Exp01 sparse regression | 750 | Main `(d,k)=(256,16)` switch `103.1 +/- 1.8`, saving `61.2%`, maximum certified endpoint gap `0`. Across five settings, 31/150 do not certify by step 300; 30 are `(1024,64)` with only 128 measurements and are retained with zero saving. |
| Exp02 smoothing | 180 | 30/30 certify for every `mu <= 1e-2`; 0/30 certify at `3e-2`. Mean switch rises from 99 to `108.2 +/- 1.1` before failure. FD-bound violations: 0; endpoint mismatches: 0. |
| Exp03 margin | 150 | All certify. Mean measured `delta_star` rises from 0.0161 to 0.0476 across the endpoint groups while mean switch falls from 123.3 to 100.2; pooled Pearson correlation is -0.695. (`delta_star` corrected to `min(gamma_star/2, tau_star)` and recomputed on 2026-09-10; switch trajectories identical.) |
| Exp04 certificate vs heuristic | 180 | In the deliberately difficult low-margin regime, Certified refuses all 30 unsafe switches and has 0 endpoint errors. Stable-3/5/10 have 14/14/12 endpoint errors out of 30; Oracle-EndSupport has 0. |
| Exp05 rate/accuracy | 1,260 | All paired targets are reached. For logistic `beta=.99`, empirical contraction is `0.9820 +/- 0.0001` versus theorem `rho=.99`; the ratio at `epsilon=1e-3` is `0.7146 +/- 0.0022`, trending downward but still above asymptotic `k/d=.0625` because pre-switch queries dominate. |
| Exp06 phase retrieval | 480 | 120/120 ZO-IDEP local runs certify, maximum endpoint/shadow gap `0`, mean saving `48.5%`. |
| Exp07 scaling | 120 | At fixed `k/d=1/16`, mean query ratios are 0.375, 0.388, 0.391, and 0.417 for dimensions 128--1024; all paired endpoints match exactly. |

The fixed-radius nonlinear distance to the high-precision restricted optimum
increases from `1.90e-10` at `mu=1e-4` to `2.34e-5` at `mu=3e-2`.  It remains
below the theorem's conservative `O(sqrt(k) mu)` neighborhood scale.  All 180
offline solutions have restricted gradient norm at most `1.52e-9`.

## Required caveats retained

- Global and local rates remain separate: the paper calls the global result an
  `O(T^-1/2)` stationarity guarantee and reserves geometric convergence and
  `O(k log(1/epsilon))` for the certified local regime.
- Exp04 is intentionally a safety example: the certificate gains no queries
  because it rejects every unsafe candidate.  Failed certificates are not
  discarded.
- Exp06 uses a planted-signal, controlled local-basin warm start.  It is clearly
  labeled as a local theorem diagnostic and cannot be claimed as deployable
  end-to-end sparse phase retrieval or evidence of global convergence.
- The finite target range in Exp05 does not reach the asymptotic `k/d` ratio;
  the paper reports the observed ratio rather than overstating the limit.

## Paper/asset acceptance

- `figures/theory_diagnostics.pdf` is a one-page vector PDF with the required
  smoothing, measured-margin, and oracle-ratio panels and mean +/- s.e. where
  applicable.
- `main.tex` contains frozen numerical results rather than future-tense
  placeholders.  Tables and the nonconvex/safety text were updated.
- `main.pdf` compiles successfully, has exactly five US-letter pages, has no
  overfull boxes or undefined references, and was inspected page by page.
  Pages 1--4 contain the technical paper; page 5 contains References only.
- `zo_idep/results/summary/validation.json` is the machine-readable final gate.

Temporary six-way Exp02 shard files were removed after their 180 records were
validated and merged into the canonical raw and summary files; the canonical
records are reproducible from the stored scripts/configs/seeds.
