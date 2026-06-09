# RED-CI-CU: covariance-union safety fallback

This extension adds a conservative safety valve for the unknown-correlation RED fusion experiments.

The earlier RED-CI/RED-GCI variants assume that the two local tracks refer to the same target and are mutually compatible after RED chart alignment.  That is the right model for ordinary track-to-track fusion under unknown cross-correlation.  It is not enough when there may also be hidden bias, stale tracks, data-association mistakes, or strongly inconsistent local extents.

The new variants are:

- `RED-CU`: covariance intersection followed by unconditional covariance-union-style inflation.
- `RED-CI-CU`: covariance intersection unless a compatibility gate is exceeded; if the inputs disagree strongly, inflate the fused covariance so it covers the input Gaussian components.

For a fixed fused mean `m_f` and covariance `C_f`, the safety step constructs, for each input component,

```text
C_cover_i = C_i + (m_i - m_f)(m_i - m_f)^T
```

and then scales `C_f` until it dominates all `C_cover_i` in the Loewner order.  This is not a global minimum-volume covariance-union solver, but it is deterministic, cheap, and useful as a conservative fallback.

For the RED shape state `[alpha, length, width]`, the default compatibility gate is evaluated in ESR/MMGW square-root shape space rather than raw parameter space.  This makes the gate less brittle under equivalent RED charts and near-circular ellipses.

## New command-line options

`run_red_unknown_corr_experiment.py` now accepts:

```bash
--include-ci-cu
--cu-gate-threshold <float>
--cu-inflation-margin <float>
--cu-esr-gate / --no-cu-esr-gate
```

Example:

```bash
python run_red_unknown_corr_experiment.py \
  --runs 1000 \
  --time-steps 15 \
  --scenario 1 \
  --include-ci-cu \
  --include-gci-esr \
  --shape-omega-criterion esr_trace \
  --component-weight-mode chernoff \
  --component-pairing-mode gated
```

The controlled benchmark adds two method keys:

```text
red_cu
red_ci_cu
```

These are now included in the GitHub Actions paper-method set.

## Paper positioning

`RED-CI-CU` is not intended to beat `RED-CI` in clean, correctly associated experiments.  It is intended to improve consistency and coverage in stress cases where local track estimates are mutually inconsistent.  In a paper, it should be evaluated with NEES/coverage, high-correlation sweeps, wrong-common-information sweeps, and adversarial chart/association stress tests.
