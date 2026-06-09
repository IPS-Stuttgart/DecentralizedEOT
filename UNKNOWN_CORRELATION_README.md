# Unknown-correlation RED fusion add-on

This add-on extends `Fusion_2022_Thormann_RED-IF` with conservative track-to-track fusion when the cross-correlation/common information between local ellipse tracks is unknown.

The original RED-IF code is still the correct reference when the common prior and process noise are known. The unknown-correlation add-on is meant for the harder case where that information is missing, approximate, stale, or supplied by a black-box tracker.

## Added methods

### RED-CI

Component-wise covariance intersection inside the RED representation:

```text
RED component alignment -> Gaussian CI -> RED mixture reduction -> MMGW estimate
```

This is the safest baseline. It does not require a cross-covariance.

### RED-ICI

Experimental inverse covariance intersection. It is usually sharper than CI, but it should be treated as an ablation rather than the main conservative result.

### RED-GCI / RED-GCI-ESR

This improvement treats covariance intersection as a Chernoff/geometric-mean density fusion and uses the corresponding Gaussian normalization coefficient as the RED component-pair weight. This is more principled than an independence-style likelihood weight for unknown correlations.

The `ESR` variant also chooses the CI weight for the shape part in the RED/MMGW square-root geometry:

```text
[alpha, length, width] -> [S_11, S_12, S_22]
S = R(alpha) diag(length, width) R(alpha)^T
```

This aligns the weight selection with the square-root-space approximation used by the MMGW estimator.

## New options in `UnknownCorrelationFusionCenter`

```python
unknown_corr_method="ci"          # "ci" or "ici"
omega_criterion="logdet"          # backward-compatible criterion for both state parts
kin_omega_criterion="logdet"      # "logdet" or "trace"
shape_omega_criterion="esr_trace" # "logdet", "trace", "esr_trace", "esr_logdet"
component_weight_mode="chernoff"  # "likelihood", "esr_likelihood", "chernoff", "prior", "uniform"
component_pairing_mode="gated"    # "all", "best", "gated"
component_gate_log_weight=12.0
```

## Suggested paper baselines

Run the original unknown-correlation baseline:

```bash
python run_red_unknown_corr_experiment.py \
  --runs 1000 \
  --time-steps 15 \
  --scenario 1 \
  --seed 7 \
  --omega-grid-size 31 \
  --estimate-samples 1000
```

Run with the proposed RED-GCI/ESR preset:

```bash
python run_red_unknown_corr_experiment.py \
  --runs 1000 \
  --time-steps 15 \
  --scenario 1 \
  --seed 7 \
  --omega-grid-size 31 \
  --estimate-samples 1000 \
  --include-gci-esr
```

Run only the proposed preset as the main RED-CI configuration:

```bash
python run_red_unknown_corr_experiment.py \
  --runs 1000 \
  --time-steps 15 \
  --scenario 1 \
  --seed 7 \
  --shape-omega-criterion esr_trace \
  --component-weight-mode chernoff \
  --component-pairing-mode gated \
  --component-gate-log-weight 12 \
  --omega-grid-size 31
```

## Tests

```bash
python -m pytest tests/test_unknown_corr_fusion.py
```

The tests check that CI does not double-count identical estimates, that the RED square-root transform is invariant to equivalent ellipse parameterizations, that the analytic square-root Jacobian matches a finite-difference Jacobian, and that the Chernoff coefficient behaves correctly for identical Gaussian components.
