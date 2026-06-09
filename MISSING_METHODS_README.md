# Controlled RED Fusion Methods Implemented

This overlay supplies a self-contained controlled benchmark script implementing
the previously missing comparison methods:

- `oracle_correlated_red`
- `red_if_correct`
- `red_if_wrong`
- `parameter_ci`
- `red_ci_esr`
- `red_gci_esr`

It also keeps the existing comparison methods `red_normal`, `red_ci`, and
`red_ici`, plus the covariance-union safety variants `red_cu` and
`red_ci_cu`, so the full method table can be run from one script.

## Files

```text
tools/evaluation/run_controlled_unknown_corr_benchmark.py
tests/test_controlled_red_methods.py
.github/workflows/evaluate-red-controlled-methods.yml
```

## Local Smoke Test

```bash
python -m pip install numpy scipy matplotlib shapely pytest
python -m pytest tests/test_controlled_red_methods.py
python tools/evaluation/run_controlled_unknown_corr_benchmark.py \
  --suite all \
  --trials 20 \
  --rho-values 0.00,0.75,0.99 \
  --gamma-values 0.50,1.00,2.00 \
  --methods oracle_correlated_red,red_normal,red_if_correct,red_if_wrong,parameter_ci,red_ci,red_ici,red_ci_esr,red_gci_esr,red_cu,red_ci_cu \
  --omega-grid-size 9 \
  --out-dir evaluation_results/controlled_smoke
```

## Method Definitions In Brief

- `oracle_correlated_red`: uses the hidden cross-covariance in the controlled synthetic setup; this is an unattainable upper-bound baseline.
- `red_if_correct`: subtracts the correct common information once, then fuses information increments.
- `red_if_wrong`: same as RED-IF, but subtracts a scaled or mis-specified common information matrix.
- `parameter_ci`: applies covariance intersection directly in the raw parameter chart, without RED chart alignment.
- `red_ci_esr`: aligns the equivalent RED chart first, then chooses the CI weight by an ESR/square-root-shape uncertainty objective.
- `red_gci_esr`: fuses over multiple equivalent RED charts, uses ESR-based CI weights, and uses Chernoff/geometric-mean normalizers as component compatibility weights.
- `red_cu`: applies covariance-union safety inflation unconditionally after RED-aware CI.
- `red_ci_cu`: applies covariance-union safety inflation only when the compatibility gate marks tracks as inconsistent.

`oracle_correlated_red`, `red_if_correct`, and `red_if_wrong` are controlled
benchmark implementations. They require synthetic hidden common-information or
cross-covariance quantities, so they are not meant as deployable sensor-network
algorithms unless those quantities are available or modeled.
