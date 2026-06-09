# RED unknown-correlation evaluation workflow

This overlay adds a GitHub Actions workflow and a controlled benchmark script for
paper-style evaluation of RED fusion under known, unknown, mis-specified, and
strongly inconsistent track correlations.

## Files

```text
.github/workflows/evaluate-red-unknown-correlation.yml
tools/evaluation/run_controlled_unknown_corr_benchmark.py
run_red_unknown_corr_experiment.py
EVALUATION_WORKFLOW_README.md
```

The workflow expects the unknown-correlation fusion code to be present, including
`run_red_unknown_corr_experiment.py` and `Filters/fusioncenter_unknowncorr.py`
for the full tracking benchmark. The controlled benchmark is standalone and
implements the full method table:

| Method key | Meaning |
|---|---|
| `oracle_correlated_red` | Oracle RED-aware fusion using the hidden cross-covariance |
| `red_normal` | RED normal fusion assuming independent estimates |
| `red_if_correct` | RED information-form fusion with the correct common information |
| `red_if_wrong` | RED information-form fusion with deliberately mis-scaled common information |
| `parameter_ci` | Covariance intersection in raw orientation/axis coordinates, without RED chart alignment |
| `red_ci` | RED-aware covariance intersection |
| `red_ici` | RED-aware inverse covariance intersection |
| `red_ci_esr` | RED-CI with ESR/square-root geometry-aware omega selection |
| `red_gci_esr` | RED-GCI/ESR, all-chart CI with Chernoff component weights |
| `red_cu` | RED-CU, unconditional covariance-union safety inflation |
| `red_ci_cu` | RED-CI-CU, covariance-union safety fallback for incompatible tracks |

## Run Locally

```bash
python -m pip install numpy scipy matplotlib shapely pytest tikzplotlib pandas

python tools/evaluation/run_controlled_unknown_corr_benchmark.py \
  --suite all \
  --trials 500 \
  --rho-values 0.00,0.25,0.50,0.75,0.90,0.99 \
  --gamma-values 0.00,0.25,0.50,1.00,2.00,4.00 \
  --out-dir evaluation_results/controlled

python run_red_unknown_corr_experiment.py \
  --runs 100 \
  --time-steps 15 \
  --scenario 1 \
  --include-ici \
  --include-gci-esr \
  --include-ci-cu \
  --shape-omega-criterion esr_trace \
  --component-weight-mode chernoff \
  --component-pairing-mode gated \
  --out-dir evaluation_results/tracking/scenario_1
```

## GitHub Actions Usage

After committing these files, open the Actions tab, choose
**RED unknown-correlation evaluation**, and run it manually. The workflow also
runs a short smoke version on pushes and pull requests that touch relevant code.

Artifacts include CSV summaries, PNG plots, and a GitHub step summary.
