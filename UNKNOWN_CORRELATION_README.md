# RED fusion under unknown correlation

This add-on targets the repository:

```text
Fusion-Goettingen/Fusion_2022_Thormann_RED-IF
```

It adds a conservative RED track-to-track fusion baseline for the case where the
cross-correlation / common information between local ellipse tracks is unknown.
The original RED-IF method should remain the sharper reference when the common
prior and process information are known.  This patch is for the complementary
unknown-correlation case.

## Added files

```text
Filters/fusioncenter_unknowncorr.py
run_red_unknown_corr_experiment.py
tests/test_unknown_corr_fusion.py
```

No existing repository file has to be edited for the first test.

## What the new fusion class does

`UnknownCorrelationFusionCenter` keeps the original RED representation and MMGW
point-estimate logic, but replaces the Kalman/information-form product update by
component-wise conservative fusion:

1. Convert the incoming shape estimate to a four-component RED via `turn_mult`.
2. Align each incoming RED component to the current global RED component's angle
   chart.
3. Fuse each aligned component pair using covariance intersection:

   ```text
   C_f(omega) = inv(omega inv(C_a) + (1 - omega) inv(C_b))
   m_f(omega) = C_f(omega) [omega inv(C_a) m_a + (1 - omega) inv(C_b) m_b]
   ```

4. Select `omega` by minimizing `logdet(C_f)` or `trace(C_f)`.
5. Reduce the resulting RED mixture and compute the MMGW/ESR estimate.

An experimental ICI mode is also available with `unknown_corr_method="ici"`.
CI should be the baseline for robust unknown-correlation tests.

## Smoke test

From the repository root after copying the files:

```bash
python -m pytest tests/test_unknown_corr_fusion.py
python run_red_unknown_corr_experiment.py --runs 20 --time-steps 15 --seed 7 --estimate-samples 300
```

Expected behavior in the tests:

* Identical estimates fused by CI keep the same covariance, rather than shrinking
  to half covariance as an independent-information product would.
* RED alignment finds the equivalent axis-swapped representation of the same
  ellipse.

## Longer experiment

```bash
python run_red_unknown_corr_experiment.py \
  --runs 1000 \
  --time-steps 15 \
  --scenario 1 \
  --seed 7 \
  --omega-grid-size 31 \
  --estimate-samples 1000 \
  --include-ici
```

Outputs are written to `plots_unknown_corr/`:

```text
red_unknown_corr_errors.csv
red_unknown_corr_errors.npz
gw_error.png
iou_error.png
vel_error.png
```

## Interpretation

The comparison is not expected to show RED-CI always beating RED-IF.  In the
original paper's setup, RED-IF is allowed to use known common prior/process
information.  RED-CI deliberately assumes this information is unavailable, so it
should be judged as a conservative fallback.  The useful comparison is:

```text
RED normal fusion      : can double-count unknown common information.
RED information fusion : strong if common information is known.
RED-CI                 : robust unknown-correlation baseline.
RED-ICI                : sharper experimental unknown-correlation baseline.
```

For a publication-quality extension, add NEES/consistency plots, because the
main promise of CI is conservatism rather than always lower point-estimate error.
