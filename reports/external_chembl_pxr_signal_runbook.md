# External ChEMBL PXR Signal Runbook

## Why This Is The Next Real Signal

The current final model is anchored on complementary molecular predictions plus guarded
Phase 1 residual calibration. Most additional candidates tested so far were not
independent enough: descriptor-only models, representation extensions, and tree
residuals either tracked the anchor too closely or failed fold-wise validation.

The first genuinely independent signal to test is public human PXR pharmacology
from ChEMBL target `CHEMBL3401`, which corresponds to human NR1I2 / PXR. The
experiment downloads public PXR activity records, curates activation-like and
inhibition-like potency labels, trains external-only auxiliary predictors, and
uses those predictions as features in a capped residual model on top of the
locked molecular-ensemble anchor.

## Command

```bash
python scripts/run_external_pxr_signal_experiment.py --root . --n-boot 5000
```

## Acceptance Policy

Do not replace `submissions/activity_predictions_final.csv` unless
the report says `REAL_IMPROVEMENT_REVIEW_BEFORE_REPLACEMENT` or
`SUB040_CREDIBLE_REVIEW_FOR_REPLACEMENT`.

The test must beat the anchor in nested CV. Full-fit or exact-filled Phase 1
scores are not valid evidence.

## Expected Outputs

- `reports/external_chembl_pxr_signal_experiment/experiment_report.md`
- `reports/external_chembl_pxr_signal_experiment/experiment_summary.json`
- `reports/external_chembl_pxr_signal_experiment/fold_metrics.csv`
- `reports/external_chembl_pxr_signal_experiment/inner_config_scores.csv`
- `submissions/external_chembl_pxr_signal_oof_candidate.csv`
- `submissions/external_chembl_pxr_signal_upload_candidate.csv`

## Scientific Interpretation

If this helps, it means public PXR biology contains an orthogonal activity signal
that the molecular ensemble did not learn from the OpenADMET challenge chemistry. If it
fails or is highly correlated with the anchor, then the remaining realistic path
is receptor-structure docking/interaction fingerprints, not more Phase 1
residual tuning.
