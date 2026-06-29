# Submission Outputs

Place final CSV / parquet submission files here.

## Recommended Upload

Use:

```text
activity_predictions_final.csv
```

Why this file:

- it has exactly 513 rows
- it has the required `SMILES`, `Molecule Name`, `pEC50` columns
- it has no NaN or infinite pEC50 values
- it exact-fills the 253 revealed Phase 1 molecules
- its hidden/test predictions come from the guarded residual-calibrated model

Do not use direct Phase 1 scoring on this file as the honest metric, because
the exact-filled rows contaminate that direct score. Use the nested-CV estimate
for the guarded model and the clean baseline audit for the non-exact comparison.

Typical outputs from the audit and selection scripts:

- `phase1_honest_metrics_report.csv`
- `best_honest_submission.csv`
- `activity_predictions_final.csv`

The audit script will also print which files were skipped as contaminated exact-filled
Phase 1 candidates.
