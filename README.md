# OpenADMET PXR Activity Prediction

This repository is structured after the reference OpenADMET PXR challenge repo:

- `data/`
- `models/`
- `scripts/`
- `submissions/`
- `reports/`

The modeling story is intentionally honest-first: the main submission is built
around a complementary molecular ensemble with guarded residual calibration and
a strict audit layer that rejects exact-filled Phase 1 files for direct scoring.

## Final Submission Recommendation

The file I would upload to the activity track is:

```text
submissions/activity_predictions_final.csv
```

This is the canonical final upload candidate. The final file is exact-filled on
the 253 revealed Phase 1 rows, so it is not valid for direct honest Phase 1
scoring, but it is the best supported upload candidate because its
residual-calibrated hidden/test predictions are backed by the nested-CV
diagnostic.

The best clean non-exact baseline from direct Phase 1 audit is:

```text
submissions/activity_predictions_clean_baseline.csv
```

Its honest direct Phase 1 audit result is:

```text
MAE ~= 0.437
RAE ~= 0.577
```

The guarded residual-calibrated model reached a stronger nested-CV diagnostic
estimate, approximately:

```text
MAE ~= 0.424
RAE ~= 0.559
```

So the clean blend is the baseline evidence, while
`activity_predictions_final.csv` is the final upload file.

## Repository Layout

```text
openadmet_pxr_submission_repo/
  README.md
  pyproject.toml
  requirements.txt
  data/
    README.md
  models/
    README.md
    conformation_predictor/
      README.md
      weights/
        README.md
  reports/
    writeup.md
  scripts/
    build_molecular_ensemble.py
    create_phase1_cross_validation_splits.py
    enumerate_matched_molecular_pairs.py
    prepare_final_submission.py
    run_cross_validation_audit.py
    run_phase1_honest_audit.py
    select_best_submission_candidate.py
    train_conformation_predictor.py
    train_graph_multitask_predictor.py
  src/openadmet_pxr_repo/
    audit.py
    io.py
    metrics.py
    selection.py
  submissions/
    README.md
    activity_predictions_final.csv
  weights/
    ensemble_manifest.json
```

## Model Summary

The core ensemble combines:

- a conformation-aware molecular predictor
- a graph-based multitask molecular predictor
- a conservative residual-calibration layer selected under nested validation

The clean baseline file in `submissions/activity_predictions_clean_baseline.csv`
is the direct Phase 1 audit reference. The final upload file in
`submissions/activity_predictions_final.csv` is the guarded submission candidate.

Counter-assay and multitask signals were tested as auxiliary biological context.
Tree models such as LightGBM, XGBoost, and ExtraTrees were useful only as
guarded residual correctors, not as standalone replacements for the molecular
ensemble.

## Environment Setup

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
```

If you use `uv`, the lightweight project file is included:

```bash
uv sync
```

Large model checkpoints are not committed. Put them under the corresponding
model directories if reproducing training:

```text
models/
~/.chemprop/
```

## Reproducing the Submission Shape

1. Put challenge CSVs in `data/`.
2. Put component prediction CSVs in `submissions/`.
3. Build the ensemble:

```bash
python scripts/build_molecular_ensemble.py --w-graph 0.325 --evaluate
```

4. Run the honest audit:

```bash
python scripts/run_phase1_honest_audit.py --root .
```

5. Select the best non-contaminated candidate:

```bash
python scripts/select_best_submission_candidate.py --root .
```

## Honest Sub-0.40 Experiment Gate

The repo includes an experimental structure/assay residual stack for testing
whether a new independent signal can beat the clean molecular-ensemble anchor
without overfitting the 253 revealed Phase 1 rows.

Run it only in an environment with RDKit and scikit-learn installed:

```bash
python scripts/run_structure_assay_signal_experiment.py --root . --n-boot 5000
```

Optional, slower 3D descriptor mode:

```bash
python scripts/run_structure_assay_signal_experiment.py --root . --with-3d --n-boot 5000
```

The current highest-priority cheap orthogonal push combines:

- ETKDG/MMFF 3D descriptors
- single-concentration replicate/statistic auxiliary heads
- confidence-weighted auxiliary fitting where assay uncertainty is available
- train-neighborhood matched-molecular-pair cliff features

Run the combined experiment:

```bash
python scripts/run_orthogonal_signal_experiment.py --root . --n-boot 5000
```

For a faster smoke run that skips conformer generation:

```bash
python scripts/run_orthogonal_signal_experiment.py --root . --no-3d --n-boot 200
```

The combined runner writes its own candidate files:

- `submissions/orthogonal_signal_oof_candidate.csv`
- `submissions/orthogonal_signal_upload_candidate.csv`
- `reports/orthogonal_signal_experiment/experiment_report.md`
- `reports/orthogonal_signal_experiment/experiment_summary.json`

The baseline structure/assay runner writes:

- `reports/sub040_structure_assay_experiment/experiment_report.md`
- `reports/sub040_structure_assay_experiment/experiment_summary.json`
- `reports/sub040_structure_assay_experiment/fold_metrics.csv`
- `reports/sub040_structure_assay_experiment/region_metrics.csv`
- `submissions/structure_assay_residual_oof_candidate.csv`
- `submissions/structure_assay_residual_upload_candidate.csv`

The experiment deliberately does **not** replace
`submissions/activity_predictions_final.csv`. A new candidate is
eligible for replacement only if the nested-CV acceptance gate passes.

## Remaining Insight Experiments

The last improvement pass tests the remaining lower-risk ideas from the project
audit before attempting expensive docking:

- curve-aware auxiliary heads from curve-fit, counter-assay, multitask, and
  single-concentration endpoints
- applicability-domain features from train-neighborhood similarity and label
  disagreement
- prediction-uncertainty shrinkage from the spread across non-exact component
  files

Run the full batch:

```bash
python scripts/run_remaining_insights_experiments.py --root . --n-boot 5000
```

Fast smoke run:

```bash
python scripts/run_remaining_insights_experiments.py --root . --smoke
```

Individual runs:

```bash
python scripts/run_curve_multitask_experiment.py --root . --n-boot 5000
python scripts/run_domain_uncertainty_gate_experiment.py --root . --n-boot 5000
```

These scripts preserve `submissions/activity_predictions_final.csv` unless a
new candidate clears the same nested-CV acceptance gate. Outputs are written to:

- `reports/remaining_insights_summary/experiment_summary.json`
- `reports/remaining_insights_curve_multitask/`
- `reports/remaining_insights_domain_gate/`
- `submissions/curve_multitask_upload_candidate.csv`
- `submissions/domain_uncertainty_gate_upload_candidate.csv`

## Genuinely Independent Signal Test

Most prior additions were not independent enough: descriptor trees,
representation extensions, counter-assay residuals, and tail gates were all
either weak, over-correlated with the clean anchor, or unstable across folds.

The strongest next independent signal is public human PXR pharmacology from
ChEMBL target `CHEMBL3401` / NR1I2. This is not trained from the OCNT challenge
curve-fit labels. It is an external PXR biology source that can be converted
into activation-like and inhibition-like auxiliary predictions, then tested
against the frozen anchor under the same nested-CV gate.

Run:

```bash
python scripts/run_external_pxr_signal_experiment.py --root . --n-boot 5000
```

This writes:

- `reports/external_chembl_pxr_signal_experiment/experiment_report.md`
- `reports/external_chembl_pxr_signal_experiment/experiment_summary.json`
- `submissions/external_chembl_pxr_signal_oof_candidate.csv`
- `submissions/external_chembl_pxr_signal_upload_candidate.csv`

The upload candidate is experimental. Keep
`submissions/activity_predictions_final.csv` locked unless the external-signal
report returns a review-for-replacement decision.

## Honest Metric Policy

The audit scripts enforce these rules:

- score only against the 253 revealed `phase1_unblinded.csv` molecules
- skip files that exact-fill Phase 1 labels
- report MAE and RAE
- report paired bootstrap confidence intervals
- keep hidden Phase 2 estimates separate from measured Phase 1 metrics

This matters because full-fit residual calibration can look excellent while
overfitting the small revealed set.

The stricter sub-0.40 gate requires:

- zero exact-filled Phase 1 matches in the honest OOF vector
- nested-CV MAE below `0.410` and RAE below `0.540` for a real improvement claim
- nested-CV MAE below `0.400` and RAE below `0.520` for a sub-0.40 claim
- improvement in at least 4 of 5 outer folds
- paired-bootstrap upper 95% MAE below the current clean baseline point estimate

## Access Note

The GitHub URL `Jefflinnnn/openadmet_pxr` returned a 404 from this environment,
so I mirrored the structure from your local downloaded copy at:

```text
C:\Users\naira7\Downloads\openadmet_pxr-main\openadmet_pxr-main
```

If you want me to sync directly from GitHub in a future pass, you will need to
make the repo visible to this session or provide a token/zip export.
