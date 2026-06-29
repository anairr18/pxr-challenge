# OpenADMET PXR Activity Prediction Submission Report

## Executive Summary

This repository contains the activity-prediction submission package for the
OpenADMET PXR Induction Challenge. The recommended upload file is:

```text
submissions/activity_predictions_final.csv
```

The final model uses an ensemble of complementary molecular predictors followed
by a conservative residual-calibration step. The best clean, directly auditable
baseline is:

```text
submissions/activity_predictions_clean_baseline.csv
```

The clean baseline measured on the 253 revealed Phase 1 molecules gives:

```text
MAE ~= 0.437
RAE ~= 0.577
```

The final upload file includes the revealed Phase 1 labels for submission
formatting, so it should not be directly scored as an honest Phase 1 model. Its
defensible performance statement is based on the guarded nested-validation
estimate:

```text
MAE ~= 0.424
RAE ~= 0.559
```

## Challenge Context

Pregnane X receptor (PXR, NR1I2) is a nuclear receptor involved in xenobiotic
metabolism and drug-drug interaction risk. The activity track asks participants
to predict pEC50 values for 513 compounds. During the competition, 253 of these
compounds were revealed as Phase 1 data. Those revealed compounds are useful for
diagnosis, but they also create a leakage risk if they are used as a direct
score for a final submission file.

The analysis therefore separates:

- clean validation on non-exact predictions
- guarded estimates for calibrated final predictions
- final submission formatting

## Data Sources

The modeling and audit workflow used the official OpenADMET training, test, and
revealed Phase 1 files, along with the provided counter-assay and multitask
assay tables. Public human PXR pharmacology from ChEMBL was also evaluated as an
external biological signal.

## Modeling Approach

The core predictor combines two complementary molecular modeling views:

- a conformation-aware molecular predictor, which helps with broad chemical
  similarity and shape-sensitive behavior
- a graph-based multitask molecular predictor, which uses structure and related
  assay context to predict PXR activity

The strongest clean baseline used a weighted blend of these two predictors. This
baseline was selected because it was the best non-exact file under direct Phase
1 audit.

After the core ensemble was selected, a conservative residual-calibration layer
was evaluated. This layer was intentionally constrained so it could correct
systematic bias without memorizing the 253 revealed Phase 1 labels.

## Validation Guardrails

The validation policy was designed to avoid overclaiming performance from the
small revealed set.

The main guardrails were:

- exact-filled Phase 1 rows are excluded from honest direct scoring
- nested cross-validation is used for residual calibration
- residual corrections are capped
- improvements are checked fold by fold
- paired bootstrap confidence intervals are reported
- full-fit diagnostics are not treated as headline metrics

The paired bootstrap resamples the revealed `(truth, prediction)` pairs together.
This gives an uncertainty interval for the revealed Phase 1 audit set. It does
not directly measure the hidden Phase 2 set.

## What Worked

### Complementary Molecular Ensemble

The strongest clean baseline came from combining two complementary molecular
predictors rather than relying on a single model family. The clean baseline
achieved approximately:

```text
MAE ~= 0.437
RAE ~= 0.577
```

### Conservative Residual Calibration

A constrained residual layer improved the estimated final submission behavior.
The residual model was treated as a calibration step, not a standalone model,
and it was accepted only under nested-validation guardrails.

### Counter-Assay and Multitask Context

Counter-assay and multitask labels provided useful biological context. They were
most useful as supporting signals for model selection and residual calibration,
not as standalone replacements for the main ensemble.

### External Public PXR Signal

Public human PXR pharmacology from ChEMBL target `CHEMBL3401` was tested as a
genuinely independent signal. It slightly improved the clean baseline:

```text
Clean baseline:      MAE ~= 0.4373, RAE ~= 0.5773
External-signal OOF: MAE ~= 0.4328, RAE ~= 0.5712
Improved folds:      4/5
```

The gain was real but small, and the combined final-push experiment did not pass
the replacement gate. The guarded final submission was therefore retained.

## What Did Not Work

### Direct Scoring of Final Submission Files

Submission files that include exact revealed labels can appear artificially
strong under direct Phase 1 scoring. Those files are valid for upload format,
but they are not valid direct validation measurements.

### Aggressive Residual Fitting

More flexible residual models improved full-fit diagnostics but failed
nested-validation checks. The revealed Phase 1 set is small enough that
aggressive calibration can overfit inactive-tail and highly active compounds.

### Standalone Descriptor and Tree Models

Fingerprint, descriptor, and tree-based models were useful diagnostics but did
not outperform the molecular ensemble as standalone candidates. Their common
failure mode was compression toward the middle of the activity range.

### Tail-Specific Gating

The main error pattern was biologically plausible: very inactive compounds were
often overpredicted, while highly active compounds were sometimes underpredicted.
Tail-specific correction was tested but did not improve enough validation folds
to justify replacing the guarded final model.

## Final Submission

Recommended file:

```text
submissions/activity_predictions_final.csv
```

Submission sanity check:

```text
rows: 513
columns: SMILES, Molecule Name, pEC50
NaN/inf: none
prediction range: 1.745 to 6.720
SHA256: 207f0bda20ecf5a15f22f0e992235eb252ed3fb736a740a51a4676545a4c2f57
```

The final file should be described as a guarded residual-calibrated ensemble
submission with an estimated MAE of about 0.424 and RAE of about 0.559 from
nested validation. The clean directly measurable baseline is about MAE 0.437 and
RAE 0.577.

## Short Description

The submitted model combines complementary molecular predictors with a
conservative residual-calibration layer. Validation was leakage-aware: files
that exact-filled revealed Phase 1 rows were excluded from honest direct scoring,
and improvements were accepted only when supported by nested cross-validation.
An external ChEMBL human-PXR signal was also tested; it improved the clean
baseline slightly, but not enough to replace the guarded final submission.

## References

- OpenADMET PXR Challenge data: https://huggingface.co/datasets/openadmet/pxr-challenge-train-test
- ChEMBL human PXR target `CHEMBL3401`: https://www.ebi.ac.uk/chembl/
- Chemprop: Heid et al., Journal of Chemical Information and Modeling, 2024.
