# OpenADMET PXR Activity Prediction Submission Report

## Executive Summary

This repository contains a leakage-aware submission package for the OpenADMET
PXR activity prediction track. The final recommended upload is:

```text
submissions/openadmet_pxr_activity_final_submission.csv
```

The submission is based on a Suiren/CheMeleon neural core with guarded residual
calibration. The best clean non-exact baseline is:

```text
submissions/suiren_chemeleon_blend_weight_0p325_predictions.csv
```

That clean baseline measured on the 253 revealed Phase 1 molecules gives:

```text
MAE ~= 0.437
RAE ~= 0.577
```

The final upload file exact-fills the 253 revealed Phase 1 rows, so it is not
valid for direct Phase 1 scoring. Its defensible performance statement is the
guarded residual-calibration estimate from nested validation:

```text
MAE ~= 0.424
RAE ~= 0.559
```

The practical conclusion is: use the final guarded submission file for upload,
but report the clean Suiren/CheMeleon baseline as the direct non-contaminated
Phase 1 measurement.

## Challenge Context

Pregnane X receptor (PXR, NR1I2) is a nuclear receptor that regulates drug
metabolism and transporter genes, especially pathways relevant to drug-drug
interaction risk. The activity track asks participants to predict pEC50 values
for 513 blinded compounds. During the competition, 253 of these compounds became
unblinded as Phase 1 data. These 253 molecules are useful for model diagnosis,
but they create a major overfitting risk: direct scoring of any final CSV that
contains exact Phase 1 labels is contaminated.

The repository therefore separates three concepts:

- clean non-exact Phase 1 measurements
- guarded residual calibration estimates
- final upload formatting

## Data Used

The main files used in the modeling/audit package are:

- `data/pxr-challenge_TRAIN.csv`
- `data/pxr-challenge_TEST_BLINDED.csv`
- `data/phase1_unblinded.csv`
- `data/pxr-challenge_counter-assay_TRAIN.csv`
- `data/pxr-challenge_single_concentration_TRAIN.csv`
- `data/multitask_train.csv`

The Phase 1 file contains the 253 revealed molecules from the blinded test set.
The full test file contains 513 molecules. The hidden Phase 2 subset cannot be
directly scored from public files.

## Core Model

The best clean model family was a Suiren/CheMeleon hybrid.

Suiren provided a conformation-aware molecular signal. It was used as a strong
base predictor, especially for broad chemical similarity and inactive-tail
behavior.

CheMeleon/Chemprop provided graph-neural molecular property predictions. The
best CheMeleon-style component used multitask information, including PXR curve
fit labels and related assay signals where available. The useful ensemble form
was a simple blend:

```text
pEC50 = 0.325 * CheMeleon + 0.675 * Suiren
```

The corresponding clean prediction file is:

```text
submissions/suiren_chemeleon_blend_weight_0p325_predictions.csv
```

This blend was selected because it was the strongest non-exact, directly
auditable candidate among the available Suiren/CheMeleon blend weights.

## Guarded Residual Calibration

The final upload candidate adds a conservative residual-calibration layer to the
neural core. The purpose of this layer is to correct systematic errors without
letting a flexible model memorize the 253 revealed Phase 1 labels.

The guardrails were:

- nested cross-validation over the revealed Phase 1 molecules
- rejection of exact-filled files for honest metric ranking
- capped residual corrections
- fold-wise improvement checks
- paired bootstrap confidence intervals
- separate treatment of full-fit diagnostics and honest OOF estimates

This procedure produced the final guarded candidate:

```text
submissions/openadmet_pxr_activity_final_submission.csv
```

Because the file exact-fills Phase 1 rows for final submission formatting, its
direct Phase 1 score is not reported as an honest metric. The honest statement is
the nested-CV-supported guarded estimate of about MAE 0.424 and RAE 0.559.

## Validation Policy

The validation policy is intentionally conservative.

Files are excluded from honest direct scoring if they exact-fill Phase 1 labels.
The audit scripts check for exact matches between predictions and the 253
revealed pEC50 values.

For honest candidates, metrics are computed on paired `(truth, prediction)`
observations from the 253 revealed molecules:

- MAE
- RAE using the fixed leaderboard-style denominator
- paired bootstrap confidence intervals

Bootstrap intervals resample the 253 paired observations with replacement. This
estimates uncertainty in the revealed Phase 1 audit set, not hidden Phase 2
performance.

## What Worked

### Suiren/CheMeleon Blending

The strongest clean result came from combining Suiren and CheMeleon rather than
using either component alone. The best blend weight was approximately 0.325 on
CheMeleon and 0.675 on Suiren.

Clean Phase 1 measurement:

```text
MAE ~= 0.437
RAE ~= 0.577
```

### Counter-Assay and Multitask Context

Counter-assay and multitask labels were useful as biological context. They were
tested both as multitask training signals and as auxiliary features for residual
models. They helped inform model design, but they did not by themselves produce
a replacement-level final candidate.

### Conservative Tree Residuals

Tree models such as LightGBM/XGBoost/ExtraTrees-style residual correctors were
useful only when heavily constrained. Unconstrained tree models can fit the 253
revealed molecules too closely, so the final approach used capped corrections
and nested validation.

### External ChEMBL PXR Signal

A final independent-signal push tested public human PXR pharmacology from
ChEMBL target `CHEMBL3401` / NR1I2. This was genuinely independent of the OCNT
challenge curve-fit labels.

The external ChEMBL signal improved the clean Suiren/CheMeleon anchor slightly:

```text
Anchor:        MAE ~= 0.4373, RAE ~= 0.5773
ChEMBL OOF:    MAE ~= 0.4328, RAE ~= 0.5712
Improved folds: 4/5
```

However, the improvement was small and the ChEMBL-corrected predictions remained
highly correlated with the anchor. A later combined big-push experiment did not
pass the replacement gate, so the final guarded submission was retained.

## What Did Not Work

### Exact-Fill Scoring

Any CSV that exact-fills the 253 revealed Phase 1 values can look artificially
excellent when directly scored. Those files are valid as final submission
formats but invalid as honest audit measurements.

### Aggressive Residual Fitting

Aggressive residual stacking improved full-fit metrics but did not survive
nested-CV guardrails. The small 253-molecule Phase 1 set makes overfitting easy,
especially in the inactive tail and active extreme.

### Broad RDKit/Tree Standalone Models

RDKit descriptors, fingerprints, and tree-based models were useful diagnostics,
but they did not produce a robust standalone replacement for the neural core.
The common failure mode was prediction compression toward the middle of the
activity range.

### UniMol/ChemBERTa Extension Attempts

Additional representation models such as UniMol and ChemBERTa-style extensions
were tested as possible independent feature blocks. In the tested setup they did
not clear the nested-CV threshold and were not used to replace the final model.

### Tail/Active Gating

The main model error mode was chemically plausible: very inactive compounds were
often overpredicted, and highly active compounds were sometimes underpredicted.
A tail/active classifier-gated correction was tested to fix this compression,
but it failed to improve enough folds and was rejected.

## Final Submission Decision

The final decision was to keep:

```text
submissions/openadmet_pxr_activity_final_submission.csv
```

Sanity check:

```text
rows: 513
columns: SMILES, Molecule Name, pEC50
NaN/inf: none
prediction range: 1.745 to 6.720
SHA256: 207f0bda20ecf5a15f22f0e992235eb252ed3fb736a740a51a4676545a4c2f57
```

The final file should be described as a guarded residual-calibrated
Suiren/CheMeleon submission with an estimated MAE of about 0.424 and RAE of
about 0.559 from nested validation. The clean directly measurable baseline is
the non-exact Suiren/CheMeleon 0.325 blend at about MAE 0.437 and RAE 0.577.

## Recommended Language For Reporting

> I used a Suiren/CheMeleon neural ensemble as the core predictor and added a
> conservative residual-calibration layer under nested cross-validation. I kept
> exact-filled Phase 1 files out of honest metric ranking and used paired
> bootstrap intervals for uncertainty. The final upload is the guarded
> residual-calibrated submission, while the clean non-exact Suiren/CheMeleon
> blend provides the direct Phase 1 audit baseline. I also tested public ChEMBL
> human-PXR pharmacology as a genuinely independent signal; it improved the
> clean anchor slightly, but not enough to replace the guarded final submission.

## References

- OpenADMET PXR Challenge data: https://huggingface.co/datasets/openadmet/pxr-challenge-train-test
- ChEMBL human PXR target `CHEMBL3401`: https://www.ebi.ac.uk/chembl/
- Chemprop: Heid et al., Journal of Chemical Information and Modeling, 2024.
- CheMeleon pretrained molecular representation model: https://arxiv.org/abs/2506.15792
- Suiren-ConfAvg molecular representation approach: https://arxiv.org/abs/2603.21942
