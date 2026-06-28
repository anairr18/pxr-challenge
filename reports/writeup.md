# OpenADMET PXR Submission Write-up

## Executive Summary

This submission pack is built around a strict separation between:

1. honest measured performance on the 253 revealed Phase 1 molecules, and
2. heuristic hidden Phase 2 transfer estimates.

The repo is designed so that only the honest Phase 1 set is used for audit-grade
metric reporting. Files that exact-fill Phase 1 labels are retained as formatting
artifacts, but they are excluded from honest ranking.

## Data Sources

- `pxr-challenge_TRAIN.csv`
- `pxr-challenge_TEST_BLINDED.csv`
- `phase1_unblinded.csv`
- `pxr-challenge_counter-assay_TRAIN.csv`
- `multitask_train.csv`
- `iw2_3seed_ep17-23.csv`
- `ens_cm_lr3e-04_3seed_sur_w0.200.csv`
- `ens_cm_lr3e-04_3seed_sur_w0.250.csv`
- `ens_cm_lr3e-04_3seed_sur_w0.300.csv`
- `ens_cm_lr3e-04_3seed_sur_w0.325.csv`
- `ens_cm_lr3e-04_3seed_sur_w0.350.csv`
- `ens_cm_lr3e-04_3seed_sur_w0.400.csv`
- `chemeleon_mt_lr3e-04_s0.csv`
- `chemeleon_mt_lr3e-04_s1.csv`
- `chemeleon_mt_lr3e-04_s2.csv`
- `chemeleon_mt_lr3e-04_3seed.csv`

## Modeling Strategy

The modeling work that fed this repo followed a layered approach:

- a strong neural core from Suiren / CheMeleon style multi-task predictions
- residual calibration against the Phase 1 holdout
- counter-assay-informed auxiliary signals
- tree-based stacking and capped residual correction
- honest nested-CV audit before calling any improvement real

The important point is that the final honest metric comes from out-of-fold
predictions, not from full-fit training loss.

## Honest Metric Policy

We use these rules:

- exact-filled Phase 1 files are not used as honest candidates
- honest MAE is computed only on the 253 revealed Phase 1 compounds
- RAE is reported in two forms:
  - fixed leaderboard-denominator style
  - resampled-denominator bootstrap style
- confidence intervals are paired bootstrap percentile intervals

The paired bootstrap resamples the 253 `(truth, prediction)` pairs together.
That gives a proper uncertainty band for the Phase 1 audit set.

## What Was Excluded

The following files are useful for formatting or diagnostics, but not for honest
direct scoring:

- `MOONSHOT_FINAL_RECOMMENDED.csv`
- `big_push_guarded_aux_residual.csv`
- `big_push_high_upside_aux_residual.csv`
- `rank1_tail_active_guarded.csv`

Those files exact-fill the Phase 1 rows and therefore cannot be treated as
independent measurements.

## Current Honest Result

In the current bundle audit, the strongest non-contaminated CSV was:

- `ens_cm_lr3e-04_3seed_sur_w0.325.csv`

That file is the best honest candidate to carry forward if the goal is a clean
Phase 1-based comparison.

## Limitations

- Hidden Phase 2 performance cannot be directly measured from the public files.
- Any Phase 2 estimate is necessarily a transfer heuristic.
- Nested-CV numbers are much more trustworthy than full-fit numbers.

## Bottom Line

This repo is built to be honest first and aggressive second.
It gives you a clean, reproducible way to justify the submission choice without
mixing in contaminated files or overclaiming hidden-score certainty.

