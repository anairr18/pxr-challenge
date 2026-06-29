# OpenADMET PXR - Leakage-Aware Suiren/CheMeleon Ensemble

This repository is structured after the reference OpenADMET PXR challenge repo:

- `data/`
- `models/`
- `scripts/`
- `submissions/`
- `reports/`

The modeling story is intentionally honest-first: the main submission is built
around a Suiren/CheMeleon neural ensemble, with optional guarded residual
calibration and a strict audit layer that rejects exact-filled Phase 1 files.

## Final Submission Recommendation

The file I would upload to the activity track is:

```text
submissions/openadmet_pxr_activity_final_submission.csv
```

This is a canonical copy of:

```text
submissions/guarded_residual_calibrated_submission.csv
```

`guarded_residual_calibrated_submission.csv` and
`guarded_residual_calibrated_submission_duplicate.csv` are byte-for-byte
identical in this repo. The guarded file is exact-filled on the 253 revealed
Phase 1 rows, so it is not valid for direct honest Phase 1 scoring, but it is
the best supported upload candidate because its residual-calibrated hidden/test
predictions are backed by the nested-CV diagnostic.

The best clean non-exact baseline from direct Phase 1 audit is:

```text
submissions/suiren_chemeleon_blend_weight_0p325_predictions.csv
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
`openadmet_pxr_activity_final_submission.csv` is the final upload file.

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
    suiren/
      README.md
      weights/
        README.md
  reports/
    writeup.md
  scripts/
    assemble_neural_ensemble.py
    create_phase1_cross_validation_splits.py
    enumerate_matched_molecular_pairs.py
    prepare_final_submission.py
    run_cross_validation_audit.py
    run_phase1_honest_audit.py
    select_best_submission_candidate.py
    train_chemeleon_multitask_model.py
    train_suiren_inactive_tail_weighted_model.py
  src/openadmet_pxr_repo/
    audit.py
    io.py
    metrics.py
    selection.py
  submissions/
    README.md
    openadmet_pxr_activity_final_submission.csv
  weights/
    ensemble_manifest.json
```

## Model Summary

The core ensemble combines:

- **Suiren inactive-tail weighted model**: the conformation-aware / graph-based
  component represented by
  `suiren_inactive_tail_weighted_three_seed_predictions.csv`.
- **CheMeleon**: Chemprop/CheMeleon-style molecular graph neural predictions,
  averaged over seeds when available.
- **Blend**: `pEC50 = w * CheMeleon + (1 - w) * Suiren`, with `w=0.325` as the
  current clean candidate from the bundle audit.

Counter-assay and multitask signals were tested as auxiliary biological context.
Tree models such as LightGBM, XGBoost, and ExtraTrees were useful only as
guarded residual correctors, not as standalone replacements for the neural core.

## Environment Setup

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
```

If you use `uv`, the lightweight project file is included:

```bash
uv sync
```

Large model checkpoints are not committed. Put them under:

```text
models/suiren/weights/
~/.chemprop/chemeleon_mp.pt
```

## Reproducing the Submission Shape

1. Put challenge CSVs in `data/`.
2. Put component prediction CSVs in `submissions/`.
3. Build the ensemble:

```bash
python scripts/assemble_neural_ensemble.py --w-cm 0.325 --evaluate
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
whether a new independent signal can beat the Suiren/CheMeleon anchor without
overfitting the 253 revealed Phase 1 rows.

Run it only in an environment with RDKit and scikit-learn installed:

```bash
python scripts/run_sub040_signal_experiment.py --root . --n-boot 5000
```

Optional, slower 3D descriptor mode:

```bash
python scripts/run_sub040_signal_experiment.py --root . --with-3d --n-boot 5000
```

This command writes:

- `reports/sub040_structure_assay_experiment/experiment_report.md`
- `reports/sub040_structure_assay_experiment/experiment_summary.json`
- `reports/sub040_structure_assay_experiment/fold_metrics.csv`
- `reports/sub040_structure_assay_experiment/region_metrics.csv`
- `submissions/structure_assay_residual_oof_candidate.csv`
- `submissions/structure_assay_residual_upload_candidate.csv`

The experiment deliberately does **not** replace
`submissions/openadmet_pxr_activity_final_submission.csv`. A new candidate is
eligible for replacement only if the nested-CV acceptance gate passes.

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
