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
submissions/submission_activity_phase2.csv
```

This is a canonical copy of:

```text
submissions/rank1_tail_active_guarded.csv
```

`rank1_tail_active_guarded.csv` and `MOONSHOT_FINAL_RECOMMENDED.csv` are
byte-for-byte identical in this repo. The guarded file is exact-filled on the
253 revealed Phase 1 rows, so it is not valid for direct honest Phase 1 scoring,
but it is the best supported upload candidate because its residual-calibrated
hidden/test predictions are backed by the nested-CV diagnostic.

The best clean non-exact baseline from direct Phase 1 audit is:

```text
submissions/ens_cm_lr3e-04_3seed_sur_w0.325.csv
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
`submission_activity_phase2.csv` is the final upload file.

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
    build_ensemble.py
    enumerate_mmps.py
    make_cv_splits.py
    run_honest_audit.py
    select_best_candidate.py
    train_chemeleon.py
    train_cv.py
    train_final.py
    train_inactive_weight.py
  src/openadmet_pxr_repo/
    audit.py
    io.py
    metrics.py
    selection.py
  submissions/
    README.md
    submission_activity_phase2.csv
  weights/
    ensemble_manifest.json
```

## Model Summary

The core ensemble combines:

- **Suiren iw2**: the conformation-aware / graph-based component represented by
  `iw2_3seed_ep17-23.csv`.
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
python scripts/build_ensemble.py --w-cm 0.325 --evaluate
```

4. Run the honest audit:

```bash
python scripts/run_honest_audit.py --root .
```

5. Select the best non-contaminated candidate:

```bash
python scripts/select_best_candidate.py --root .
```

## Honest Metric Policy

The audit scripts enforce these rules:

- score only against the 253 revealed `phase1_unblinded.csv` molecules
- skip files that exact-fill Phase 1 labels
- report MAE and RAE
- report paired bootstrap confidence intervals
- keep hidden Phase 2 estimates separate from measured Phase 1 metrics

This matters because full-fit residual calibration can look excellent while
overfitting the small revealed set.

## Access Note

The GitHub URL `Jefflinnnn/openadmet_pxr` returned a 404 from this environment,
so I mirrored the structure from your local downloaded copy at:

```text
C:\Users\naira7\Downloads\openadmet_pxr-main\openadmet_pxr-main
```

If you want me to sync directly from GitHub in a future pass, you will need to
make the repo visible to this session or provide a token/zip export.
