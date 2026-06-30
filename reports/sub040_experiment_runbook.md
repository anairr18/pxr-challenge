# Honest Sub-0.40 Experiment Runbook

This runbook implements the research plan without replacing the current final
submission by accident.

## Current Locked Baselines

- Clean non-exact baseline: `submissions/activity_predictions_clean_baseline.csv`
- Current final upload: `submissions/activity_predictions_final.csv`
- Clean direct Phase 1 evidence: MAE `0.4373`, RAE `0.5773`
- Guarded residual diagnostic: about MAE `0.424`, RAE `0.559`

## Experiment Command

```bash
python scripts/run_structure_assay_signal_experiment.py --root . --n-boot 5000
```

For slower ETKDG/MMFF shape descriptors:

```bash
python scripts/run_structure_assay_signal_experiment.py --root . --with-3d --n-boot 5000
```

For the current cheap orthogonal push recommended by the audit:

```bash
python scripts/run_orthogonal_signal_experiment.py --root . --n-boot 5000
```

This turns on 3D descriptors, single-concentration assay-statistic features,
weighted auxiliary heads, and matched-molecular-pair cliff features. For a quick
smoke test, add `--no-3d --n-boot 200`. Its candidate files are
`submissions/orthogonal_signal_oof_candidate.csv` and
`submissions/orthogonal_signal_upload_candidate.csv`.

## What It Tests

- Scaffold/bin-stratified outer folds over the 253 revealed Phase 1 molecules.
- RDKit descriptors, train-fitted ECFP/FCFP SVD components, nearest-neighbor
  activity signals, assay auxiliary predictions, and optional 3D descriptors.
- Optional single-concentration replicate/statistic features.
- Optional matched-molecular-pair cliff-risk features.
- Optional confidence-weighted auxiliary feature heads.
- Capped residual models on top of the molecular-ensemble anchor.
- Inner-fold model/config selection only, followed by outer-fold OOF scoring.
- Activity-region reports for inactive tail, low, mid, high, and active compounds.

## Acceptance Gate

A candidate is only a real improvement if:

- exact Phase 1 matches in the OOF candidate are zero
- nested-CV MAE is below `0.410`
- nested-CV RAE is below `0.540`
- at least 4 of 5 outer folds improve over the anchor
- the paired-bootstrap upper 95% MAE is below the anchor MAE

A candidate is only sub-0.40 credible if:

- nested-CV MAE is below `0.400`
- nested-CV RAE is below `0.520`
- no fold has a material degradation versus the anchor
- the improvement comes from structure/assay signal, not Phase 1 exact filling

## Output Interpretation

Use `experiment_summary.json` as the source of truth:

- `EXPLORATORY_DO_NOT_REPLACE_FINAL`: keep the current final upload.
- `REAL_IMPROVEMENT_REVIEW_BEFORE_REPLACEMENT`: inspect errors before replacing.
- `SUB040_CREDIBLE_REVIEW_FOR_REPLACEMENT`: candidate passed the strict gate.

Direct scoring of exact-filled upload files is not an honest metric.
