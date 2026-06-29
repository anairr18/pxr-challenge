# Models Directory

This directory mirrors the reference challenge repository layout.

Large model checkpoints should be stored here locally, but are not committed by
default.

Expected optional paths:

- `models/conformation_predictor/weights/model.pt`
- `models/conformation_predictor/repo/`

Graph multitask predictor weights are usually cached outside the repo at:

```text
~/.chemprop/
```
