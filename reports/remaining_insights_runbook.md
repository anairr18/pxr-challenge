# Remaining Insights Runbook

The current recommended upload remains:

```text
submissions/activity_predictions_final.csv
```

The remaining-insights experiments test additional signals but do not replace
the final submission unless the nested-CV gate passes.

## Restart-Proof Colab Cell

Use this in a fresh Colab runtime:

```python
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/anairr18/pxr-challenge.git"
ROOT = Path("/content/pxr-challenge")

def run(cmd, cwd=None):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, text=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)

run([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "scikit-learn", "rdkit", "lightgbm", "xgboost", "pyarrow"])

if ROOT.exists():
    run(["git", "-C", str(ROOT), "pull", "--ff-only"])
else:
    run(["git", "clone", REPO_URL, str(ROOT)])

run([
    sys.executable,
    "scripts/run_remaining_insights_experiments.py",
    "--root",
    ".",
    "--n-boot",
    "5000",
], cwd=ROOT)

print("\nRecommended file remains unless the summary says a replacement passed:", flush=True)
print(ROOT / "submissions" / "activity_predictions_final.csv", flush=True)
print("\nSummary:", flush=True)
print((ROOT / "reports" / "remaining_insights_summary" / "experiment_summary.json").read_text()[:4000])
```

## Fast Smoke Cell

Use this to confirm the runtime and dependencies:

```python
import subprocess
import sys
from pathlib import Path

ROOT = Path("/content/pxr-challenge")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "scipy", "scikit-learn", "rdkit", "lightgbm", "xgboost", "pyarrow"], check=True)
subprocess.run(["git", "clone", "https://github.com/anairr18/pxr-challenge.git", str(ROOT)], check=False)
subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only"], check=False)
subprocess.run([sys.executable, "scripts/run_remaining_insights_experiments.py", "--root", ".", "--smoke"], cwd=ROOT, check=True)
```

## Interpretation

Use `activity_predictions_final.csv` unless
`reports/remaining_insights_summary/experiment_summary.json` reports a
candidate with decision:

```text
REAL_IMPROVEMENT_REVIEW_BEFORE_REPLACEMENT
```

or:

```text
SUB040_CREDIBLE_REVIEW_FOR_REPLACEMENT
```

If no candidate passes, the new results are exploratory diagnostics only.
