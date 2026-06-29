from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Placeholder for matched molecular pair enumeration."
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "pxr_train_mmps.json")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        args.out.write_text(json.dumps({"pairs": [], "note": "Add MMP enumeration here."}, indent=2))
    print(f"MMP placeholder written/read at: {args.out}")


if __name__ == "__main__":
    main()
