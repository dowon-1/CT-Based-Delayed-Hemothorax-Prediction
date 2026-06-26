"""
Standalone checkpoint evaluation template.

The current training scripts already perform final evaluation after training.
This file is included to keep the repository structure clean and to provide
a place for future validation/test-only evaluation code.

TODO:
- Import the desired model class from src/training or a refactored src/models module.
- Load the H5 index CSV.
- Load a saved checkpoint.
- Collect validation probabilities.
- Select validation-based threshold.
- Apply the same threshold to the test split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.metrics import binary_classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=False, help="CSV with columns: id, split, label, prob")
    parser.add_argument("--out_json", type=str, default="evaluation_summary.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.pred_csv is None:
        print("This is a template. Provide --pred_csv if prediction probabilities are already available.")
        return

    df = pd.read_csv(args.pred_csv)
    results = {}
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        results[split] = binary_classification_metrics(
            sub["label"].values,
            sub["prob"].values,
            threshold=args.threshold,
        )

    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
