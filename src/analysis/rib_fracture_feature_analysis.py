"""
Rib fracture feature analysis template.

This file is intended for tabular analysis of rib fracture extent/location
and delayed hemothorax labels.

Suggested patient-level features:
- total fractured rib count
- count of ribs with fracture extent >= 50%
- count of ribs with fracture extent == 100%
- R1-R7 and R8-R12 group counts
- anterior/lateral/posterior location counts
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=False)
    parser.add_argument("--output_csv", type=str, default="rib_fracture_features.csv")
    args = parser.parse_args()

    if args.input_csv is None:
        print("Template only. Provide --input_csv to run analysis.")
        return

    df = pd.read_csv(args.input_csv)
    # TODO: adapt to the local rib-fracture table format.
    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
