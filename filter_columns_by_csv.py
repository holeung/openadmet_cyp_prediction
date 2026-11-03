"""
filter_columns_by_csv.py

Reads a CSV file and writes a new CSV containing exactly the columns (fields)
found in the header of a second "reference" CSV file, in the reference file's
order.

All reference columns are kept: if one is missing from the input file, it is
still included in the output, filled with empty values (or --fill-value).
Input columns that are not in the reference file are dropped.

Requirements:
    pip install pandas

Usage:
    python filter_columns_by_csv.py --input data.csv --reference top100.csv --output filtered.csv
    python filter_columns_by_csv.py -i data.csv -r top100.csv -o filtered.csv --fill-value 0
"""

import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Write a CSV containing all columns from a reference CSV."
    )
    p.add_argument("--input", "-i", required=True, help="CSV to filter.")
    p.add_argument(
        "--reference", "-r", required=True,
        help="CSV whose column names define the output columns.",
    )
    p.add_argument("--output", "-o", required=True, help="Output CSV path.")
    p.add_argument(
        "--fill-value", default=None,
        help="Value for reference columns missing from the input "
        "(default: empty). Parsed as a number if possible.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    ref_cols = list(pd.read_csv(args.reference, nrows=0).columns)  # header only

    missing = [c for c in ref_cols if c not in df.columns]
    if missing:
        print(
            f"Warning: {len(missing)} of {len(ref_cols)} reference columns not in "
            f"input; adding them as empty/filled columns: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )

    fill = args.fill_value
    if fill is not None:
        try:
            fill = float(fill)
        except ValueError:
            pass

    out = df.reindex(columns=ref_cols, fill_value=fill) if fill is not None else df.reindex(columns=ref_cols)
    out.to_csv(args.output, index=False)
    print(f"Wrote {out.shape[0]} rows x {out.shape[1]} columns to {args.output}")


if __name__ == "__main__":
    main()
