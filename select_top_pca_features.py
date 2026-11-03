"""
select_top_pca_features.py

Reads a CSV containing fingerprint columns fp_0000 ... fp_2047, runs PCA on
those columns, ranks each column by how much it contributes to the variance
captured by the retained principal components, and writes a new CSV that
keeps only the top-N columns (default 100).

Contribution score for feature j:

    score_j = sum_k  explained_variance_ratio_k * loading_{k,j}^2

where loading_{k,j} is the weight of feature j in principal component k
(components have unit norm, so each component's squared loadings sum to 1).
A feature scores high if it has large loadings in components that explain a
lot of variance.

Requirements:
    pip install numpy pandas scikit-learn

Usage:
    python select_top_pca_features.py --input fingerprints.csv --output top100.csv
    python select_top_pca_features.py -i fingerprints.csv -o top100.csv \
        --top-k 100 --variance-threshold 0.95
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FP_PATTERN = re.compile(r"^fp_(\d{4})$")


def get_fp_columns(df: pd.DataFrame, start: int, end: int) -> list:
    """Return columns named fp_XXXX with start <= XXXX <= end, in numeric order."""
    cols = []
    for c in df.columns:
        m = FP_PATTERN.match(c)
        if m and start <= int(m.group(1)) <= end:
            cols.append((int(m.group(1)), c))
    return [c for _, c in sorted(cols)]


def main():
    p = argparse.ArgumentParser(
        description="Keep the top-N fingerprint columns by PCA variance contribution."
    )
    p.add_argument("--input", "-i", required=True, help="Input CSV path.")
    p.add_argument("--output", "-o", required=True, help="Output CSV path.")
    p.add_argument("--top-k", type=int, default=100, help="Features to keep (default 100).")
    p.add_argument("--start", type=int, default=0, help="First fp index (default 0).")
    p.add_argument("--end", type=int, default=2047, help="Last fp index (default 2047).")
    p.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Retain enough principal components to explain this fraction of "
        "total variance (default 0.95).",
    )
    p.add_argument(
        "--no-scale",
        action="store_true",
        help="Do not standardize features before PCA (default: standardize).",
    )
    p.add_argument(
        "--drop-other-columns",
        action="store_true",
        help="Output only the selected fp columns (default: also keep all "
        "non-fp columns such as 'smiles').",
    )
    p.add_argument(
        "--scores-output",
        default=None,
        help="Optional path to write a CSV of all feature scores.",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    fp_cols = get_fp_columns(df, args.start, args.end)
    if not fp_cols:
        raise SystemExit(
            f"No columns matching fp_{args.start:04d}..fp_{args.end:04d} found."
        )
    if args.top_k > len(fp_cols):
        raise SystemExit(
            f"--top-k ({args.top_k}) exceeds number of fp columns ({len(fp_cols)})."
        )

    X = df[fp_cols].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise SystemExit("Fingerprint columns contain NaN values; clean them first.")

    # Constant columns have zero variance; standardizing them would divide by
    # zero, and they contribute nothing to PCA anyway.
    if not args.no_scale:
        X = StandardScaler().fit_transform(X)

    n_samples, n_features = X.shape
    print(f"Running PCA on {n_samples} samples x {n_features} features...")
    pca = PCA(n_components=args.variance_threshold, svd_solver="full")
    pca.fit(X)
    n_comp = pca.n_components_
    print(
        f"Retained {n_comp} components explaining "
        f"{pca.explained_variance_ratio_.sum():.1%} of variance."
    )

    # score_j = sum_k evr_k * loading_kj^2
    scores = (pca.explained_variance_ratio_[:, None] * pca.components_ ** 2).sum(axis=0)

    score_df = (
        pd.DataFrame({"feature": fp_cols, "pca_variance_contribution": scores})
        .sort_values("pca_variance_contribution", ascending=False)
        .reset_index(drop=True)
    )
    selected = score_df["feature"].iloc[: args.top_k].tolist()
    # Preserve original column order in the output for readability.
    selected_ordered = [c for c in fp_cols if c in set(selected)]

    if args.drop_other_columns:
        out_df = df[selected_ordered]
    else:
        other_cols = [c for c in df.columns if c not in set(fp_cols)]
        out_df = df[other_cols + selected_ordered]

    out_df.to_csv(args.output, index=False)
    print(f"Wrote {out_df.shape[0]} rows x {out_df.shape[1]} columns to {args.output}")

    if args.scores_output:
        score_df.to_csv(args.scores_output, index=False)
        print(f"Wrote feature scores to {args.scores_output}")


if __name__ == "__main__":
    main()
