"""
kumo_tabular_predict.py

Reads a training CSV and a test CSV, then uses NVIDIA Kumo Tabular (large) to
make in-context predictions for the test rows, for EVERY column whose name
starts with a given prefix (default "CYP").

Kumo Tabular predicts one target at a time, so each matching column is handled
in turn: the training rows (features + that target) are given to the model as
labeled context, and the test rows are predicted. No weights are updated.
All matching columns are excluded from the features (so one CYP target is never
used to predict another), as are any --drop-columns.

For each target <T> the output gets:
  * classification -> <T>_prediction and one <T>_prob_<class> column per class
  * regression     -> <T>_prediction (mean of predicted quantiles),
                      <T>_median, <T>_q025, <T>_q975 (95% interval)

Training rows whose value for a given target is missing are skipped for that
target only. If the test CSV also contains the target columns, simple metrics
are printed per target.

Requirements (Python >= 3.11, PyTorch >= 2.7):
    pip install structured-data-models pandas numpy

Ensemble size:
    sdm runs ONE estimator unless told otherwise. This script defaults to 16
    estimators for --size large and 8 for small/medium, matching NVIDIA's
    benchmark configuration. Override with --num-estimators.

GDS / cuFile:
    This script never calls GPUDirect Storage or cuFile itself. As a precaution
    it forces KvikIO into POSIX (non-GDS) mode and hides the optional RAPIDS
    packages cudf/kvikio from sdm, so no GDS/cuFile code path is used. Use
    --allow-rapids to turn this off.

Usage:
    python kumo_tabular_predict.py --train train.csv --test test.csv \
        --drop-columns smiles --output predictions.csv

    python kumo_tabular_predict.py --train train.csv --test test.csv \
        --target-prefix CYP --task classification --output predictions.csv
"""

import argparse
import os
import sys
import traceback

# --- Keep NVIDIA GPUDirect Storage (GDS) / cuFile out of the picture --------
# This must happen BEFORE torch/sdm are imported, so it is done at the top.
#  * KVIKIO_COMPAT_MODE=ON makes KvikIO (which wraps cuFile) use ordinary POSIX
#    file I/O and never call cuFile, if anything in the environment uses it.
#  * cudf and kvikio are optional RAPIDS packages that sdm can use for GPU
#    dataframe operations. Marking them as unavailable makes sdm fall back to
#    its plain pandas/PyTorch path. Pass --allow-rapids to skip this.
os.environ.setdefault("KVIKIO_COMPAT_MODE", "ON")
if "--allow-rapids" not in sys.argv:
    for _mod in ("cudf", "kvikio"):
        sys.modules.setdefault(_mod, None)

import numpy as np
import pandas as pd
import torch

import sdm


def infer_task(y: pd.Series) -> str:
    """Heuristic used when --task is auto."""
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 10:
        return "classification"
    if y.dropna().isin([0, 1]).all():  # 0/1 stored as floats because of NaNs
        return "classification"
    return "regression"


def print_metrics(name: str, task: str, y_true: pd.Series, preds: pd.DataFrame) -> None:
    mask = y_true.notna().to_numpy()
    if mask.sum() == 0:
        return
    y_true = y_true[mask]
    pred = preds[f"{name}_prediction"][mask]
    if task == "regression":
        err = pred.to_numpy(float) - y_true.to_numpy(float)
        ss_res = float((err ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(
            f"  metrics: RMSE={np.sqrt((err ** 2).mean()):.4f}  "
            f"MAE={np.abs(err).mean():.4f}  R2={r2:.4f}  (n={mask.sum()})"
        )
    else:
        def norm(s: pd.Series) -> np.ndarray:
            # Predicted labels come from column names (strings), true labels may
            # be numeric (0.0 vs "0"); compare them on a common footing.
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().all():
                return num.astype(float).to_numpy()
            return s.astype(str).to_numpy()

        acc = (norm(pred) == norm(y_true)).mean()
        print(f"  metrics: accuracy={acc:.4f}  (n={mask.sum()})")


def predict_target(model, train, test, target, feature_cols, task, args, device):
    """Fit on the labeled training rows for `target`; return a predictions frame."""
    labeled = train.dropna(subset=[target])
    if args.max_context_rows and len(labeled) > args.max_context_rows:
        labeled = labeled.sample(args.max_context_rows, random_state=args.seed)
    labeled = labeled.reset_index(drop=True)
    n_ctx = len(labeled)

    # Features: train + test as ONE table so categorical encodings agree.
    feats = pd.concat([labeled[feature_cols], test[feature_cols]], ignore_index=True)
    feat_table = sdm.TableTensor.from_pandas(
        df=feats, stypes=sdm.infer_stypes(feats), device=device
    )
    x_context, x_query = feat_table[:n_ctx], feat_table[n_ctx:]

    y_df = labeled[[target]]
    if task == "classification" and pd.api.types.is_float_dtype(y_df[target]):
        y_df = y_df.astype({target: int})  # 0.0/1.0 -> 0/1 so classes are clean
    y_stype = "categorical" if task == "classification" else "numerical"
    y_table = sdm.TableTensor.from_pandas(
        df=y_df,
        stypes=sdm.infer_stypes(y_df, overrides={target: y_stype}),
        device=device,
    )
    y_context = y_table[:, target]

    model.fit(x=x_context, y=y_context, num_estimators=args.num_estimators)

    outs = []
    try:
        with torch.no_grad():
            for batch in x_query.split(args.batch_size):
                outs.append(model.predict(batch).to_pandas())
    finally:
        model.clear()
    raw = pd.concat(outs, ignore_index=True)
    raw.columns = [str(c) for c in raw.columns]

    if task == "classification":
        preds = raw.add_prefix(f"{target}_prob_")
        preds.insert(0, f"{target}_prediction", raw.idxmax(axis=1).to_numpy())
    else:
        preds = pd.DataFrame(
            {f"{target}_prediction": raw.mean(axis=1), f"{target}_median": raw["q500"]}
        )
        if {"q025", "q975"} <= set(raw.columns):
            preds[f"{target}_q025"] = raw["q025"]
            preds[f"{target}_q975"] = raw["q975"]
    return preds, n_ctx


def main():
    p = argparse.ArgumentParser(
        description="In-context prediction of all CYP* columns with NVIDIA Kumo Tabular."
    )
    p.add_argument("--train", required=True, help="Training CSV (features + target columns).")
    p.add_argument("--test", required=True, help="Test CSV (features; targets optional).")
    p.add_argument("--output", "-o", required=True, help="Output predictions CSV.")
    p.add_argument(
        "--target-prefix", default="CYP",
        help="Predict every column whose name starts with this (default: CYP).",
    )
    p.add_argument(
        "--task", choices=["auto", "classification", "regression"], default="auto",
        help="Task type for all targets (default: auto-detect per target).",
    )
    p.add_argument(
        "--drop-columns", nargs="*", default=[],
        help="Columns to exclude from the features (e.g. smiles, id columns).",
    )
    p.add_argument("--size", choices=["small", "medium", "large"], default="large")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: cuda if available).")
    p.add_argument("--batch-size", type=int, default=1024, help="Query rows per predict() call.")
    p.add_argument(
        "--max-context-rows", type=int, default=None,
        help="Randomly subsample the labeled training rows per target to at most this many.",
    )
    p.add_argument(
        "--num-estimators", type=int, default=None,
        help="Ensemble size passed to fit(). Default: 16 for --size large, 8 for "
        "small/medium (the settings NVIDIA's benchmarks use). Leaving it unset "
        "in sdm gives a single estimator, which is noticeably weaker.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow-rapids", action="store_true",
        help="Do not hide the optional cudf/kvikio packages from sdm (they may use GDS/cuFile).",
    )
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if args.num_estimators is None:
        args.num_estimators = 16 if args.size == "large" else 8

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test).reset_index(drop=True)

    targets = [c for c in train.columns if c.startswith(args.target_prefix)]
    if not targets:
        raise SystemExit(
            f"No columns starting with '{args.target_prefix}' found in {args.train}."
        )
    excluded = set(targets) | set(args.drop_columns)
    # Also keep any test-only columns starting with the prefix out of the features.
    excluded |= {c for c in test.columns if c.startswith(args.target_prefix)}
    feature_cols = [c for c in train.columns if c not in excluded]
    missing = [c for c in feature_cols if c not in test.columns]
    if missing:
        raise SystemExit(f"{len(missing)} feature column(s) missing from test CSV, e.g. {missing[:5]}")

    print(
        f"{len(targets)} target(s): {targets}\n"
        f"Train rows: {len(train)} | test rows: {len(test)} | "
        f"features: {len(feature_cols)} | device: {device} | "
        f"estimators: {args.num_estimators}"
    )

    # Kumo Tabular has separate classification and regression checkpoints;
    # load only the one(s) needed, on first use.
    models = {}

    def get_model(task):
        if task not in models:
            m = sdm.models.KumoTabular(task=task, size=args.size, device=device)
            m.eval()
            models[task] = m
        return models[task]

    all_preds, failed = [], []
    for t in targets:
        n_labeled = int(train[t].notna().sum())
        if n_labeled == 0:
            print(f"[{t}] no labeled training rows; skipping.")
            failed.append(t)
            continue
        task = infer_task(train[t].dropna()) if args.task == "auto" else args.task
        print(f"[{t}] task: {task} | labeled training rows: {n_labeled}")
        try:
            preds, n_ctx = predict_target(
                get_model(task), train, test, t, feature_cols, task, args, device
            )
        except Exception as e:  # keep going so one bad target doesn't lose the rest
            print(f"[{t}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(t)
            continue
        all_preds.append(preds)
        if t in test.columns:
            print_metrics(t, task, test[t], preds)

    if not all_preds:
        raise SystemExit("All targets failed; no output written.")

    result = pd.concat([test] + all_preds, axis=1)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows x {result.shape[1]} columns to {args.output}")
    if failed:
        print(f"Warning: no predictions for: {failed}")


if __name__ == "__main__":
    main()
