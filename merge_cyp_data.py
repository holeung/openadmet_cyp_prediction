"""
merge_cyp_data.py

Builds a new CSV from two files:

  * the blinded test file (e.g. cyp-challenge-TEST-BLINDED.csv), from which
    ONLY the molecule name column (e.g. `Molecule_Name`) is kept. Its other
    columns are not written to the output; its SMILES column is read solely
    to verify the match, and
  * a reference file containing the SMILES strings and the four columns
        CYP1A2_pIC50_direct_inhibition_median
        CYP2C9_pIC50_direct_inhibition_median
        CYP2D6_pIC50_direct_inhibition_median
        CYP3A4_pIC50_direct_inhibition_median

Rows are paired up and the SMILES from the two files are compared for every
entry. Only rows whose SMILES agree are written to the output; everything else
is listed in a separate problems report so nothing is silently dropped.

How rows are paired (--pair-on, default auto):
  * If the reference file has a fully filled name column (override the column
    name with --key), rows are paired on that name.
  * If that column is missing or has blanks, rows are paired on the SMILES
    string itself (compared in canonical form when RDKit is available). The
    name written to the output always comes from the test file.

How SMILES are compared, per row:
  * ok_exact       identical strings
  * ok_canonical   different strings, same molecule once canonicalized with
                    RDKit (e.g. "OCC" vs "CCO"). Needs RDKit.
  * smiles_mismatch  the files disagree about this molecule -> NOT written
  * missing_in_reference / unparseable_smiles / ... -> NOT written

Requirements:
    pip install pandas rdkit

Usage:
    python merge_cyp_data.py --test cyp-challenge-TEST-BLINDED.csv \
        --reference cyp_data_with_smiles.csv --output merged.csv

    # exact string comparison only (no RDKit needed):
    python merge_cyp_data.py --test ... --reference ... --output merged.csv --exact
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:  # handled in main()
    HAVE_RDKIT = False

CYP_COLUMNS = [
    "CYP1A2_pIC50_direct_inhibition_median",
    "CYP2C9_pIC50_direct_inhibition_median",
    "CYP2D6_pIC50_direct_inhibition_median",
    "CYP3A4_pIC50_direct_inhibition_median",
]


def find_column(df: pd.DataFrame, wanted: str, file_label: str, required: bool = True):
    """Find a column by name, ignoring case and surrounding whitespace."""
    for c in df.columns:
        if c.strip().lower() == wanted.strip().lower():
            return c
    if required:
        raise SystemExit(
            f"Column '{wanted}' not found in {file_label}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def canonical(smiles, use_rdkit: bool):
    """Canonical SMILES, or None if missing/unparseable."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    smiles = smiles.strip()
    if not use_rdkit:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def classify(row, use_rdkit: bool) -> str:
    """Compare the two SMILES strings of a paired row."""
    s_test, s_ref = row["smiles_test"], row["smiles_ref"]
    if pd.isna(s_ref) and pd.isna(row["_in_reference"]):
        return "missing_in_reference"
    if not isinstance(s_test, str) or not isinstance(s_ref, str):
        return "smiles_missing"
    if s_test.strip() == s_ref.strip():
        return "ok_exact"
    if not use_rdkit:
        return "smiles_mismatch"
    c_test, c_ref = canonical(s_test, True), canonical(s_ref, True)
    if c_test is None or c_ref is None:
        return "unparseable_smiles"
    return "ok_canonical" if c_test == c_ref else "smiles_mismatch"


def main():
    p = argparse.ArgumentParser(
        description="Merge molecule_name (blinded test file) with SMILES and CYP "
        "pIC50 columns (reference file), verifying that SMILES agree."
    )
    p.add_argument("--test", required=True, help="Blinded test CSV (provides molecule_name).")
    p.add_argument("--reference", required=True, help="CSV with SMILES and the four CYP columns.")
    p.add_argument("--output", "-o", required=True, help="Output CSV path.")
    p.add_argument("--key", default="molecule_name",
                   help="Column used to pair rows if present in both files (default: molecule_name).")
    p.add_argument("--pair-on", choices=["auto", "name", "smiles"], default="auto",
                   help="How to pair rows. auto (default): on the name column if the reference "
                   "file has a fully filled one, otherwise on SMILES.")
    p.add_argument("--smiles-column", default="smiles",
                   help="SMILES column name in both files (case-insensitive, default: smiles).")
    p.add_argument("--test-smiles-column", default=None, help="SMILES column in the test file, if different.")
    p.add_argument("--reference-smiles-column", default=None, help="SMILES column in the reference file, if different.")
    p.add_argument("--exact", action="store_true",
                   help="Compare SMILES as exact strings only (no RDKit canonicalization).")
    p.add_argument("--report", default=None,
                   help="Where to write rows that could not be verified "
                   "(default: <output>_problems.csv; only written if there are problems).")
    p.add_argument("--strict", action="store_true",
                   help="Exit with an error status if any row could not be verified.")
    args = p.parse_args()

    use_rdkit = HAVE_RDKIT and not args.exact
    if not args.exact and not HAVE_RDKIT:
        print("Warning: RDKit not installed; SMILES are compared as exact strings only. "
              "Install it with 'pip install rdkit' for canonical comparison.")

    test = pd.read_csv(args.test)
    ref = pd.read_csv(args.reference)

    name_col_t = find_column(test, args.key, args.test, required=False)
    name_col_r = find_column(ref, args.key, args.reference, required=False)
    if name_col_t is None:
        raise SystemExit(
            f"'{args.key}' not found in {args.test}; it is needed for the output. "
            f"Available columns: {list(test.columns)}"
        )
    smi_t = find_column(test, args.test_smiles_column or args.smiles_column, args.test)
    smi_r = find_column(ref, args.reference_smiles_column or args.smiles_column, args.reference)
    cyp_cols = [find_column(ref, c, args.reference) for c in CYP_COLUMNS]

    t = pd.DataFrame({
        "molecule_name": test[name_col_t].astype(str).str.strip(),
        "smiles_test": test[smi_t],
    })
    r = ref[[smi_r] + cyp_cols].copy()
    r.columns = ["smiles_ref"] + CYP_COLUMNS
    r["_in_reference"] = 1

    if args.pair_on == "name" and name_col_r is None:
        raise SystemExit(f"--pair-on name requested but '{args.key}' is not in {args.reference}.")
    use_name = name_col_r is not None and args.pair_on != "smiles"
    if use_name and args.pair_on == "auto":
        n_blank = int(ref[name_col_r].isna().sum() + (ref[name_col_r].astype(str).str.strip() == "").sum())
        if n_blank:
            print(f"Note: '{name_col_r}' in the reference file is blank for {n_blank} of {len(ref)} rows, "
                  "so rows are paired on SMILES instead.")
            use_name = False

    if use_name:
        mode = f"'{name_col_r}'"
        r["molecule_name"] = ref[name_col_r].astype(str).str.strip()
        on = "molecule_name"
    else:
        mode = "SMILES" + (" (canonical)" if use_rdkit else " (exact string)")
        t["_canon"] = t["smiles_test"].map(lambda s: canonical(s, use_rdkit))
        r["_canon"] = r["smiles_ref"].map(lambda s: canonical(s, use_rdkit))
        r = r[r["_canon"].notna()]
        on = "_canon"
        n_bad = t["_canon"].isna().sum()
        if n_bad:
            print(f"Warning: {n_bad} test row(s) have a missing/unparseable SMILES and cannot be matched.")

    dup = r.duplicated(on, keep=False)
    if dup.any():
        print(f"Warning: {r.loc[dup, on].nunique()} key(s) appear more than once in the reference file; "
              "keeping the first occurrence of each.")
        r = r.drop_duplicates(on, keep="first")

    print(f"Pairing rows on {mode}. Test rows: {len(t)} | reference rows: {len(r)} | "
          f"SMILES comparison: {'canonical (RDKit)' if use_rdkit else 'exact string'}")

    merged = t.merge(r, on=on, how="left")  # keeps the test file's row order
    if on == "_canon":
        merged = merged.drop(columns="_canon")
    merged["status"] = merged.apply(lambda row: classify(row, use_rdkit), axis=1)

    ok = merged["status"].str.startswith("ok")
    counts = merged["status"].value_counts()
    print("Row status counts:\n" + counts.to_string())

    # Reference rows never used (informational).
    used = set(merged.loc[merged["_in_reference"].notna(), "molecule_name"]) if on == "molecule_name" else None
    if used is not None:
        unused = len(set(r["molecule_name"]) - used)
        if unused:
            print(f"Note: {unused} reference row(s) have no matching test entry and were ignored.")

    # Only the name column is taken from the test file, and it keeps the header
    # spelling used there (e.g. "Molecule_Name"). Nothing else from the test
    # file is written; its SMILES column is used only for the comparison.
    out = merged.loc[ok, ["molecule_name", "smiles_ref"] + CYP_COLUMNS].rename(
        columns={"smiles_ref": smi_r, "molecule_name": name_col_t}
    )
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} verified rows to {args.output}")

    problems = merged.loc[~ok, ["molecule_name", "smiles_test", "smiles_ref", "status"]].rename(
        columns={"molecule_name": name_col_t}
    )
    if len(problems):
        report = args.report or str(Path(args.output).with_suffix("")) + "_problems.csv"
        problems.to_csv(report, index=False)
        print(f"WARNING: {len(problems)} row(s) could NOT be verified and were left out. "
              f"Details in {report}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
