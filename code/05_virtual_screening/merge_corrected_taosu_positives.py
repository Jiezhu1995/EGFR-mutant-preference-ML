#!/usr/bin/env python3
"""Merge original and missing-hit-id regression positives and structure-deduplicate."""

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return pd.Series([None, None])
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        return pd.Series([Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                          Chem.MolToInchiKey(mol)])
    except Exception:
        return pd.Series([None, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--addendum", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(args.original); old["screen_origin"] = "original_nonempty_hit_id"
    new = pd.read_csv(args.addendum); new["screen_origin"] = "missing_hit_id_addendum"
    merged = pd.concat([old, new], ignore_index=True, sort=False)
    merged[["standardized_parent_smiles", "standardized_parent_inchikey"]] = \
        merged.smiles.apply(standardize)
    valid = merged.standardized_parent_inchikey.notna()
    merged["minimum_selectivity_fold"] = merged[["direct_selectivity_fold",
                                                  "derived_selectivity_fold"]].min(axis=1)
    ranked = merged.loc[valid].sort_values(
        ["all_four_filters_votes", "minimum_selectivity_fold", "mut_ic50_nM"],
        ascending=[False, False, True])
    dedup = ranked.drop_duplicates("standardized_parent_inchikey", keep="first").copy()
    dedup.insert(0, "candidate_rank", range(1, len(dedup)+1))
    merged.to_csv(args.output_dir / "taosu_corrected_all_regression_positive_rows.csv.gz",
                  index=False, compression="gzip")
    dedup.to_csv(args.output_dir / "taosu_corrected_regression_positives_deduplicated.csv.gz",
                 index=False, compression="gzip")
    audit = {
        "original_positive_rows": len(old), "addendum_positive_rows": len(new),
        "combined_positive_rows": len(merged), "valid_standardized_rows": int(valid.sum()),
        "deduplicated_positive_structures": len(dedup),
        "missing_hit_id_positive_structures": int((dedup.screen_origin == "missing_hit_id_addendum").sum()),
        "rdkit_version": rdBase.rdkitVersion,
        "deduplication_key": "uncharged largest-fragment full InChIKey; stereochemistry retained",
    }
    (args.output_dir / "corrected_merge_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
