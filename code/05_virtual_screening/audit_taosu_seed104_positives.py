#!/usr/bin/env python3
"""Audit and conservatively deduplicate seed-104 TaoSu positive predictions."""

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


def structure_keys(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return pd.Series([None] * 5)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    full_key = Chem.MolToInchiKey(mol)
    try:
        parent = rdMolStandardize.FragmentParent(mol)
        parent = rdMolStandardize.Uncharger().uncharge(parent)
    except Exception:
        parent = mol
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    parent_key = Chem.MolToInchiKey(parent)
    connectivity = parent_key.split("-", 1)[0] if parent_key else None
    return pd.Series([canonical, full_key, parent_smiles, parent_key, connectivity])


def group_report(frame, key, group_type):
    repeated = frame[frame[key].notna() & frame.duplicated(key, keep=False)].copy()
    rows = []
    for value, block in repeated.groupby(key, sort=True):
        rows.append({
            "group_type": group_type, "group_key": value, "n_records": len(block),
            "hit_ids": ";".join(block.hit_id.astype(str)),
            "parent_smiles": ";".join(sorted(set(block.standardized_parent_smiles.astype(str)))),
            "mut_ic50_nM_range": f"{block.mut_ic50_nM.min():.6g}-{block.mut_ic50_nM.max():.6g}",
            "wt_ic50_nM_range": f"{block.wt_ic50_nM.min():.6g}-{block.wt_ic50_nM.max():.6g}",
            "direct_selectivity_range": f"{block.direct_selectivity_fold.min():.6g}-{block.direct_selectivity_fold.max():.6g}",
            "derived_selectivity_range": f"{block.derived_selectivity_fold.min():.6g}-{block.derived_selectivity_fold.max():.6g}",
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    key_columns = ["canonical_isomeric_smiles", "full_inchikey", "standardized_parent_smiles",
                   "standardized_parent_inchikey", "parent_connectivity_key"]
    frame[key_columns] = frame.smiles.apply(structure_keys)
    frame["minimum_selectivity_fold"] = frame[["direct_selectivity_fold",
                                                "derived_selectivity_fold"]].min(axis=1)
    frame["source_record_count"] = frame.groupby("standardized_parent_inchikey")["hit_id"].transform("size")
    frame["source_hit_ids"] = frame.groupby("standardized_parent_inchikey")["hit_id"].transform(
        lambda x: ";".join(x.astype(str)))

    duplicate_groups = pd.concat([
        group_report(frame, "standardized_parent_inchikey", "standardized_parent_exact"),
        group_report(frame, "parent_connectivity_key", "parent_connectivity_only"),
    ], ignore_index=True)
    duplicate_groups.to_csv(args.output_dir / "duplicate_structure_groups.csv", index=False)

    ranked = frame.sort_values(
        ["all_four_filters_votes", "minimum_selectivity_fold", "mut_ic50_nM",
         "wt_ic50_nM", "hit_id"], ascending=[False, False, True, False, True])
    deduplicated = ranked.drop_duplicates("standardized_parent_inchikey", keep="first").copy()
    deduplicated = deduplicated.sort_values(
        ["all_four_filters_votes", "minimum_selectivity_fold", "mut_ic50_nM"],
        ascending=[False, False, True]).reset_index(drop=True)
    deduplicated.insert(0, "candidate_rank", range(1, len(deduplicated) + 1))
    deduplicated.to_csv(args.output_dir / "taosu_seed104_potential_positives_deduplicated.csv.gz",
                        index=False, compression="gzip")
    frame.to_csv(args.output_dir / "taosu_seed104_potential_positives_structure_annotated.csv.gz",
                 index=False, compression="gzip")

    audit = {
        "input_rows": len(frame),
        "unique_hit_ids": int(frame.hit_id.nunique()),
        "duplicate_hit_id_rows": int(frame.duplicated("hit_id", keep=False).sum()),
        "unique_raw_smiles": int(frame.smiles.nunique()),
        "duplicate_raw_smiles_rows": int(frame.duplicated("smiles", keep=False).sum()),
        "unique_canonical_isomeric_smiles": int(frame.canonical_isomeric_smiles.nunique()),
        "duplicate_canonical_rows": int(frame.duplicated("canonical_isomeric_smiles", keep=False).sum()),
        "unique_standardized_parent_inchikeys": int(frame.standardized_parent_inchikey.nunique()),
        "duplicate_standardized_parent_rows": int(frame.duplicated("standardized_parent_inchikey", keep=False).sum()),
        "deduplicated_candidate_rows": len(deduplicated),
        "unique_parent_connectivity_keys": int(frame.parent_connectivity_key.nunique()),
        "rows_in_repeated_parent_connectivity_groups": int(frame.duplicated("parent_connectivity_key", keep=False).sum()),
        "invalid_smiles_in_positive_table": int(frame.standardized_parent_inchikey.isna().sum()),
        "rdkit_version": rdBase.rdkitVersion,
        "deduplication_key": "uncharged largest-fragment full InChIKey; stereochemistry retained",
        "ranking_rule": "all-four-filter fold votes descending; minimum of direct/derived selectivity descending; mutant IC50 ascending; WT IC50 descending; hit_id ascending",
        "connectivity_warning": "Connectivity groups ignore stereochemistry and are reported for review, not automatically collapsed."
    }
    (args.output_dir / "duplicate_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
