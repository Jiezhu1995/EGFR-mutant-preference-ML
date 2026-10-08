#!/usr/bin/env python3
"""Attach audited chemistry and freeze balanced scaffold 5-fold splits."""
from __future__ import annotations

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold


CHEMISTRY_COLUMNS = [
    "inchikey_connectivity", "standardized_smiles", "murcko_scaffold",
    "molecular_weight", "logp", "tpsa", "standardization_status", "exclusion_reason",
]


def scaffold_from_smiles(smiles: str, key: str) -> tuple[str, str]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return "", "rdkit_parse_failed"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
    if not scaffold:
        scaffold = f"ACYCLIC:{key}"
    return scaffold, ""


def balanced_mapping(frame: pd.DataFrame, group_column: str, n_folds: int, seed: int) -> tuple[dict, list[int]]:
    sizes = frame.groupby(group_column).size().reset_index(name="n")
    rng = np.random.default_rng(seed)
    sizes["tie"] = rng.random(len(sizes))
    sizes = sizes.sort_values(["n", "tie"], ascending=[False, True])
    loads = [0] * n_folds
    mapping = {}
    for row in sizes.itertuples(index=False):
        fold = min(range(n_folds), key=lambda value: (loads[value], value))
        mapping[getattr(row, group_column)] = fold
        loads[fold] += int(row.n)
    return mapping, loads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--chemistry", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs, sep="\t", low_memory=False)

    chemistry_parts = []
    for rank, path in enumerate(args.chemistry):
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        frame = frame[[column for column in CHEMISTRY_COLUMNS if column in frame.columns]].copy()
        frame["chemistry_source"] = str(path)
        frame["chemistry_source_rank"] = rank
        chemistry_parts.append(frame)
    chemistry = pd.concat(chemistry_parts, ignore_index=True, sort=False)
    chemistry = chemistry.dropna(subset=["inchikey_connectivity", "standardized_smiles"])
    chemistry = chemistry[chemistry.inchikey_connectivity.isin(set(pairs.inchikey_connectivity))].copy()
    chemistry["standardized_smiles"] = chemistry.standardized_smiles.astype(str)
    variants = chemistry.groupby("inchikey_connectivity").standardized_smiles.nunique().rename("n_structure_variants")
    chemistry["computed_scaffold"] = [
        scaffold_from_smiles(smiles, key)[0]
        for smiles, key in zip(chemistry.standardized_smiles, chemistry.inchikey_connectivity)
    ]
    # Union scaffold aliases observed for the same connectivity InChIKey.  This
    # prevents tautomer/protonation standardization differences from placing an
    # alternative scaffold representation in another fold.
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for _, group in chemistry.groupby("inchikey_connectivity"):
        aliases = sorted(set(group.computed_scaffold) - {""})
        for alias in aliases[1:]:
            union(aliases[0], alias)
    # Resolve stable component names after all unions.
    scaffold_component = {value: find(value) for value in set(chemistry.computed_scaffold) - {""}}
    chemistry["scaffold_component"] = chemistry.computed_scaffold.map(scaffold_component)
    key_component = chemistry.groupby("inchikey_connectivity").scaffold_component.first()
    key_alias_count = chemistry.groupby("inchikey_connectivity").computed_scaffold.nunique().rename("n_scaffold_aliases")
    chemistry = chemistry.sort_values(
        ["inchikey_connectivity", "chemistry_source_rank", "standardized_smiles"]
    ).drop_duplicates("inchikey_connectivity", keep="first")
    chemistry = chemistry.merge(variants, on="inchikey_connectivity", how="left")
    chemistry = chemistry.merge(key_alias_count, on="inchikey_connectivity", how="left")

    pairs = pairs.drop(columns=["standardized_smiles"], errors="ignore").merge(
        chemistry,
        on="inchikey_connectivity",
        how="left",
        validate="many_to_one",
    )
    computed = [scaffold_from_smiles(smiles, key) if pd.notna(smiles) else ("", "missing_smiles")
                for smiles, key in zip(pairs.standardized_smiles, pairs.inchikey_connectivity)]
    pairs["scaffold_group"] = pairs.inchikey_connectivity.map(key_component)
    pairs["chemistry_qc_reason"] = [value[1] for value in computed]
    valid = pairs.chemistry_qc_reason.eq("")
    excluded = pairs[~valid].copy()
    final = pairs[valid].copy()

    mapping, loads = balanced_mapping(final, "scaffold_group", args.n_folds, args.seed)
    final["scaffold_fold"] = final.scaffold_group.map(mapping).astype(int)
    for fold in range(args.n_folds):
        validation_fold = (fold + 1) % args.n_folds
        column = f"unseen_compound_fold{fold}"
        final[column] = "train"
        final.loc[final.scaffold_fold.eq(validation_fold), column] = "val"
        final.loc[final.scaffold_fold.eq(fold), column] = "test"

    final.to_csv(out / "final_training_pairs_scaffold5_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    excluded.to_csv(out / "chemistry_excluded_pairs_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(sorted(mapping.items()), columns=["scaffold_group", "fold"]).to_csv(
        out / "scaffold_fold_map_v2.csv", index=False
    )
    split_rows = []
    for fold in range(args.n_folds):
        counts = final[f"unseen_compound_fold{fold}"].value_counts()
        split_rows.append({"fold": fold, **{name: int(counts.get(name, 0)) for name in ["train", "val", "test"]}})
    pd.DataFrame(split_rows).to_csv(out / "split_sizes_v2.csv", index=False)

    audit = {
        "input_pairs": int(len(pairs)),
        "chemistry_valid_pairs": int(len(final)),
        "chemistry_excluded_pairs": int(len(excluded)),
        "unique_compounds": int(final.inchikey_connectivity.nunique()),
        "compounds_with_multiple_structure_variants": int(
            final.loc[final.n_structure_variants.fillna(0).gt(1), "inchikey_connectivity"].nunique()
        ),
        "compounds_with_multiple_scaffold_aliases": int(
            final.loc[final.n_scaffold_aliases.fillna(0).gt(1), "inchikey_connectivity"].nunique()
        ),
        "scaffolds": int(final.scaffold_group.nunique()),
        "fold_loads": loads,
        "egfr_pairs": int(final.uniprot_accession.eq("P00533").sum()),
        "rdkit_version": rdBase.rdkitVersion,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "chemistry_exclusion_reasons": excluded.chemistry_qc_reason.value_counts().to_dict(),
    }
    (out / "scaffold_split_audit_v2.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
