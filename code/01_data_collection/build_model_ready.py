#!/usr/bin/env python3
"""Build model-ready Papyrus/BindingDB kinase tables with auditable chemistry."""

from __future__ import annotations

# Resolve shared modules across the six workflow stages.
import sys as _sys
from pathlib import Path as _Path
_sys.path[:0] = [str(p) for p in _Path(__file__).resolve().parents[1].glob("[0-9][0-9]_*") if p.is_dir()]

import argparse
import hashlib
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ENDPOINTS = {"IC50", "Kd", "Ki"}
MUTATION_RE = re.compile(r"^[A-Z]\d+[A-Z]$")


def standardize_smiles(smiles: str) -> tuple:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError("parse_failed")
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        Chem.GetSymmSSSR(mol)
        if mol.GetNumHeavyAtoms() == 0:
            raise ValueError("no_heavy_atoms")
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        nonisomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        inchikey = Chem.MolToInchiKey(mol)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True, isomericSmiles=False)
        return (str(smiles), canonical, nonisomeric, inchikey, inchikey.split("-")[0], scaffold,
                Descriptors.MolWt(mol), Crippen.MolLogP(mol), rdMolDescriptors.CalcTPSA(mol),
                Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol), mol.GetNumHeavyAtoms(), "ok", "")
    except Exception as exc:
        return (str(smiles), "", "", "", "", "", np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, "excluded", str(exc)[:160])


def construct_fields(row: pd.Series) -> tuple[str, str, int, bool, bool]:
    accession = str(row.get("uniprot_accession", "")).strip()
    target_id = str(row.get("target_id", "")).strip()
    sequence = str(row.get("protein_sequence", "")).strip()
    mutations: list[str] = []
    if target_id and target_id.lower() != "nan":
        parts = target_id.split("_")[1:]
        mutations = sorted({part for part in parts if MUTATION_RE.fullmatch(part)},
                           key=lambda x: int(x[1:-1]))
        is_wt = any(part.upper() == "WT" for part in parts) and not mutations
        construct_id = target_id
    else:
        is_wt = False
        digest = hashlib.sha1(sequence.encode()).hexdigest()[:12] if sequence else "noseq"
        construct_id = f"{accession}__seq_{digest}"
    mutation_string = ";".join(mutations)
    return construct_id, mutation_string, len(mutations), bool(is_wt), bool(mutations)


def read_eligible(path: Path, source: str, chunksize: int) -> tuple[pd.DataFrame, dict]:
    frames, scanned, retained = [], 0, 0
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=False):
        scanned += len(chunk)
        mask = (chunk["eligible_regression"].eq(True) & chunk["relation"].eq("=")
                & chunk["endpoint"].isin(ENDPOINTS) & chunk["p_activity"].notna()
                & chunk["canonical_smiles"].notna() & chunk["protein_sequence"].notna())
        keep = chunk.loc[mask].copy()
        keep["input_source"] = source
        frames.append(keep)
        retained += len(keep)
        print(f"{source}: scanned={scanned:,} retained={retained:,}", flush=True)
    return pd.concat(frames, ignore_index=True), {"scanned": scanned, "eligible_exact": retained}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--papyrus", type=Path, required=True)
    p.add_argument("--bindingdb", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunksize", type=int, default=200_000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    papyrus, papyrus_audit = read_eligible(args.papyrus, "Papyrus", args.chunksize)
    bindingdb, bindingdb_audit = read_eligible(args.bindingdb, "BindingDB", args.chunksize)
    rows = pd.concat([papyrus, bindingdb], ignore_index=True, sort=False)

    unique_smiles = rows["canonical_smiles"].astype(str).drop_duplicates().tolist()
    with Pool(processes=args.workers) as pool:
        chemistry_rows = list(pool.imap_unordered(standardize_smiles, unique_smiles, chunksize=250))
    chemistry_columns = [
        "original_smiles", "standardized_smiles", "nonisomeric_smiles", "standard_inchikey",
        "inchikey_connectivity", "murcko_scaffold", "molecular_weight", "logp", "tpsa",
        "hbd", "hba", "rotatable_bonds", "heavy_atoms", "standardization_status", "exclusion_reason",
    ]
    chemistry = pd.DataFrame(chemistry_rows, columns=chemistry_columns)
    rows = rows.merge(chemistry, left_on="canonical_smiles", right_on="original_smiles",
                      how="left", validate="many_to_one")
    exclusions = rows[rows["standardization_status"].ne("ok")].copy()
    rows = rows[rows["standardization_status"].eq("ok")].copy()

    construct_values = rows.apply(construct_fields, axis=1, result_type="expand")
    construct_values.columns = ["construct_id", "mutation_string", "mutation_count", "is_wild_type", "is_mutant"]
    rows = pd.concat([rows, construct_values], axis=1)
    rows["p_activity"] = pd.to_numeric(rows["p_activity"], errors="coerce")
    rows["measurement_key"] = (rows["inchikey_connectivity"].astype(str) + "::" +
                               rows["construct_id"].astype(str) + "::" + rows["endpoint"].astype(str))

    compounds = chemistry[chemistry.standardization_status.eq("ok")].drop_duplicates("standard_inchikey")
    proteins = rows[["construct_id", "uniprot_accession", "protein_sequence", "organism",
                    "mutation_string", "mutation_count", "is_wild_type", "is_mutant"]].drop_duplicates("construct_id")
    measurements = rows[["measurement_key", "input_source", "record_id", "compound_id",
                        "standard_inchikey", "inchikey_connectivity", "construct_id", "endpoint",
                        "relation", "p_activity"]].copy()

    grouped = measurements.groupby(["input_source", "inchikey_connectivity", "construct_id", "endpoint"],
                                   dropna=False)["p_activity"]
    aggregate = grouped.agg(median_p_activity="median", mean_p_activity="mean", n_measurements="size",
                            p_activity_std="std", p_activity_min="min", p_activity_max="max").reset_index()
    aggregate["p_activity_range"] = aggregate.p_activity_max - aggregate.p_activity_min
    aggregate["conflict_flag"] = aggregate.p_activity_range.gt(1.5)
    aggregate["sample_weight"] = np.where(aggregate.conflict_flag, 0.25,
                                          1.0 / np.sqrt(aggregate.n_measurements.clip(lower=1)))

    compounds.to_csv(args.output / "compounds.tsv.gz", sep="\t", index=False)
    proteins.to_csv(args.output / "protein_constructs.tsv.gz", sep="\t", index=False)
    measurements.to_csv(args.output / "measurements_exact.tsv.gz", sep="\t", index=False)
    aggregate.to_csv(args.output / "measurements_within_source_aggregated.tsv.gz", sep="\t", index=False)
    if len(exclusions):
        exclusions[["input_source", "record_id", "canonical_smiles", "exclusion_reason"]].to_csv(
            args.output / "chemistry_exclusions.tsv.gz", sep="\t", index=False)

    audit = {
        "software": {"rdkit": rdBase.rdkitVersion, "pandas": pd.__version__},
        "papyrus": papyrus_audit, "bindingdb": bindingdb_audit,
        "combined_exact_before_chemistry": len(papyrus) + len(bindingdb),
        "unique_input_smiles": len(unique_smiles), "valid_standardized_compounds": len(compounds),
        "chemistry_excluded_measurements": len(exclusions), "retained_measurements": len(measurements),
        "protein_constructs": len(proteins), "within_source_aggregates": len(aggregate),
        "conflicting_aggregates": int(aggregate.conflict_flag.sum()),
        "endpoint_counts": measurements.endpoint.value_counts().to_dict(),
    }
    (args.output / "model_ready_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
