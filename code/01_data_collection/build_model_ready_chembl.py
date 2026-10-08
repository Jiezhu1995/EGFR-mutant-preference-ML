#!/usr/bin/env python3
"""Standardize extracted ChEMBL kinase measurements into the project schema."""

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

import pandas as pd
from rdkit import rdBase

from build_model_ready import standardize_smiles


MUTATION_RE = re.compile(r"[A-Z]\d+[A-Z]")
ENDPOINT_MAP = {"IC50": "IC50", "KI": "Ki", "KD": "Kd"}


def sequence_hash(sequence: str) -> str:
    clean = "".join(str(sequence).split()).upper()
    return hashlib.sha1(clean.encode()).hexdigest()[:12]


def mutation_string(value) -> str:
    hits = set(MUTATION_RE.findall(str(value).upper()))
    return ";".join(sorted(hits, key=lambda x: int(x[1:-1])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    usecols = [
        "activity_id", "compound_chembl_id", "canonical_smiles", "endpoint", "relation",
        "value_nM", "p_activity_calculated", "target_name", "target_organism",
        "uniprot_accession", "protein_sequence", "variant_id", "annotated_mutation",
        "assay_id", "assay_description", "confidence_score", "year", "pubmed_id", "doi",
        "patent_id", "document_chembl_id", "source_name", "eligible_regression",
    ]
    frames = []
    scanned = retained = 0
    for chunk in pd.read_csv(args.input, sep="\t", usecols=usecols, chunksize=args.chunksize,
                             low_memory=False):
        scanned += len(chunk)
        eligible = pd.to_numeric(chunk.eligible_regression, errors="coerce").eq(1)
        keep = chunk.loc[eligible & chunk.canonical_smiles.notna()
                         & chunk.protein_sequence.notna()].copy()
        frames.append(keep)
        retained += len(keep)
        print(f"read scanned={scanned:,} retained={retained:,}", flush=True)
    rows = pd.concat(frames, ignore_index=True)
    rows["endpoint"] = rows.endpoint.map(ENDPOINT_MAP)
    rows["p_activity"] = pd.to_numeric(rows.p_activity_calculated, errors="coerce")
    rows = rows[rows.endpoint.notna() & rows.p_activity.notna()].copy()

    unique_smiles = rows.canonical_smiles.astype(str).drop_duplicates().tolist()
    with Pool(processes=args.workers) as pool:
        chemistry_rows = list(pool.imap_unordered(standardize_smiles, unique_smiles, chunksize=250))
    chemistry_columns = [
        "original_smiles", "standardized_smiles", "nonisomeric_smiles", "standard_inchikey",
        "inchikey_connectivity", "murcko_scaffold", "molecular_weight", "logp", "tpsa",
        "hbd", "hba", "rotatable_bonds", "heavy_atoms", "standardization_status",
        "exclusion_reason",
    ]
    chemistry = pd.DataFrame(chemistry_rows, columns=chemistry_columns)
    rows = rows.merge(chemistry, left_on="canonical_smiles", right_on="original_smiles",
                      how="left", validate="many_to_one")
    exclusions = rows[rows.standardization_status.ne("ok")].copy()
    rows = rows[rows.standardization_status.eq("ok")].copy()

    rows["protein_sequence"] = rows.protein_sequence.astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    rows["construct_id"] = rows.uniprot_accession.astype(str) + "__seq_" + rows.protein_sequence.map(sequence_hash)
    rows["mutation_string"] = rows.annotated_mutation.map(mutation_string)
    rows["mutation_count"] = rows.mutation_string.map(lambda x: 0 if not x else len(x.split(";")))
    rows["is_mutant"] = rows.variant_id.notna() & rows.mutation_count.gt(0)
    # A missing ChEMBL variant annotation is not sufficient evidence of WT.
    rows["is_wild_type"] = False
    rows["input_source"] = "ChEMBL37"
    rows["record_id"] = "CHEMBL_ACTIVITY_" + rows.activity_id.astype(str)
    rows["compound_id"] = rows.compound_chembl_id
    rows["organism"] = rows.target_organism
    rows["measurement_key"] = (
        rows.inchikey_connectivity.astype(str) + "::" + rows.construct_id.astype(str)
        + "::" + rows.endpoint.astype(str)
    )

    compounds = chemistry[chemistry.standardization_status.eq("ok")].drop_duplicates("standard_inchikey")
    proteins = rows[[
        "construct_id", "uniprot_accession", "protein_sequence", "organism",
        "mutation_string", "mutation_count", "is_wild_type", "is_mutant",
    ]].drop_duplicates("construct_id")
    measurements = rows[[
        "measurement_key", "input_source", "record_id", "compound_id", "standard_inchikey",
        "inchikey_connectivity", "construct_id", "endpoint", "relation", "p_activity",
    ]].copy()
    provenance = rows[[
        "record_id", "activity_id", "assay_id", "compound_chembl_id", "construct_id",
        "variant_id", "annotated_mutation", "assay_description", "confidence_score",
        "value_nM", "endpoint", "p_activity", "year", "pubmed_id", "doi", "patent_id",
        "document_chembl_id", "source_name",
    ]].copy()

    compounds.to_csv(args.output / "compounds.tsv.gz", sep="\t", index=False)
    proteins.to_csv(args.output / "protein_constructs.tsv.gz", sep="\t", index=False)
    measurements.to_csv(args.output / "measurements_exact.tsv.gz", sep="\t", index=False)
    provenance.to_csv(args.output / "measurements_provenance.tsv.gz", sep="\t", index=False)
    if len(exclusions):
        exclusions[["activity_id", "canonical_smiles", "exclusion_reason"]].to_csv(
            args.output / "chemistry_exclusions.tsv.gz", sep="\t", index=False
        )

    audit = {
        "software": {"rdkit": rdBase.rdkitVersion, "pandas": pd.__version__},
        "input_rows": scanned,
        "eligible_input_rows": retained,
        "unique_input_smiles": len(unique_smiles),
        "valid_standardized_compounds": len(compounds),
        "chemistry_excluded_measurements": len(exclusions),
        "retained_measurements": len(measurements),
        "protein_constructs": len(proteins),
        "explicit_mutant_measurements": int(rows.is_mutant.sum()),
        "endpoint_counts": measurements.endpoint.value_counts().to_dict(),
    }
    (args.output / "model_ready_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
