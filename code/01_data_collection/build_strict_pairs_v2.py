#!/usr/bin/env python3
"""Build provenance-preserving WT-mutant pairs from v2 assay annotations."""
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


def clean(value) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None"}:
        return ""
    return str(value).strip()


def first_document(row, fields) -> str:
    for prefix, field in fields:
        value = clean(row.get(field))
        if value:
            return f"{prefix}:{value}"
    return ""


def join_unique(values) -> str:
    return ";".join(sorted({clean(v) for v in values if clean(v)}))


def prepare_chembl(annotation_path: str, exact_path: str) -> pd.DataFrame:
    annotations = pd.read_csv(annotation_path, sep="\t", low_memory=False)
    exact = pd.read_csv(exact_path, sep="\t", low_memory=False)
    data = exact.merge(
        annotations.drop(columns=["endpoint", "p_activity", "construct_id"], errors="ignore"),
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
    data["document_key"] = data.apply(
        first_document,
        axis=1,
        fields=[("DOI", "doi"), ("PMID", "pubmed_id"), ("CHEMBL_DOC", "document_chembl_id"), ("PATENT", "patent_id")],
    )
    data["source_database"] = "ChEMBL37"
    data["source_record_id"] = data.record_id
    data["source_assay_id"] = data.assay_id.astype(str)
    data["source_target_text"] = data.assay_description.fillna("")
    data["standardized_smiles"] = np.nan
    return data


def prepare_bindingdb(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, sep="\t", low_memory=False)
    data = data[data.is_exact.eq(True) & data.eligible_regression.eq(True)].copy()
    data["document_key"] = data.apply(
        first_document,
        axis=1,
        fields=[("DOI", "Article DOI"), ("PMID", "PMID"), ("PATENT", "Patent Number")],
    )
    data["source_database"] = "BindingDB"
    data["source_record_id"] = data.record_id
    data["source_assay_id"] = "not_available"
    data["source_target_text"] = data.target_name.fillna("")
    data["inchikey_connectivity"] = data.inchikey.astype(str).str.split("-").str[0]
    data["standardized_smiles"] = data.canonical_smiles
    return data


def pair_source(data: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = (
        data.document_key.ne("")
        & data.inchikey_connectivity.notna()
        & data.uniprot_accession.notna()
        & data.endpoint.isin(["IC50", "Ki", "Kd"])
        & data.relation.eq("=")
        & data.v2_state.isin(["WT", "MUTANT"])
    )
    eligible = data[required].copy()
    eligible["pair_tier"] = np.where(eligible.v2_assay_class.eq("unknown"), "B_unknown_assay_class", "A_strict")
    group = [
        "source_database", "document_key", "inchikey_connectivity", "uniprot_accession",
        "endpoint", "v2_assay_class", "pair_tier",
    ]
    wt = eligible[eligible.v2_state.eq("WT")].groupby(group, as_index=False).agg(
        p_activity_wt=("p_activity", "median"),
        n_wt_records=("p_activity", "size"),
        wt_record_ids=("source_record_id", join_unique),
        wt_assay_ids=("source_assay_id", join_unique),
        wt_label_reasons=("v2_label_reason", join_unique),
        wt_target_texts=("source_target_text", join_unique),
        standardized_smiles=("standardized_smiles", "first"),
    )
    mut_group = group + ["v2_variant_token", "v2_variant_types"]
    mutant = eligible[eligible.v2_state.eq("MUTANT")].groupby(mut_group, as_index=False).agg(
        p_activity_mut=("p_activity", "median"),
        n_mut_records=("p_activity", "size"),
        mutant_record_ids=("source_record_id", join_unique),
        mutant_assay_ids=("source_assay_id", join_unique),
        mutant_label_reasons=("v2_label_reason", join_unique),
        mutant_target_texts=("source_target_text", join_unique),
        mutant_smiles=("standardized_smiles", "first"),
    )
    pairs = mutant.merge(wt, on=group, how="inner", validate="many_to_one")
    pairs["standardized_smiles"] = pairs.standardized_smiles.fillna(pairs.mutant_smiles)
    pairs = pairs.drop(columns=["mutant_smiles"])
    pairs["delta_pactivity_mut_minus_wt"] = pairs.p_activity_mut - pairs.p_activity_wt
    pairs["same_assay_id"] = [
        bool(set(clean(a).split(";")) & set(clean(b).split(";"))) and clean(a) != "not_available"
        for a, b in zip(pairs.wt_assay_ids, pairs.mutant_assay_ids)
    ]
    excluded = data[~required].copy()
    excluded["v2_pair_exclusion_reason"] = np.select(
        [
            excluded.document_key.eq(""),
            ~excluded.endpoint.isin(["IC50", "Ki", "Kd"]),
            ~excluded.relation.eq("="),
            excluded.v2_state.eq("UNKNOWN"),
            excluded.v2_state.eq("AMBIGUOUS"),
        ],
        ["missing_document", "unsupported_endpoint", "non_exact_relation", "unknown_construct", "ambiguous_construct"],
        default="missing_required_identifier",
    )
    return pairs, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chembl-annotations", required=True)
    parser.add_argument("--chembl-exact", required=True)
    parser.add_argument("--bindingdb-annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    chembl = prepare_chembl(args.chembl_annotations, args.chembl_exact)
    bindingdb = prepare_bindingdb(args.bindingdb_annotations)
    cp, cx = pair_source(chembl, "ChEMBL37")
    bp, bx = pair_source(bindingdb, "BindingDB")
    all_pairs = pd.concat([cp, bp], ignore_index=True, sort=False)
    strict = all_pairs[all_pairs.pair_tier.eq("A_strict")].copy()
    tier_b = all_pairs[all_pairs.pair_tier.ne("A_strict")].copy()
    all_pairs.to_csv(out / "all_pair_candidates_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    strict.to_csv(out / "strict_pairs_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    tier_b.to_csv(out / "tier_b_unknown_assay_pairs_v2.tsv.gz", sep="\t", index=False, compression="gzip")
    # Keep compact exclusion inventories rather than duplicating large assay text.
    pd.concat([cx, bx], ignore_index=True, sort=False)[
        ["source_database", "source_record_id", "uniprot_accession", "endpoint", "v2_state", "v2_pair_exclusion_reason"]
    ].to_csv(out / "excluded_records_v2.tsv.gz", sep="\t", index=False, compression="gzip")

    audit = {
        "all_candidate_pairs": int(len(all_pairs)),
        "strict_pairs": int(len(strict)),
        "tier_b_unknown_assay_pairs": int(len(tier_b)),
        "strict_unique_compounds": int(strict.inchikey_connectivity.nunique()),
        "strict_kinases": int(strict.uniprot_accession.nunique()),
        "strict_mutation_tasks": int(strict.groupby(["uniprot_accession", "v2_variant_token"]).ngroups),
        "strict_egfr_pairs": int(strict.uniprot_accession.eq("P00533").sum()),
        "strict_by_source": strict.source_database.value_counts().to_dict(),
        "strict_by_endpoint": strict.endpoint.value_counts().to_dict(),
        "strict_by_assay_class": strict.v2_assay_class.value_counts().to_dict(),
    }
    (out / "pair_build_audit_v2.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
